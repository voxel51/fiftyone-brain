"""
PGVector similarity backend.

| Copyright 2017-2024, Voxel51, Inc.
| `voxel51.com <https://voxel51.com/>`_
|
"""
import logging
import numpy as np
import psycopg2
from psycopg2.errors import UndefinedColumn
import eta.core.utils as etau
import fiftyone.core.utils as fou
from fiftyone.brain.similarity import (
    SimilarityConfig,
    Similarity,
    SimilarityIndex,
)
import fiftyone.brain.internal.core.utils as fbu

pgvector = fou.lazy_import("pgvector")
logger = logging.getLogger(__name__)

_SUPPORTED_METRICS = ("cosine", "dotproduct", "euclidean")


class PgVectorSimilarityConfig(SimilarityConfig):
    def __init__(
        self,
        connection_string=None,
        metric="cosine",
        ssl_cert=None,
        ssl_key=None,
        ssl_root_cert=None,
        work_mem="64MB",  # Default work_mem value
        **kwargs,
    ):
        if metric not in _SUPPORTED_METRICS:
            raise ValueError(
                f"Unsupported metric '{metric}'. Supported values are {_SUPPORTED_METRICS}"
            )
        super().__init__(**kwargs)
        self._connection_string = connection_string
        self.metric = metric
        self.ssl_cert = ssl_cert
        self.ssl_key = ssl_key
        self.ssl_root_cert = ssl_root_cert
        self.work_mem = work_mem

    @property
    def method(self):
        return "pgvector"

    @property
    def connection_string(self):
        return self._connection_string

    @connection_string.setter
    def connection_string(self, value):
        self._connection_string = value

    @property
    def max_k(self):
        return 10000

    @property
    def supports_least_similarity(self):
        return False

    @property
    def supported_aggregations(self):
        return ("mean",)


class PgVectorSimilarity(Similarity):
    def ensure_requirements(self):
        fou.ensure_package("pgvector")
        fou.ensure_package("psycopg2-binary")

    def initialize(self, samples, brain_key):
        return PgVectorSimilarityIndex(
            samples, self.config, brain_key, backend=self
        )

class PgVectorSimilarityIndex(SimilarityIndex):
    def __init__(self, samples, config, brain_key, backend=None):
        super().__init__(samples, config, brain_key, backend=backend)
        self._conn = None
        self._cur = None
        self._initialize()

    def _initialize(self):
        try:
            ssl_options = {}
            if self.config.ssl_cert:
                ssl_options["sslcert"] = self.config.ssl_cert
            if self.config.ssl_key:
                ssl_options["sslkey"] = self.config.ssl_key
            if self.config.ssl_root_cert:
                ssl_options["sslrootcert"] = self.config.ssl_root_cert

            self._conn = psycopg2.connect(
                self.config.connection_string, **ssl_options
            )
            self._cur = self._conn.cursor()
        except Exception as e:
            raise ValueError(
                "Failed to connect to PostgreSQL database. "
                "Check your connection string and ensure the database is running."
            ) from e

    def get_embeddings(self):
        try:
            logger.info("Fetching embeddings from the database...")
            self._cur.execute("SELECT id, embedding FROM embeddings")
            return {row[0]: np.array(row[1]) for row in self._cur.fetchall()}
        except Exception as e:
            logger.error("Failed to fetch embeddings.")
            raise RuntimeError("Failed to fetch embeddings") from e

    def remove_from_index(self, sample_ids):
        try:
            logger.info(f"Removing {len(sample_ids)} samples from the index...")
            for sample_id in sample_ids:
                self._cur.execute("DELETE FROM embeddings WHERE id = %s", (sample_id,))
            self._conn.commit()
            logger.info("Samples removed successfully.")
        except Exception as e:
            logger.error("Error removing samples. Rolling back transaction.")
            self._conn.rollback()
            raise e

    def cleanup(self):
        try:
            logger.info("Cleaning up: Deleting HNSW index and embeddings table...")

            # Drop the HNSW index if it exists
            index_name = "embedding_hnsw_index"
            self._cur.execute(f"DROP INDEX IF EXISTS {index_name};")
            logger.info(f"HNSW index '{index_name}' deleted successfully.")

            # Optionally, drop the embeddings table
            self._cur.execute("DROP TABLE IF EXISTS embeddings;")
            logger.info("Embeddings table deleted successfully.")

            # Commit changes to the database
            self._conn.commit()
        except Exception as e:
            logger.error("Error during cleanup operation. Rolling back transaction.")
            self._conn.rollback()
            raise e

    def add_to_index(self, embeddings, sample_ids, m=16, ef_construction=64, **kwargs):
        try:
            logger.info("Adding embeddings to the database...")
            logger.info(f"Setting work_mem to {self.config.work_mem} for index creation")
            self._cur.execute(f"SET work_mem TO '{self.config.work_mem}'")

            n_dimensions = embeddings.shape[1]
            logger.info(f"Ensuring database column 'embedding' has {n_dimensions} dimensions...")
            
            # Create table or resize column dynamically if needed
            self._cur.execute(
                f'''
                CREATE TABLE IF NOT EXISTS embeddings (
                    id TEXT PRIMARY KEY,
                    embedding VECTOR({n_dimensions})
                )
                '''
            )
            
            try:
                self._cur.execute(
                    f"ALTER TABLE embeddings ALTER COLUMN embedding TYPE vector({n_dimensions});"
                )
                logger.info(f"Resized column 'embedding' to {n_dimensions} dimensions.")
            
            except UndefinedColumn:
                logger.warning("Column resizing skipped: column does not exist or already matches dimensions.")

            for embedding, id in zip(embeddings, sample_ids):
                self._cur.execute(
                    '''
                    INSERT INTO embeddings (id, embedding)
                    VALUES (%s, %s::vector)
                    ON CONFLICT (id) DO UPDATE SET embedding = EXCLUDED.embedding;
                    ''',
                    (id, embedding.tolist()),
                )
            
            logger.info("Committing changes to the database...")
            self._conn.commit()

            # Create HNSW index if it doesn't already exist
            self.create_hnsw_index(m=m, ef_construction=ef_construction)

            logger.info("Resetting work_mem to default")
            self._cur.execute("RESET work_mem")
        
        except Exception as e:
            logger.error("Error during add_to_index operation. Rolling back transaction.")
            self._conn.rollback()
            raise e

    def create_hnsw_index(self, m=16, ef_construction=64):
        try:
            logger.info("Creating HNSW index...")

            metric_to_operator_class = {
                "cosine": "vector_cosine_ops",
                "dotproduct": "vector_ip_ops",
                "euclidean": "vector_l2_ops",
            }

            query_metric_op = metric_to_operator_class.get(self.config.metric)
            
            if query_metric_op is None:
                raise ValueError(f"Unsupported metric: {self.config.metric}")

            index_name = "embedding_hnsw_index"
            
            self._cur.execute(
                f'''
                CREATE INDEX IF NOT EXISTS {index_name}
                ON embeddings USING hnsw (embedding {query_metric_op})
                WITH (m = %s, ef_construction = %s);
                ''',
                (m, ef_construction),
            )
            
            logger.info("HNSW index created successfully.")
        
        except Exception as e:
            logger.error(f"Error creating HNSW index: {e}")
            raise

    def _kneighbors(self, query, k=None, return_dists=False, **kwargs):
        try:
            query_metric_op = f"{self.config.metric}_ops"
            
            logger.info(f"Performing k-NN search with top-{k} neighbors...")
            
            query_str = ",".join(map(str, query))
            
            self._cur.execute(
                f'''
                SELECT id, embedding <-> ARRAY[{query_str}]::vector AS distance
                FROM embeddings ORDER BY distance ASC LIMIT %s;
                ''',
                (k,),
            )

            results = self._cur.fetchall()
            
            ids = [r[0] for r in results]
            distances = [r[1] for r in results]

            if return_dists:
                return ids, distances
            return ids
        
        except Exception as e:
            logger.error("Error during k-NN search.")
