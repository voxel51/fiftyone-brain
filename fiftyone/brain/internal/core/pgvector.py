"""
PGVector similarity backend.
"""

import os
import logging
import uuid
import re
import numpy as np
import psycopg2
from psycopg2.errors import UndefinedColumn
from dotenv import load_dotenv
from fiftyone.brain.similarity import (
    SimilarityConfig,
    Similarity,
    SimilarityIndex,
)
import fiftyone.core.utils as fou

logger = logging.getLogger(__name__)

# Supported metrics for pgvector
_SUPPORTED_METRICS = ("cosine", "dotproduct", "euclidean", "l1", "jaccard", "hamming")

# Load environment variables
load_dotenv()

class PgVectorSimilarityConfig(SimilarityConfig):
    def __init__(
        self,
        connection_string=None,
        metric="cosine",
        ssl_cert=None,
        ssl_key=None,
        ssl_root_cert=None,
        work_mem="64MB",
        index_name="embedding_hnsw_index",
        embedding_column="clip_embeddings",
        patches_field=None,
        roi_field=None,
        model=None,
        model_kwargs=None,
        supports_prompts=True,
        **kwargs,
    ):
        if metric not in _SUPPORTED_METRICS:
            raise ValueError(
                f"Unsupported metric '{metric}'. "
                f"Supported values are {_SUPPORTED_METRICS}"
            )

        self.metric = metric
        self.ssl_cert = ssl_cert
        self.ssl_key = ssl_key
        self.ssl_root_cert = ssl_root_cert
        self.work_mem = work_mem
        self.index_name = index_name
        self.embedding_column = embedding_column
        self.patches_field = patches_field
        self.roi_field = roi_field
        self.model = model
        self.model_kwargs = model_kwargs or {}
        self.supports_prompts = supports_prompts
        self._connection_string = connection_string or os.getenv("PGVECTOR_CONNECTION_STRING")

    @property
    def method(self):
        return "pgvector"

    @property
    def connection_string(self):
        return self._connection_string

    @connection_string.setter
    def connection_string(self, value):
        self._connection_string = value

    def serialize(self, **kwargs):
        return {
            "cls": "fiftyone.brain.internal.core.pgvector.PgVectorSimilarityConfig",
            "connection_string": self._connection_string,
            "metric": self.metric,
            "ssl_cert": self.ssl_cert,
            "ssl_key": self.ssl_key,
            "ssl_root_cert": self.ssl_root_cert,
            "work_mem": self.work_mem,
            "index_name": self.index_name,
            "embedding_column": self.embedding_column,
            "patches_field": self.patches_field,
            "roi_field": self.roi_field,
            "model": self.model,
            "model_kwargs": self.model_kwargs,
            "supports_prompts": self.supports_prompts,
        }

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
        fou.ensure_package("psycopg2-binary")

    def ensure_usage_requirements(self):
        fou.ensure_package("psycopg2-binary")

    def initialize(self, samples, brain_key):
        return PgVectorSimilarityIndex(
            samples, self.config, brain_key, backend=self
        )

class PgVectorSimilarityIndex(SimilarityIndex):
    def __init__(self, samples, config, brain_key, backend=None, embedding_model=None):
        super().__init__(samples, config, brain_key, backend=backend)
        self._conn = None
        self._cur = None
        self.embedding_model = embedding_model
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

            logger.info(f"Connecting to PostgreSQL database: {self.config.connection_string}")
            self._conn = psycopg2.connect(
                self.config.connection_string, **ssl_options
            )
            self._cur = self._conn.cursor()

            logger.info("Ensuring pgvector extension is installed...")
            self._cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
            self._conn.commit()

            self._cur.execute("SELECT current_database();")
            db_name = self._cur.fetchone()[0]
            logger.info(f"Connected to database: {db_name}")

        except Exception as e:
            if self._conn:
                self._conn.close()
            raise ValueError(
                f"Failed to connect to PostgreSQL database: {str(e)}. "
                "Check your connection string and ensure PostgreSQL is running."
            ) from e

    def add_to_index(
        self,
        embeddings,
        sample_ids,
        overwrite=True,
        allow_existing=True,
        warn_existing=False,
        reload=True,
        **kwargs,
    ):
        try:
            logger.info("Adding embeddings to the database...")
            self._cur.execute(f"SET work_mem TO '{self.config.work_mem}'")
            n_dimensions = embeddings.shape[1]

            if overwrite:
                logger.info("Dropping existing embeddings table.")
                self._cur.execute("DROP TABLE IF EXISTS embeddings;")

            logger.info(f"Creating embeddings table with dimension: {n_dimensions}")
            self._cur.execute(
                f'''
                CREATE TABLE IF NOT EXISTS embeddings (
                    id TEXT PRIMARY KEY,
                    {self.config.embedding_column} VECTOR({n_dimensions})
                );
                '''
            )

            batch_size = 1000
            for i in range(0, len(sample_ids), batch_size):
                batch_ids = sample_ids[i : i + batch_size]
                batch_embeddings = embeddings[i : i + batch_size]

                values = [
                    (sid, emb.tolist())
                    for sid, emb in zip(batch_ids, batch_embeddings)
                ]

                logger.info(f"Inserting batch of size {len(values)} into database.")

                self._cur.executemany(
                    f'''
                    INSERT INTO embeddings (id, {self.config.embedding_column})
                    VALUES (%s, %s::vector)
                    ON CONFLICT (id) DO NOTHING;
                    ''',
                    values
                )

            self._conn.commit()
            self.create_hnsw_index()
            self._cur.execute("RESET work_mem")

        except Exception as e:
            self._conn.rollback()
            logger.error(f"Error adding embeddings: {str(e)}")
            raise

    def create_hnsw_index(self, m=16, ef_construction=64):
        try:
            metric_map = {
                "cosine": "vector_cosine_ops",
                "dotproduct": "vector_ip_ops",
                "euclidean": "vector_l2_ops",
                "l1": "vector_l1_ops",
                "jaccard": "vector_jaccard_ops",
                "hamming": "vector_hamming_ops",
            }

            if self.config.metric not in metric_map:
                raise ValueError(f"Unsupported metric: {self.config.metric}")

            operator_class = metric_map[self.config.metric]

            logger.info(f"Creating HNSW index '{self.config.index_name}' with metric: {self.config.metric}")
            self._cur.execute(
                f'''
                CREATE INDEX IF NOT EXISTS {self.config.index_name}
                ON embeddings USING hnsw ({self.config.embedding_column} {operator_class})
                WITH (m = %s, ef_construction = %s);
                ''',
                (m, ef_construction)
            )
            self._conn.commit()

        except Exception as e:
            logger.error(f"Error creating HNSW index: {str(e)}")
            raise

    def get_embedding_by_id(self, sample_id):
        try:
            self._cur.execute(
                f"SELECT {self.config.embedding_column} FROM embeddings WHERE id = %s",
                (sample_id,)
            )
            result = self._cur.fetchone()
            if result:
                # Convert string "[1.2,3.4,5.6]" to float array
                if isinstance(result[0], str):
                    arr = np.array(
                        [float(x) for x in result[0].strip('[]').split(',')],
                        dtype=np.float32
                    )
                    return arr
                else:
                    # Already numeric
                    return np.array(result[0], dtype=np.float32)

            raise ValueError(f"Sample ID {sample_id} not found in DB")

        except Exception as e:
            logger.error(f"Error fetching embedding for {sample_id}: {str(e)}")
            raise

    def _kneighbors(
        self,
        query=None,
        k=None,
        reverse=False,
        aggregation=None,
        return_dists=False,
    ):
        """
        Perform k-NN search on the embeddings table.
        """
        try:
            if k is None:
                k = 20

            # Handle case where query is a single-element list
            if isinstance(query, list) and len(query) == 1:
                query = query[0]

            # Distinguish query by type
            if isinstance(query, str):
                # Check if it's either a valid sample ID or treat as text
                if self._is_valid_sample_id(query):
                    query_embedding = self.get_embedding_by_id(query)
                else:
                    query_embedding = self.text_to_embedding(query)
            elif isinstance(query, (list, np.ndarray)):
                query_embedding = np.array(query, dtype=np.float32)
            else:
                raise ValueError(f"Invalid query type: {type(query)}")

            # Ensure embedding is a numpy array
            if not isinstance(query_embedding, np.ndarray):
                raise ValueError(
                    f"Retrieved embedding is not a numpy array: {type(query_embedding)}"
                )

            # Ensure embedding is float32 and 1-D
            if query_embedding.dtype != np.float32:
                query_embedding = query_embedding.astype(np.float32)

            if query_embedding.ndim != 1:
                query_embedding = query_embedding.ravel()  # Flatten to 1-D

            # Convert to list for PostgreSQL compatibility
            query_vector = query_embedding.tolist()

            sort_order = "DESC" if reverse else "ASC"

            # Execute k-NN search
            self._cur.execute(
                f"""
                SELECT id, {self.config.embedding_column} <-> %s::vector AS distance
                FROM embeddings
                ORDER BY distance {sort_order}
                LIMIT %s;
                """,
                (query_vector, k),
            )

            results = self._cur.fetchall()
            ids = [r[0] for r in results]
            distances = [r[1] for r in results]

            return (ids, distances) if return_dists else ids

        except Exception as e:
            logger.error(f"Query failed: {str(e)}")
            raise
        
    def _is_valid_sample_id(self, value: str) -> bool:
        """
        Returns True if `value` is either a valid 36-char UUID
        or a valid 24-char MongoDB ObjectID (hex).
        """
        # Check standard UUID
        try:
            uuid.UUID(value)
            return True
        except ValueError:
            pass

        # Check 24-hex pattern (Mongo ObjectID)
        if len(value) == 24 and re.fullmatch(r"[0-9a-fA-F]{24}", value):
            return True

        return False

    def cleanup(self, drop_table=False):
        """
        Clean up the database by dropping the HNSW index and optionally the embeddings table.
        """
        try:
            logger.info(f"Cleaning up: Deleting HNSW index '{self.config.index_name}'...")
            self._cur.execute(f"DROP INDEX IF EXISTS {self.config.index_name};")
            logger.info(f"HNSW index '{self.config.index_name}' deleted successfully.")

            if drop_table:
                self._cur.execute("DROP TABLE IF EXISTS embeddings;")
                logger.info("Embeddings table deleted successfully.")

            self._conn.commit()
        except Exception as e:
            logger.error("Error during cleanup operation. Rolling back transaction.")
            self._conn.rollback()
            raise e
        finally:
            # Close the database connection
            if self._cur:
                self._cur.close()
            if self._conn:
                self._conn.close()
            logger.info("Database connection closed.")

    @property
    def total_index_size(self):
        try:
            self._cur.execute("SELECT COUNT(*) FROM embeddings;")
            return self._cur.fetchone()[0]
        except Exception as e:
            logger.error(f"Error getting index size: {str(e)}")
            return 0

    def remove_from_index(
        self,
        sample_ids=None,
        label_ids=None,
        allow_missing=True,
        warn_missing=False,
        reload=True,
    ):
        """Removes embeddings from the PostgreSQL database by IDs."""
        try:
            if self._conn.closed:
                self._initialize()  # Reconnect if closed

            logger.info(f"Removing {len(sample_ids)} embeddings from index...")
            
            # Use parameterized query to delete multiple IDs
            self._cur.execute(
                f"DELETE FROM embeddings WHERE id IN %s;",
                (tuple(sample_ids),)
            )
            
            deleted_count = self._cur.rowcount
            self._conn.commit()
            
            logger.info(f"Successfully removed {deleted_count} entries")
            
            if warn_missing and deleted_count != len(sample_ids):
                missing = len(sample_ids) - deleted_count
                logger.warning(f"{missing} IDs not found in index")

        except Exception as e:
            self._conn.rollback()
            logger.error(f"Error removing embeddings: {str(e)}")
            if not allow_missing:
                raise            
