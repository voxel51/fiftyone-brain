"""
PGVector similarity backend.

| Copyright 2017-2024, Voxel51, Inc.
| `voxel51.com <https://voxel51.com/>`_
"""
import os
from dotenv import load_dotenv

import logging
import numpy as np
import psycopg2
from psycopg2.errors import UndefinedColumn
import eta.core.utils as etau
from eta.core.serial import Serializable
import fiftyone.core.utils as fou
from fiftyone.brain.similarity import (
    SimilarityConfig,
    Similarity,
    SimilarityIndex,
)
import fiftyone.brain.internal.core.utils as fbu
import ast

pgvector = fou.lazy_import("pgvector")
logger = logging.getLogger(__name__)

# Supported metrics for pgvector
_SUPPORTED_METRICS = ("cosine", "dotproduct", "euclidean", "l1", "jaccard", "hamming")

# Load environment variables from .env file
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
        embedding_column="embedding",
        patches_field=None,  # Add patches_field parameter
        roi_field=None,
        model=None,
        model_kwargs=None,
        supports_prompts=True,
        **kwargs,
    ):
        if metric not in _SUPPORTED_METRICS:
            raise ValueError(
                f"Unsupported metric '{metric}'. Supported values are {_SUPPORTED_METRICS}"
            )
        self.metric = metric
        self.ssl_cert = ssl_cert
        self.ssl_key = ssl_key
        self.ssl_root_cert = ssl_root_cert
        self.work_mem = work_mem
        self.index_name = index_name
        self.embedding_column = embedding_column
        self.patches_field = patches_field  # Store patches_field
        self.roi_field = roi_field
        self.model = model
        self.model_kwargs = model_kwargs or {}
        self.supports_prompts = supports_prompts

        # Store privately to avoid serialization
        self._connection_string = connection_string or os.getenv("PG_VECTOR_CONNECTION_STRING")

        logger.info(f"Using metric: {self.metric}")

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
        """Serialize the configuration."""
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
            "patches_field": self.patches_field,  # Include patches_field in serialization
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
        """Ensure that the required packages are installed."""
        fou.ensure_package("pgvector")
        fou.ensure_package("psycopg2-binary")

    def ensure_usage_requirements(self):
        """Ensure that the brain result saved from a run can be used."""
        fou.ensure_package("pgvector")
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
        self.embedding_model = embedding_model  # Store the embedding model
        self._initialize()

    def text_to_embedding(self, text):
        """Convert a text string to an embedding using the provided model."""
        if not self.embedding_model:
            raise ValueError("No embedding model provided for text-to-embedding conversion.")
        return self.embedding_model.encode(text)

    @classmethod
    def _from_dict(cls, d, samples, config, brain_key):
        """Builds a :class:`PgVectorSimilarityIndex` from a JSON representation of it."""
        if "connection_string" in d:
            config.connection_string = d["connection_string"]
        index = cls(samples, config, brain_key)
        index._initialize()  # Re-establish the database connection
        return index

    def _initialize(self):
        """Initialize the database connection and cursor."""
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

            self._cur.execute("SELECT current_database();")
            db_name = self._cur.fetchone()[0]
            logger.info(f"Connected to database: {db_name}")

        except Exception as e:
            raise ValueError(
                "Failed to connect to PostgreSQL database. "
                "Check your connection string and ensure the database is running."
            ) from e

    @property
    def total_index_size(self):
        """The total number of embeddings in the PostgreSQL database."""
        try:
            self._cur.execute("SELECT COUNT(*) FROM embeddings")
            result = self._cur.fetchone()
            return result[0] if result else 0
        except Exception as e:
            logger.error(f"Error fetching total index size: {e}")
            raise

    def get_embeddings(self):
        """Fetch all embeddings from the database."""
        try:
            logger.info("Fetching embeddings from the database...")
            self._cur.execute(f"SELECT id, {self.config.embedding_column} FROM embeddings")
            return {row[0]: np.array(row[1]) for row in self._cur.fetchall()}
        except Exception as e:
            logger.error("Failed to fetch embeddings.")
            raise RuntimeError("Failed to fetch embeddings") from e

    def remove_from_index(self, sample_ids):
        """Remove embeddings from the database for the given sample IDs and their associated patches."""
        try:
            logger.info(f"Removing {len(sample_ids)} samples and their patches from the index...")
            
            # Remove sample embeddings
            for sample_id in sample_ids:
                self._cur.execute("DELETE FROM embeddings WHERE id = %s", (sample_id,))
            
            # Remove patch embeddings (if patches_field is specified)
            if self.config.patches_field:
                for sample_id in sample_ids:
                    # Remove patches associated with the sample
                    self._cur.execute("DELETE FROM embeddings WHERE id LIKE %s", (f"{sample_id}_patch%",))
            
            self._conn.commit()
            logger.info("Samples and patches removed successfully.")
        except Exception as e:
            logger.error("Error removing samples and patches. Rolling back transaction.")
            self._conn.rollback()
            raise e

    def cleanup(self, drop_table=False):
        """Clean up the database by dropping the HNSW index and optionally the embeddings table."""
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
        """Add embeddings to the database and create an HNSW index."""
        try:
            logger.info("Adding embeddings to the database...")
            logger.info(f"Setting work_mem to {self.config.work_mem} for index creation")
            self._cur.execute(f"SET work_mem TO '{self.config.work_mem}'")

            n_dimensions = embeddings.shape[1]
            logger.info(f"Ensuring database column '{self.config.embedding_column}' has {n_dimensions} dimensions...")

            # Drop the existing table and recreate it to ensure no duplicates
            if overwrite:
                self._cur.execute("DROP TABLE IF EXISTS embeddings;")
                self._cur.execute(
                    f'''
                    CREATE TABLE IF NOT EXISTS embeddings (
                        id TEXT PRIMARY KEY,
                        {self.config.embedding_column} VECTOR({n_dimensions})
                    );
                    '''
                )
                logger.info(f"Created new embeddings table in database: {self._get_current_database()}")

            # Batch insert embeddings
            batch_size = 1000  # Adjust batch size as needed
            for i in range(0, len(sample_ids), batch_size):
                batch_ids = sample_ids[i:i + batch_size]
                batch_embeddings = embeddings[i:i + batch_size]
                values = [(id, embedding.tolist()) for id, embedding in zip(batch_ids, batch_embeddings)]
                logger.info(f"Inserting batch {i // batch_size + 1} with {len(values)} embeddings...")
                self._cur.executemany(
                    f'''
                    INSERT INTO embeddings (id, {self.config.embedding_column})
                    VALUES (%s, %s::vector);
                    ''',
                    values,
                )
                logger.info(f"Batch {i // batch_size + 1} inserted successfully.")

            logger.info("Committing changes to the database...")
            self._conn.commit()

            # Verify that embeddings were inserted
            self._cur.execute("SELECT COUNT(*) FROM embeddings")
            count = self._cur.fetchone()[0]
            logger.info(f"Total embeddings in the database: {count}")

            # Create HNSW index if it doesn't already exist
            m = kwargs.get("m", 16)  # Default value for m
            ef_construction = kwargs.get("ef_construction", 64)  # Default value for ef_construction
            self.create_hnsw_index(m=m, ef_construction=ef_construction)

            logger.info("Resetting work_mem to default")
            self._cur.execute("RESET work_mem")

        except Exception as e:
            logger.error("Error during add_to_index operation. Rolling back transaction.")
            self._conn.rollback()
            raise e

    def create_hnsw_index(self, m=16, ef_construction=64):
        """Create an HNSW index on the embeddings table."""
        try:
            logger.info(f"Creating HNSW index '{self.config.index_name}' with metric: {self.config.metric}")

            metric_to_operator_class = {
                "cosine": "vector_cosine_ops",
                "dotproduct": "vector_ip_ops",
                "euclidean": "vector_l2_ops",
                "l1": "vector_l1_ops",
                "jaccard": "vector_jaccard_ops",
                "hamming": "vector_hamming_ops",
            }

            query_metric_op = metric_to_operator_class.get(self.config.metric)
            
            if query_metric_op is None:
                raise ValueError(f"Unsupported metric: {self.config.metric}")

            self._cur.execute(
                f'''
                CREATE INDEX IF NOT EXISTS {self.config.index_name}
                ON embeddings USING hnsw ({self.config.embedding_column} {query_metric_op})
                WITH (m = %s, ef_construction = %s);
                ''',
                (m, ef_construction),
            )
            
            logger.info(f"HNSW index '{self.config.index_name}' created successfully with metric: {self.config.metric}")
        
        except Exception as e:
            logger.error(f"Error creating HNSW index: {e}")
            raise

    def get_embedding_by_id(self, sample_id):
        """Fetch the embedding for a given sample ID."""
        try:
            self._cur.execute(f"SELECT {self.config.embedding_column} FROM embeddings WHERE id = %s", (sample_id,))
            result = self._cur.fetchone()
            if result:
                # Parse the string representation of the embedding into a list of floats
                embedding_str = result[0]
                if isinstance(embedding_str, str):
                    embedding_list = ast.literal_eval(embedding_str)  # Safely evaluate the string as a list
                else:
                    embedding_list = embedding_str  # Assume it's already a list
                return np.array(embedding_list, dtype=np.float32)
            raise ValueError(f"Sample ID {sample_id} not found in the database.")
        except Exception as e:
            logger.error(f"Error fetching embedding for sample ID {sample_id}: {e}")
            raise

    def text_to_embedding(self, text):
        """Convert a text string to an embedding using the provided model."""
        if not self.embedding_model:
            raise ValueError("No embedding model provided for text-to-embedding conversion.")
        return self.embedding_model.encode(text)

    def _get_current_database(self):
        """Get the name of the current database."""
        try:
            self._cur.execute("SELECT current_database();")
            return self._cur.fetchone()[0]
        except Exception as e:
            logger.error(f"Error fetching current database name: {e}")
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
        Perform k-NN search. Query can be a sample ID, text string, or numeric vector.
        """
        try:
            if k is None:
                k = 20  # Default value for k
                logger.info(f"k not specified. Using default value: {k}")

            if isinstance(query, str):
                if query.isnumeric():  # Assume it's a sample ID
                    query_embedding = self.get_embedding_by_id(query)
                else:  # Assume it's a text string
                    query_embedding = self.text_to_embedding(query)
            elif isinstance(query, (list, np.ndarray)):  # Numeric vector
                query_embedding = np.array(query)
            else:
                raise ValueError("Unsupported query type. Must be sample ID, text, or numeric vector.")

            if query_embedding.ndim > 1:
                query_embedding = query_embedding.flatten()

            query_embedding_list = query_embedding.tolist()
            query_str = ",".join(map(str, query_embedding_list))

            logger.info(f"Performing k-NN search with top-{k} neighbors...")

            sort_order = "DESC" if reverse else "ASC"

            self._cur.execute(
                f"""
                SELECT id, {self.config.embedding_column} <-> ARRAY[{query_str}]::vector AS distance
                FROM embeddings ORDER BY distance {sort_order} LIMIT %s;
                """,
                (k,),
            )

            results = self._cur.fetchall()
            ids = [r[0] for r in results]
            distances = [r[1] for r in results]

            if return_dists:
                return ids, distances
            return ids

        except Exception as e:
            logger.error(f"Error during k-NN search: {e}")
            raise
