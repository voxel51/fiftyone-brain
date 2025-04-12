"""
PGVector similarity backend.

| Copyright 2017-2025, Voxel51, Inc.
| `voxel51.com <https://voxel51.com/>`_
|
"""
import logging

import numpy as np

import eta.core.utils as etau

import fiftyone.core.utils as fou
from fiftyone.brain.similarity import (
    SimilarityConfig,
    Similarity,
    SimilarityIndex,
)
import fiftyone.brain.internal.core.utils as fbu

psycopg2 = fou.lazy_import("psycopg2")
psy_extras = fou.lazy_import("psycopg2.extras")

logger = logging.getLogger(__name__)

# Supported metrics for pgvector
_SUPPORTED_METRICS = {
    "cosine": "vector_cosine_ops",
    "dotproduct": "vector_ip_ops",
    "euclidean": "vector_l2_ops",
    "l1": "vector_l1_ops",
    "jaccard": "vector_jaccard_ops",
    "hamming": "vector_hamming_ops",
}


class PgVectorSimilarityConfig(SimilarityConfig):
    """Configuration for the PGVector similarity backend.

    Args:
        index_name (None): the name of the PGVector index to use or create.
            If none is provided, a default index name will be used.
        table_name (None): the name of the table to use or create. If none is
            provided, a default table name will be used.
        metric ("cosine"): the similarity metric to use. Supported values are
            ``("cosine", "dotproduct", "euclidean", "l1", "jaccard", "hamming")``
        connection_string (None): the connection string to the PostgreSQL database
        ssl_cert (None): the path to the SSL certificate file
        ssl_key (None): the path to the secret key used for the client certificate
        ssl_root_cert (None): the path to the file containing SSL certificate
            authority (CA) certificate(s).
        work_mem ("64MB"): the base maximum amount of memory to be used by a query operation
            (such as a sort or hash table) before writing to temporary disk files
        hnsw_m (16): the max number of connections per layer in the HNSW index
        hnsw_ef_construction (64): the size of the dynamic candidate list for constructing the graph for the HNSW index
        **kwargs: keyword arguments for
            :class:`fiftyone.brain.similarity.SimilarityConfig`
    """

    def __init__(
        self,
        index_name=None,
        table_name=None,
        metric="cosine",
        connection_string=None,
        ssl_cert=None,
        ssl_key=None,
        ssl_root_cert=None,
        work_mem="64MB",
        hnsw_m=16,
        hnsw_ef_construction=64,
        **kwargs,
    ):
        if metric not in _SUPPORTED_METRICS:
            raise ValueError(
                f"Unsupported metric '{metric}'. "
                f"Supported values are {_SUPPORTED_METRICS}"
            )

        super().__init__(**kwargs)

        self.metric = metric
        self.ssl_cert = ssl_cert
        self.ssl_key = ssl_key
        self.ssl_root_cert = ssl_root_cert
        self.work_mem = work_mem
        self.index_name = index_name
        self.table_name = table_name
        self.hnsw_m = hnsw_m
        self.hnsw_ef_construction = hnsw_ef_construction

        self._connection_string = connection_string

    @property
    def method(self):
        return "pgvector"

    @property
    def connection_string(self):
        return self._connection_string

    @connection_string.setter
    def connection_string(self, connection_string):
        self._connection_string = connection_string

    @property
    def max_k(self):
        return 10000

    @property
    def supports_least_similarity(self):
        return False

    @property
    def supported_aggregations(self):
        return ("mean",)

    def load_credentials(
        self,
        connection_string=None,
    ):
        self._load_parameters(connection_string=connection_string)


class PgVectorSimilarity(Similarity):
    """PGVector similarity factory.

    Args:
        config: a :class:`PgVectorSimilarityConfig`
    """

    def ensure_requirements(self):
        fou.ensure_package("psycopg2|psycopg2-binary")

    def ensure_usage_requirements(self):
        fou.ensure_package("psycopg2|psycopg2-binary")

    def initialize(self, samples, brain_key):
        return PgVectorSimilarityIndex(
            samples, self.config, brain_key, backend=self
        )


class PgVectorSimilarityIndex(SimilarityIndex):
    """Class for interacting with PGVector similarity indexes.

    Args:
        samples: the :class:`fiftyone.core.collections.SampleCollection` used
        config: the :class:`PGVectorSimilarityConfig` used
        brain_key: the brain key
        backend (None): a :class:`PGVectorSimilarity` instance
    """

    def __init__(self, samples, config, brain_key, backend=None):
        super().__init__(samples, config, brain_key, backend=backend)
        self._conn = None
        self._cur = None
        self._initialize()

    @property
    def total_index_size(self):
        if self._conn.closed:
            self._initialize()
        try:
            self._cur.execute(
                f"""SELECT COUNT(*) FROM "{self.config.table_name}";"""
            )
            return self._cur.fetchone()[0]
        except Exception as e:
            logger.error(f"Error getting index size: {str(e)}")
            return 0

    def _initialize(self):
        ssl_options = {}
        if self.config.ssl_cert:
            ssl_options["sslcert"] = self.config.ssl_cert
        if self.config.ssl_key:
            ssl_options["sslkey"] = self.config.ssl_key
        if self.config.ssl_root_cert:
            ssl_options["sslrootcert"] = self.config.ssl_root_cert

        logger.info(f"Connecting to PostgreSQL database")
        self._conn = psycopg2.connect(
            self.config.connection_string, **ssl_options
        )
        self._cur = self._conn.cursor()
        try:
            self._cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
            self._conn.commit()
        except Exception as e:
            logger.error(f"Error creating vector extension: {str(e)}")
            raise

        if self.config.table_name is None:
            table_names = self._get_table_names()
            root = "fiftyone-" + fou.to_slug(self.samples._root_dataset.name)
            table_name = fbu.get_unique_name(root, table_names)

            self.config.table_name = table_name
            self.save_config()
            existing_indexes = []
        else:
            existing_indexes = self._get_index_names(self.config.table_name)

        if self.config.index_name is None:
            root = "fiftyone-index-" + fou.to_slug(
                self.samples._root_dataset.name
            )
            index_name = fbu.get_unique_name(root, existing_indexes)
            self.config.index_name = index_name
            self.save_config()

    def _get_table_names(self):
        self._cur.execute(
            "SELECT table_name FROM information_schema.tables WHERE table_schema = 'public';"
        )
        return [row[0] for row in self._cur.fetchall()]

    def _get_index_names(self, table_name):
        self._cur.execute(
            f"SELECT indexname FROM pg_indexes WHERE tablename = '{table_name}' AND schemaname = 'public';"
        )
        return [row[0] for row in self._cur.fetchall()]

    def _create_table(self, dimension):
        try:
            self._cur.execute(
                f"""
                    CREATE TABLE IF NOT EXISTS "{self.config.table_name}" (
                    id TEXT PRIMARY KEY,
                    sample_id TEXT,
                    embedding_vector VECTOR({dimension})
                );
                """
            )
            self._conn.commit()
        except Exception as e:
            logger.error(
                f"Error creating table: {self.config.table_name} with dimension {dimension}: {str(e)}"
            )
            raise

    def create_hnsw_index(self):
        operator_class = _SUPPORTED_METRICS[self.config.metric]
        try:
            self._cur.execute(
                f"""DROP INDEX IF EXISTS "{self.config.index_name}";"""
            )
            self._conn.commit()
            self._cur.execute(
                f"""
                CREATE INDEX "{self.config.index_name}"
                ON "{self.config.table_name}" USING hnsw (embedding_vector {operator_class})
                WITH (m = %s, ef_construction = %s);
                """,
                (self.config.hnsw_m, self.config.hnsw_ef_construction),
            )
            self._conn.commit()
        except Exception as e:
            logger.error(
                f"Error creating HNSW index on table {self.config.table_name}:{str(e)}"
            )
            raise

    def _get_index_ids(self, batch_size=1000):
        named_cursor = self._conn.cursor(
            name="id_cursor"
        )  # Named cursor for server-side query
        named_cursor.execute(f"""SELECT id FROM "{self.config.table_name}";""")

        existing_ids = []
        while True:
            rows = named_cursor.fetchmany(batch_size)
            if not rows:
                break

            ids = [row[0] for row in rows]
            existing_ids.extend(ids)

        named_cursor.close()
        return existing_ids

    def add_to_index(
        self,
        embeddings,
        sample_ids,
        label_ids=None,
        overwrite=True,
        allow_existing=True,
        warn_existing=False,
        reload=True,
        batch_size=5000,
        close_conn=True,
    ):
        if self._conn.closed:
            self._initialize()
        self._cur.execute(f"SET work_mem TO '{self.config.work_mem}'")

        if self.config.table_name not in self._get_table_names():
            self._create_table(embeddings.shape[1])

        if label_ids is not None:
            ids = label_ids
        else:
            ids = sample_ids

        if warn_existing or not allow_existing or not overwrite:
            index_ids = self._get_index_ids()

            existing_ids = set(ids) & set(index_ids)
            num_existing = len(existing_ids)

            if num_existing > 0:
                if not allow_existing:
                    raise ValueError(
                        "Found %d IDs (eg %s) that already exist in the index"
                        % (num_existing, next(iter(existing_ids)))
                    )

                if warn_existing:
                    if overwrite:
                        logger.warning(
                            "Overwriting %d IDs that already exist in the "
                            "index",
                            num_existing,
                        )
                    else:
                        logger.warning(
                            "Skipping %d IDs that already exist in the index",
                            num_existing,
                        )
        else:
            existing_ids = set()

        if existing_ids and not overwrite:
            query = f"""
                INSERT INTO "{self.config.table_name}" (id, sample_id, embedding_vector)
                VALUES %s
                ON CONFLICT (id) DO NOTHING;
                """
        else:
            query = f"""
                INSERT INTO "{self.config.table_name}" (id, sample_id, embedding_vector)
                VALUES %s
                ON CONFLICT (id) DO UPDATE
                SET sample_id = EXCLUDED.sample_id,
                    embedding_vector = EXCLUDED.embedding_vector;
                """

        embeddings = [e.tolist() for e in embeddings]
        sample_ids = list(sample_ids)
        if label_ids is not None:
            ids = list(label_ids)
        else:
            ids = list(sample_ids)

        for _embeddings, _ids, _sample_ids in zip(
            fou.iter_batches(embeddings, batch_size),
            fou.iter_batches(ids, batch_size),
            fou.iter_batches(sample_ids, batch_size),
        ):
            data = list(zip(_ids, _sample_ids, _embeddings))
            psy_extras.execute_values(self._cur, query, data)
            self._conn.commit()

        self.create_hnsw_index()

        if close_conn:
            self.close_connections()

        if reload:
            self.reload()

    def remove_from_index(
        self,
        sample_ids=None,
        label_ids=None,
        allow_missing=True,
        warn_missing=False,
        reload=True,
    ):
        if self._conn.closed:
            self._initialize()

        if label_ids is not None:
            ids = label_ids
        else:
            ids = sample_ids

        if warn_missing or not allow_missing:
            response = self.get_embeddings_by_id(ids)
            existing_ids = [id for id, emb in response]
            missing_ids = set(ids) - set(existing_ids)
            num_missing_ids = len(missing_ids)

            if num_missing_ids > 0:
                if not allow_missing:
                    raise ValueError(
                        "Found %d IDs (eg %s) that do not exist in the index"
                        % (num_missing_ids, next(iter(missing_ids)))
                    )
                if warn_missing and not allow_missing:
                    logger.warning(
                        "Skipping %d IDs that do not exist in the index",
                        num_missing_ids,
                    )
        try:
            # Use parameterized query to delete multiple IDs
            self._cur.execute(
                f"""DELETE FROM "{self.config.table_name}" WHERE id IN %s;""",
                (tuple(ids),),
            )
        except Exception as e:
            self._conn.rollback()
            logger.error(f"Error removing embeddings for ids {ids}: {str(e)}")
            raise

        deleted_count = self._cur.rowcount
        self._conn.commit()
        logger.info(f"Deleted {deleted_count} embeddings from the index.")

        if reload:
            self.reload()

    def close_connections(self):
        if not self._cur.closed:
            self._cur.close()
        if not self._conn.closed:
            self._conn.close()

    def get_embeddings_by_id(self, sample_ids=None, label_ids=None):
        if self._conn.closed:
            self._initialize()
        if label_ids is not None:
            try:
                self._cur.execute(
                    f"""SELECT id, sample_id, embedding_vector FROM "{self.config.table_name}" WHERE id = ANY(%s)""",
                    (list(label_ids),),
                )
            except Exception as e:
                logger.error(
                    f"Error fetching embeddings for labels {label_ids}: {str(e)}"
                )
                raise
        elif sample_ids is not None:
            try:
                self._cur.execute(
                    f"""SELECT id, sample_id, embedding_vector FROM "{self.config.table_name}" WHERE sample_id = ANY(%s)""",
                    (list(sample_ids),),
                )
            except Exception as e:
                logger.error(
                    f"Error fetching embeddings for samples {sample_ids}: {str(e)}"
                )
                raise
        else:
            try:
                self._cur.execute(
                    f"""SELECT id, sample_id, embedding_vector FROM "{self.config.table_name}";"""
                )
            except Exception as e:
                logger.error(
                    f"Error fetching embeddings for all samples: {str(e)}"
                )
                raise

        results = self._cur.fetchall()
        fo_id = []
        sample_id = []
        embeddings = []
        for result in results:
            # Convert string "[1.2,3.4,5.6]" to float array
            if isinstance(result[2], str):
                emb = np.array(
                    [float(x) for x in result[2].strip("[]").split(",")],
                    dtype=np.float32,
                )
                embeddings.append(emb)
            else:
                # Already numeric
                emb = np.array(result[2], dtype=np.float32)
                embeddings.append(emb)
            fo_id.append(result[0])
            sample_id.append(result[1])

        return fo_id, sample_id, embeddings

    def get_embeddings(
        self,
        sample_ids=None,
        label_ids=None,
        allow_missing=True,
        warn_missing=False,
    ):
        if label_ids is not None:
            if self.config.patches_field is None:
                raise ValueError("This index does not support label IDs")

            if sample_ids is not None:
                logger.warning(
                    "Ignoring sample IDs when label IDs are provided"
                )

        if sample_ids is not None and self.config.patches_field is not None:
            (
                label_ids,
                found_sample_ids,
                embeddings,
            ) = self.get_embeddings_by_id(sample_ids=sample_ids)
            missing_ids = list(set(sample_ids) - set(found_sample_ids))
            sample_ids = found_sample_ids
        elif self.config.patches_field is not None:
            (
                found_label_ids,
                sample_ids,
                embeddings,
            ) = self.get_embeddings_by_id(label_ids=label_ids)
            missing_ids = (
                list(set(label_ids) - set(found_label_ids))
                if label_ids is not None
                else []
            )
            label_ids = found_label_ids
        else:
            (
                label_ids,
                found_sample_ids,
                embeddings,
            ) = self.get_embeddings_by_id(sample_ids=sample_ids)
            missing_ids = (
                list(set(sample_ids) - set(found_sample_ids))
                if sample_ids is not None
                else []
            )
            sample_ids = found_sample_ids

        num_missing_ids = len(missing_ids)
        if num_missing_ids > 0:
            if not allow_missing:
                raise ValueError(
                    "Found %d IDs (eg %s) that do not exist in the index"
                    % (num_missing_ids, missing_ids[0])
                )

            if warn_missing:
                logger.warning(
                    "Skipping %d IDs that do not exist in the index",
                    num_missing_ids,
                )

        embeddings = np.array(embeddings)
        sample_ids = np.array(sample_ids)
        if label_ids is not None:
            label_ids = np.array(label_ids)

        return embeddings, sample_ids, label_ids

    def _kneighbors(
        self,
        query=None,
        k=None,
        reverse=False,
        aggregation=None,
        return_dists=False,
        close_conn=True,
    ):
        if self._conn.closed:
            self._initialize()

        if query is None:
            raise ValueError("Postgres does not support full index neighbors")

        if aggregation not in (None, "mean"):
            raise ValueError("Unsupported aggregation '%s'" % aggregation)

        if k is None:
            k = self.index_size

        query = self._parse_neighbors_query(query)
        if aggregation == "mean" and query.ndim == 2:
            query = query.mean(axis=0)

        single_query = query.ndim == 1
        if single_query:
            query = [query]

        index_ids = None
        if self.has_view:
            if self.config.patches_field is not None:
                index_ids = list(self.current_label_ids)
            else:
                index_ids = list(self.current_sample_ids)

            _filter = True
        else:
            _filter = False

        sort_order = "DESC" if reverse else "ASC"

        sample_ids = []
        label_ids = [] if self.config.patches_field is not None else None
        dists = []
        for q in query:
            if _filter:
                self._cur.execute(
                    f"""
                    SELECT id, sample_id, embedding_vector <-> %s::vector AS distance
                    FROM "{self.config.table_name}"
                    WHERE id = ANY(%s)
                    ORDER BY distance {sort_order}
                    LIMIT %s;
                    """,
                    (q.tolist(), index_ids, k),
                )
            else:
                self._cur.execute(
                    f"""
                    SELECT id, sample_id, embedding_vector <-> %s::vector AS distance
                    FROM "{self.config.table_name}"
                    ORDER BY distance {sort_order}
                    LIMIT %s;
                    """,
                    (q.tolist(), k),
                )

            results = self._cur.fetchall()

            if self.config.patches_field is not None:
                sample_ids.append([r[1] for r in results])
                label_ids.append([r[0] for r in results])
            else:
                sample_ids.append([r[0] for r in results])

            if return_dists:
                dists.append([r[2] for r in results])

        if close_conn:
            self.close_connections()

        if single_query:
            sample_ids = sample_ids[0]
            if label_ids is not None:
                label_ids = label_ids[0]
            if return_dists:
                dists = dists[0]

        if return_dists:
            return sample_ids, label_ids, dists

        return sample_ids, label_ids

    def _parse_neighbors_query(self, query):
        if etau.is_str(query):
            query_ids = [query]
            single_query = True
        else:
            query = np.asarray(query)

            # Query by vector(s)
            if np.issubdtype(query.dtype, np.number):
                return query

            query_ids = list(query)
            single_query = False

        _, _, embeddings = self.get_embeddings_by_id(label_ids=query_ids)
        if len(embeddings) == 0:
            raise ValueError(
                "Query IDs %s do not exist in this index" % query_ids
            )
        query = np.array(embeddings)

        if single_query:
            query = query[0, :]

        return query

    def cleanup(self, drop_table=False):
        """
        Clean up the database by dropping the HNSW index and optionally the embeddings table.
        """
        logger.info(
            f"Cleaning up: Deleting HNSW index '{self.config.index_name}'"
        )
        self._cur.execute(
            f"""DROP INDEX IF EXISTS "{self.config.index_name}";"""
        )

        if self._conn.closed:
            self._initialize()

        if drop_table:
            self._cur.execute(
                f"""DROP TABLE IF EXISTS "{self.config.table_name}";"""
            )
            logger.info(
                f"{self.config.table_name} table deleted successfully."
            )

        self._conn.commit()
        # Close the database connection
        self.close_connections()
        logger.info("Database connection closed.")

    @classmethod
    def _from_dict(cls, d, samples, config, brain_key):
        return cls(samples, config, brain_key)
