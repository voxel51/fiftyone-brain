"""
PGVector similarity backend.

| Copyright 2017-2024, Voxel51, Inc.
| `voxel51.com <https://voxel51.com/>`_
|
"""
import logging
import numpy as np
import psycopg2
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
        metric=None,
        **kwargs,
    ):
        if metric is not None and metric not in _SUPPORTED_METRICS:
            raise ValueError(
                "Unsupported metric '%s'. Supported values are %s" 
                % (metric, _SUPPORTED_METRICS)
            )
        super().__init__(**kwargs)
        self._connection_string = connection_string
        self.metric = metric

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
            self._conn = psycopg2.connect(self.config.connection_string)
            self._cur = self._conn.cursor()
        except Exception as e:
            raise ValueError(
                "Failed to connect to PostgreSQL database. "
                "Check your connection string and ensure the database is running"
            ) from e

    def add_to_index(self, embeddings, sample_ids, **kwargs):
        try:
            self._cur.execute('''
                CREATE TABLE IF NOT EXISTS embeddings (
                    id TEXT PRIMARY KEY,
                    embedding vector(%s)
                )
            ''', (len(embeddings[0]),))
            
            for embedding, id in zip(embeddings, sample_ids):
                self._cur.execute(
                    'INSERT INTO embeddings (id, embedding) VALUES (%s, %s::vector) ON CONFLICT (id) DO UPDATE SET embedding = EXCLUDED.embedding',
                    (id, embedding.tolist())
                )
            self._conn.commit()
        except Exception as e:
            self._conn.rollback()
            raise e

    def _kneighbors(self, query, k=None, **kwargs):
        try:
            self._cur.execute('''
                SELECT id, embedding <-> %s::vector AS distance
                FROM embeddings
                ORDER BY distance
                LIMIT %s
            ''', (query.tolist(), k))
            results = self._cur.fetchall()
            return [r[0] for r in results], [r[1] for r in results]
        except Exception as e:
            self._conn.rollback()
            raise e

    def cleanup(self):
        if self._cur is not None:
            self._cur.close()
        if self._conn is not None:
            self._conn.close()

    @property
    def total_index_size(self):
        try:
            self._cur.execute('SELECT COUNT(*) FROM embeddings')
            return self._cur.fetchone()[0]
        except:
            return 0

    @classmethod
    def _from_dict(cls, d, samples, config, brain_key):
        return cls(samples, config, brain_key)

