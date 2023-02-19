"""
Qdrant similarity backend.

| Copyright 2017-2023, Voxel51, Inc.
| `voxel51.com <https://voxel51.com/>`_
|
"""
import logging

import fiftyone.core.utils as fou
from fiftyone.brain.similarity import (
    SimilarityConfig,
    Similarity,
    SimilarityIndex,
)

qdrant = fou.lazy_import("qdrant_client")
qmodels = fou.lazy_import("qdrant_client.http.models")


logger = logging.getLogger(__name__)

_METRICS = {
    "euclidean": qmodels.Distance.EUCLID,
    "cosine": qmodels.Distance.COSINE,
    "dotproduct": qmodels.Distance.DOT,
}


class QdrantSimilarityConfig(SimilarityConfig):
    def __init__(
        self,
        embeddings_field=None,
        model=None,
        patches_field=None,
        supports_prompts=None,
        metric="euclidean",
        collection_name="fiftyone-collection",
        dimension=None,
        **kwargs,
    ):
        if metric not in _METRICS:
            raise ValueError(
                "Unsupported metric '%s'. Supported values are %s"
                % (metric, tuple(_METRICS.keys()))
            )

        super().__init__(
            embeddings_field=embeddings_field,
            model=model,
            patches_field=patches_field,
            supports_prompts=supports_prompts,
            **kwargs,
        )

        self.metric = metric
        self.collection_name = collection_name
        self.dimension = dimension

    @property
    def method(self):
        return "qdrant"

    #! TODO: Do we need a maximum here?
    @property
    def max_k(self):
        return None

    #! TODO: Implement this with -1 recommendation API
    @property
    def supports_least_similarity(self):
        return True

    @property
    def supported_aggregations(self):
        return ("mean",)


class QdrantSimilarity(Similarity):
    """Qdrant similarity factory.

    Args:
        config: a :class:`QdrantSimilarityConfig`
    """

    def ensure_requirements(self):
        fou.ensure_package("qdrant-client")

    def initialize(self, samples):
        return QdrantSimilarityIndex(samples, self.config, backend=self)

    def cleanup(self, samples, brain_key):
        pass


class QdrantSimilarityIndex(SimilarityIndex):
    """Class for interacting with Qdrant similarity indexes.

    Args:
        samples: the :class:`fiftyone.core.collections.SampleCollection` used
        config: the :class:`QdrantSimilarityConfig` used
        backend (None): a :class:`QdrantSimilarity` instance
    """

    def __init__(self, samples, config, backend=None):
        super().__init__(samples, config, backend=backend)

        self._metric = _METRICS[config.metric]
        self._dimension = config.dimension or 0
        self._collection_name = config.collection_name
        self._replication_factor = config.replication_factor
        self._shard_number = config.shard_number

    @property
    def total_index_size(self):
        return qdrant.count(self._collection_name).count

    #! TODO: Implement this
    def add_to_index(
        self,
        embeddings,
        sample_ids,
        label_ids=None,
        overwrite=True,
        allow_existing=True,
        warn_existing=False,
        upsert_pagination=100,
    ):
        pass

    #! TODO: Implement this
    def remove_from_index(
        self,
        sample_ids=None,
        label_ids=None,
        allow_missing=True,
        warn_missing=False,
    ):
        pass

    #! TODO: Implement this
    def _kneighbors(
        self,
        query=None,
        k=None,
        reverse=False,
        aggregation=None,
        return_dists=False,
    ):
        pass

    @classmethod
    def _from_dict(cls, d, samples, config):
        return cls(samples, config)
