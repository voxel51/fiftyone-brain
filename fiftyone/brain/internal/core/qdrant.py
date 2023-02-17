"""
Qdrant similarity backend.

| Copyright 2017-2023, Voxel51, Inc.
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
    SimilarityResults,
)
import fiftyone.brain.internal.core.utils as fbu

qdrant = fou.lazy_import("qdrant_client")
qmodels = fou.lazy_import("qdrant_client.http.models")

logger = logging.getLogger(__name__)

_METRICS = ["euclidean", "cosine", "dotproduct"]

_METRICS_TO_QDRANT = {
    "euclidean": qmodels.Distance.EUCLID,
    "cosine": qmodels.Distance.COSINE,
    "dotproduct": qmodels.Distance.DOT
}

class QdrantSimilarityConfig(SimilarityConfig):
    def __init__(
        self,
        embeddings_field=None,
        model=None,
        patches_field=None,
        supports_prompts=None,
        metric="euclidean",
        collection_name=None,
        dimension=None,
    ):
        
        if metric not in  _METRICS:
            raise ValueError(
                "metric must be one of {}".format(_METRICS)
            )
        
        metric = _METRICS_TO_QDRANT[metric]

        super().__init__(
            embeddings_field=embeddings_field,
            model=model,
            patches_field=patches_field,
            supports_prompts=supports_prompts,
            **kwargs,
        )

        collection_name = "fiftyone-collection" if collection_name is None else collection_name

        self.metric = metric
        self.collection_name = collection_name
        self.dimension = dimension

    @property
    def method(self):
        return "qdrant"
    
    #! TODO: Implement this with -1 recommendation API
    @property
    def supports_least_similarity(self):
        return True
    
    #! TODO: Implement this with recommendation API
    @property
    def supports_aggregate_queries(self):
        return True
    
class QdrantSimilarity(Similarity):
    """Qdrant similarity factory.

    Args:
        config: an :class:`QdrantSimilarityConfig`
    """

    def ensure_requirements(self):
        fou.ensure_package("qdrant-client")

    def initialize(self, samples):
        return QdrantSimilarityResults(samples, self.config, backend=self)

    def cleanup(self, samples, brain_key):
        pass

class QdrantSimilarityResults(SimilarityResults):
    """Class for interacting with Qdrant similarity indexes.

    Args:
        samples: the :class:`fiftyone.core.collections.SampleCollection` used
        config: the :class:`SimilarityConfig` used
        embeddings (None): a ``num_embeddings x num_dims`` array of embeddings
        sample_ids (None): a ``num_embeddings`` array of sample IDs
        label_ids (None): a ``num_embeddings`` array of label IDs, if
            applicable
        backend (None): a :class:`QdrantSimilarity` instance
    """

    def __init__(
        self,
        samples,
        config,
        embeddings=None,
        sample_ids=None,
        label_ids=None,
        backend=None,
    ):
        embeddings, sample_ids, label_ids = self._parse_data(
            samples,
            config,
            embeddings=embeddings,
            sample_ids=sample_ids,
            label_ids=label_ids,
        )

        dimension = self._parse_dimension(embeddings, config)

        self._embeddings = embeddings
        self._dimension = dimension
        self._sample_ids = sample_ids
        self._label_ids = label_ids
        self._collection_name = config.collection_name
        self._replication_factor = config.replication_factor
        self._shard_number = config.shard_number
        self._metric = config.metric

        self._neighbors_helper = None

        super().__init__(samples, config, backend=backend)

        @property
        def sample_ids(self):
            """The sample IDs of the full collection."""
            return self._sample_ids

        @property
        def label_ids(self):
            """The label IDs of the full collection, or ``None`` if not applicable."""
            return self._label_ids
        
        def connect_to_api(self):
            """Direct access to Qdrant API.
            """
            return qdrant
        
        @property
        def index_size(self):
            """The number of vectors in the collection."""
            return qdrant.count(self._collection_name).count
        
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

        def _parse_dimension(self, embeddings, config):
            if config.dimension is not None:
                return int(config.dimension)
            elif embeddings is not None:
                return int(embeddings.shape[1])
            return 0

        def attributes(self):
            attrs = super().attributes()

            if self.config.embeddings_field is not None:
                attrs = [
                    attr
                    for attr in attrs
                    if attr not in ("embeddings", "sample_ids", "label_ids")
                ]

            return attrs
        
        def _radius_neighbors(self, query=None, thresh=None, return_dists=False):
            raise ValueError(
                    "Qdrant backend does not support score thresholding."
                )
            pass


        #! TODO: Implement this and verify that it works
        def _kneighbors(
            self,
            query=None,
            k=None,
            reverse=False,
            keep_ids=None,
            aggregation=None,
            return_dists=False,
        ):
            return self._sort_by_similarity(
                query=query,
                k=k,
                reverse=reverse,
                aggregation=aggregation,
                return_dists=return_dists,
            )

        def _to_inds(self, ids):
            pass

        def _ensure_neighbors(self):
            pass

        def _get_neighbors(self, full=False):
            pass
        
        #? Does this work as is?
        @staticmethod
        def _parse_data(
            samples,
            config,
            embeddings=None,
            sample_ids=None,
            label_ids=None,
        ):
            if embeddings is None:
                embeddings, sample_ids, label_ids = fbu.get_embeddings(
                    samples._dataset,
                    patches_field=config.patches_field,
                    embeddings_field=config.embeddings_field,
                )
            elif sample_ids is None:
                sample_ids, label_ids = fbu.get_ids(
                    samples,
                    patches_field=config.patches_field,
                    data=embeddings,
                    data_type="embeddings",
                )

            return embeddings, sample_ids, label_ids
        
        @classmethod
        def _from_dict(cls, d, samples, config):
            embeddings = d.get("embeddings", None)
            if embeddings is not None:
                embeddings = np.array(embeddings)

            sample_ids = d.get("sample_ids", None)
            if sample_ids is not None:
                sample_ids = np.array(sample_ids)

            label_ids = d.get("label_ids", None)
            if label_ids is not None:
                label_ids = np.array(label_ids)

            config_attrs = [
                "collection_name",
                "replication_factor",
                "shard_number",
                "metric",
            ]

            for attr in config_attrs:
                if attr in d:
                    value = d.get(attr, None)
                    if value is not None:
                        config[attr] = value

            return cls(
                samples,
                config,
                embeddings=embeddings,
                sample_ids=sample_ids,
                label_ids=label_ids,
            )