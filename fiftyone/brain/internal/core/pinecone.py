"""
Piencone similarity backend.

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
    SimilarityIndex,
)
import fiftyone.brain.internal.core.utils as fbu

pinecone = fou.lazy_import("pinecone")


logger = logging.getLogger(__name__)

_METRICS = ["euclidean", "cosine", "dotproduct"]


class PineconeSimilarityConfig(SimilarityConfig):
    """Configuration for the Pinecone similarity backend.

    Args:
        embeddings_field (None): the sample field containing the embeddings,
            if one was provided
        model (None): the :class:`fiftyone.core.models.Model` or class name of
            the model that was used to compute embeddings, if one was provided
        patches_field (None): the sample field defining the patches being
            analyzed, if any
        supports_prompts (False): whether this run supports prompt queries
        metric ("euclidean"): the embedding distance metric to use. Supported
            values are ``("euclidean", "cosine", and "dotproduct")``
        index_name ("fiftyone-index"): the name of the Pinecone Index to use
    """

    def __init__(
        self,
        embeddings_field=None,
        model=None,
        patches_field=None,
        supports_prompts=None,
        metric="euclidean",
        index_name="fiftyone-index",
        dimension=None,
        pod_type="p1",
        pods=1,
        replicas=1,
        api_key=None,
        environment=None,
        namespace=None,
        **kwargs,
    ):
        if metric not in _METRICS:
            raise ValueError(
                "Unsupported metric '%s'. Supported values are %s"
                % (metric, _METRICS)
            )

        super().__init__(
            embeddings_field=embeddings_field,
            model=model,
            patches_field=patches_field,
            supports_prompts=supports_prompts,
            **kwargs,
        )

        self.metric = metric
        self.index_name = index_name
        self.dimension = dimension
        self.pod_type = pod_type
        self.pods = pods
        self.replicas = replicas
        self.api_key = api_key
        self.environment = environment
        self.namespace = namespace

    @property
    def method(self):
        return "pinecone"

    @property
    def supports_least_similarity(self):
        return False

    @property
    def supports_aggregate_queries(self):
        return False

    @property
    def max_k(self):
        return 10000  # Pinecone limit


class PineconeSimilarity(Similarity):
    """Pinecone similarity factory.

    Args:
        config: an :class:`PineconeSimilarityConfig`
    """

    def ensure_requirements(self):
        fou.ensure_package("pinecone-client")

    def initialize(self, samples):
        return PineconeSimilarityIndex(samples, self.config, backend=self)

    def cleanup(self, samples, brain_key):
        pass


class PineconeSimilarityIndex(SimilarityIndex):
    """Class for interacting with Pinecone similarity indexes.

    Args:
        samples: the :class:`fiftyone.core.collections.SampleCollection` used
        config: the :class:`SimilarityConfig` used
        backend (None): a :class:`PineconeSimilarity` instance
    """

    def __init__(self, samples, config, backend=None):
        super().__init__(samples, config, backend=backend)

        self._dimension = config.dimension or 0
        self._index_name = config.index_name
        self._pod_type = config.pod_type
        self._pods = config.pods
        self._replicas = config.replicas
        self._metric = config.metric
        self._api_key = config.api_key
        self._environment = config.environment
        self._namespace = config.namespace
        self._max_k = config.max_k
        self._index = self._initialize_index()

    # TODO: unneeded?
    def _initialize_connection(self):
        pinecone.init(self._api_key, self._environment)

    def _initialize_index(self):
        pinecone.init(self._api_key, self._environment)

        if self._index_name not in pinecone.list_indexes():
            pinecone.create_index(
                self._index_name,
                dimension=self._dimension,
                metric=self._metric,
                pod_type=self._pod_type,
                pods=self._pods,
                replicas=self._replicas,
                namespace=self._namespace,
            )

        return pinecone.Index(self._index_name)

    @property
    def index(self):
        """The ``pinecone.Index`` instance for this index."""
        return self._index

    @property
    def total_index_size(self):
        index_stats = self._index.describe_index_stats()
        return index_stats["total_vector_count"]

    def add_to_index(
        self,
        embeddings,
        sample_ids,
        label_ids=None,
        overwrite=True,
        allow_existing=True,
        warn_existing=False,
        upsert_pagination=100,
        namespace=None,
    ):
        embeddings_list = [arr.tolist() for arr in embeddings]
        if label_ids is not None:
            id_dicts = [
                {"id": lid, "sample_id": sid}
                for lid, sid in zip(label_ids, sample_ids)
            ]
            index_vectors = list(zip(label_ids, embeddings_list, id_dicts))
        else:
            id_dicts = [{"id": sid, "sample_id": sid} for sid in sample_ids]
            index_vectors = list(zip(sample_ids, embeddings_list, id_dicts))

        num_vectors = embeddings.shape[0]
        num_steps = int(np.ceil(num_vectors / upsert_pagination))

        # self._initialize_connection()
        index = self._index

        num_existing_ids = 0

        for i in range(num_steps):
            min_ind = upsert_pagination * i
            max_ind = min(upsert_pagination * (i + 1), num_vectors)

            if overwrite and allow_existing and not warn_existing:
                # Simplest case
                index.upsert(
                    index_vectors[min_ind:max_ind], namespace=namespace
                )
            else:
                curr_index_vectors = index_vectors[min_ind:max_ind]
                curr_index_vector_ids = [r[0] for r in curr_index_vectors]
                response = index.fetch(curr_index_vector_ids)
                curr_existing_ids = list(response.vectors.keys())
                num_existing_ids += len(curr_existing_ids)
                if num_existing_ids > 0 and not allow_existing:
                    raise ValueError(
                        "Found %d IDs (eg %s) that already exist in the index"
                        % (num_existing_ids, curr_existing_ids[0])
                    )
                elif not overwrite:
                    # Pick out non-existing vectors to add
                    curr_index_vectors = [
                        civ
                        for civ in curr_index_vectors
                        if civ[0] not in curr_existing_ids
                    ]

                index.upsert(
                    curr_index_vectors,
                    namespace=namespace,
                )

        if warn_existing and num_existing_ids > 0:
            logger.warning(
                "Ignoring %d IDs that already exist in the index",
                num_existing_ids,
            )

    def remove_from_index(
        self,
        sample_ids=None,
        label_ids=None,
        allow_missing=True,
        warn_missing=False,
    ):
        # self._initialize_connection()
        index = self._index

        if label_ids is not None:
            ids_to_remove = label_ids
        else:
            ids_to_remove = sample_ids

        if not allow_missing or warn_missing:
            existing_ids = index.fetch(ids_to_remove).vectors.keys()
            missing_ids = set(existing_ids) - set(ids_to_remove)
            num_missing = len(missing_ids)

            if num_missing > 0:
                if not allow_missing:
                    raise ValueError(
                        "Found %d IDs (eg %s) that are not present in the "
                        "index" % (num_missing, missing_ids[0])
                    )

                if warn_missing:
                    logger.warning(
                        "Ignoring %d IDs that are not present in the index",
                        num_missing,
                    )

        index.delete(ids=ids_to_remove)

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

    def _sort_by_similarity(
        self, query, k, reverse, aggregation, return_dists
    ):
        if reverse == True:
            raise ValueError(
                "Pinecone backend does not support least similarity queries"
            )

        if k is None or k > 10000:
            raise ValueError(
                "Must have k <= 10000 when using pinecone similiarity"
            )

        if query is None:
            raise ValueError(
                "A query must be provided when using pinecone similarity"
            )

        if aggregation is not None:
            logger.warning("Ignoring unsupported aggregation parameter")

        sample_ids = self.current_sample_ids
        label_ids = self.current_label_ids

        # self._initialize_connection()
        index = self._index

        if isinstance(query, np.ndarray):
            # Query by vectors
            query_embedding = query.tolist()
        elif etau.is_container(query) and etau.is_numeric(query[0]):
            query_embedding = np.array(query).tolist()
        else:
            query_id = query
            query_embedding = index.fetch(query_id)["vectors"][query_id[0]][
                "values"
            ]

        if label_ids is not None:
            response = index.query(
                vector=query_embedding,
                top_k=min(k, self._max_k),
                filter={"id": {"$in": list(label_ids)}},
            )
        else:
            response = index.query(
                vector=query_embedding,
                top_k=min(k, self._max_k),
                filter={"id": {"$in": list(sample_ids)}},
            )

        ids = [r["id"] for r in response["matches"]]

        if return_dists:
            dists = [r["score"] for r in response["matches"]]
            return ids, dists

        return ids

    @classmethod
    def _from_dict(cls, d, samples, config):
        return cls(samples, config)
