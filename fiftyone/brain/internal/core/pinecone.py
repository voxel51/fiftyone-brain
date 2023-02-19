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

pinecone = fou.lazy_import("pinecone")


logger = logging.getLogger(__name__)

_METRICS = ("euclidean", "cosine", "dotproduct")


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
        index_name ("fiftyone-index"): the name of the Pinecone Index to use
        metric ("euclidean"): the embedding distance metric to use. Supported
            values are ``("euclidean", "cosine", and "dotproduct")``
    """

    def __init__(
        self,
        embeddings_field=None,
        model=None,
        patches_field=None,
        supports_prompts=None,
        index_name="fiftyone-index",
        metric="euclidean",
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

        self.index_name = index_name
        self.metric = metric
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
    def max_k(self):
        return 10000  # Pinecone limit

    @property
    def supports_least_similarity(self):
        return False

    @property
    def supported_aggregations(self):
        return ("mean",)


class PineconeSimilarity(Similarity):
    """Pinecone similarity factory.

    Args:
        config: a :class:`PineconeSimilarityConfig`
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
        config: the :class:`PineconeSimilarityConfig` used
        backend (None): a :class:`PineconeSimilarity` instance
    """

    def __init__(self, samples, config, backend=None):
        super().__init__(samples, config, backend=backend)

        self._metric = config.metric
        self._index_name = config.index_name
        self._dimension = config.dimension or 0
        self._pod_type = config.pod_type
        self._pods = config.pods
        self._replicas = config.replicas
        self._api_key = config.api_key
        self._environment = config.environment
        self._namespace = config.namespace

        self._initialize_index()

    def _get_index(self):
        # @todo shouldn't we be able to avoid calling this every time?
        pinecone.init(self._api_key, self._environment)

        return pinecone.Index(self._index_name)

    def _initialize_index(self):
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

    @property
    def index(self):
        """The ``pinecone.Index`` instance for this index."""
        return self._get_index()

    @property
    def total_index_size(self):
        index_stats = self._get_index().describe_index_stats()
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

        index = self._get_index()
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

                if not overwrite:
                    # Pick out non-existing vectors to add
                    curr_index_vectors = [
                        civ
                        for civ in curr_index_vectors
                        if civ[0] not in curr_existing_ids
                    ]

                index.upsert(curr_index_vectors, namespace=namespace)

        if warn_existing and num_existing_ids > 0:
            logger.warning(
                "Skipped %d IDs that already exist in the index",
                num_existing_ids,
            )

    def remove_from_index(
        self,
        sample_ids=None,
        label_ids=None,
        allow_missing=True,
        warn_missing=False,
    ):
        index = self._get_index()

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
        aggregation=None,
        return_dists=False,
    ):
        if query is None:
            raise ValueError("Pinecone does not support full index neighbors")

        if reverse == True:
            raise ValueError(
                "Pinecone does not support least similarity queries"
            )

        if k is None or k > 10000:
            raise ValueError("Pincone requires k <= 10000")

        if aggregation not in (None, "mean"):
            raise ValueError("Unsupported aggregation '%s'" % aggregation)

        index = self._get_index()

        query = self._parse_neighbors_query(query, index)
        if aggregation == "mean" and query.ndim == 2:
            query = query.mean(axis=0)

        single_query = query.ndim == 1
        if single_query:
            query = [query]

        if self.config.patches_field is not None:
            index_ids = self.current_label_ids
        else:
            index_ids = self.current_sample_ids

        _filter = {"id": {"$in": list(index_ids)}}

        # @todo batch queries?
        ids = []
        dists = []
        for q in query:
            response = index.query(vector=q.tolist(), top_k=k, filter=_filter)
            ids.append([r["id"] for r in response["matches"]])
            if return_dists:
                dists.append([r["score"] for r in response["matches"]])

        if single_query:
            ids = ids[0]
            if return_dists:
                dists = dists[0]

        if return_dists:
            return ids, dists

        return ids

    def _parse_neighbors_query(self, query, index):
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

        # Query by ID(s)
        response = index.fetch(query_ids)["vectors"]
        query = np.array([response[_id]["values"] for _id in query_ids])

        if single_query:
            query = query[0, :]

        return query

    @classmethod
    def _from_dict(cls, d, samples, config):
        return cls(samples, config)
