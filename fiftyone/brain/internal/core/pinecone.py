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
        self.environment = environment
        self.namespace = namespace

        # store privately so not serialized
        self._api_key = api_key

    @property
    def api_key(self):
        return self._api_key

    @api_key.setter
    def api_key(self, value):
        self._api_key = value

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

    def ensure_usage_requirements(self):
        fou.ensure_package("pinecone-client")

    def initialize(self, samples):
        return PineconeSimilarityIndex(samples, self.config, backend=self)


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

    def load_credentials(self, api_key=None):
        """Load the Pinecone credentials from the given keyword arguments or the
        FiftyOne Brain similarity config.

        Args:
            api_key (None): the api_key for accessing the Pinecone index
        """
        self._load_config_parameters(api_key=api_key)

    def _create_index(self, dimension):
        pinecone.create_index(
            self._index_name,
            dimension=dimension,
            metric=self._metric,
            pod_type=self._pod_type,
            pods=self._pods,
            replicas=self._replicas,
        )

    def _initialize_index(self):
        pinecone.init(api_key=self._api_key, environment=self._environment)

        if self.config.dimension is not None:
            if self._index_name not in pinecone.list_indexes():
                self._create_index(self.config.dimension)
            self._index = pinecone.Index(self._index_name)
        else:
            self._index = None

    @property
    def index(self):
        """The ``pinecone.Index`` instance for this index."""
        return self._index

    @property
    def total_index_size(self):
        index_stats = self.index.describe_index_stats()
        return index_stats["total_vector_count"]

    @property
    def index_size(self):
        if self.config.patches_field is not None:
            index_ids = self.current_label_ids
        else:
            index_ids = self.current_sample_ids

        index_size = 0
        batch_size = 1000
        num_ids = len(index_ids)
        num_batches = np.ceil(num_ids / batch_size).astype(int)

        for batch in range(num_batches):
            min_ind = batch * batch_size
            max_ind = min((batch + 1) * batch_size, num_ids)
            batch_ids = list(index_ids[min_ind:max_ind])

            index_size += len(self.index.fetch(ids=batch_ids)["vectors"])

        return index_size

    @property
    def missing_size(self):
        if self.config.patches_field is not None:
            index_ids = self.current_label_ids
        else:
            index_ids = self.current_sample_ids

        missing_size = 0
        batch_size = 1000
        num_ids = len(index_ids)
        num_batches = np.ceil(num_ids / batch_size).astype(int)

        for batch in range(num_batches):
            min_ind = batch * batch_size
            max_ind = min((batch + 1) * batch_size, num_ids)
            batch_ids = list(index_ids[min_ind:max_ind])
            num_ids = len(batch_ids)
            num_ids_found = len(self.index.fetch(ids=batch_ids)["vectors"])
            missing_size += num_ids - num_ids_found

        return missing_size

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

        if self._index_name not in pinecone.list_indexes():
            dimension = embeddings.shape[1]
            self._create_index(dimension)
        if self.index is None:
            self._index = pinecone.Index(self._index_name)

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
        num_steps = np.ceil(num_vectors / upsert_pagination).astype(int)

        index = self.index
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
        index = self.index

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

    def _get_sample_embeddings(
        self,
        sample_ids,
        batch_size=1000,
        allow_missing=True,
        warn_missing=False,
    ):
        found_sample_ids, found_embeddings = [], []
        found_label_ids = None
        index = self.index

        missing_ids = []

        num_sample_ids = len(sample_ids)
        num_batches = np.ceil(num_sample_ids / batch_size).astype(int)

        for batch in range(num_batches):
            min_ind = batch * batch_size
            max_ind = min((batch + 1) * batch_size, num_sample_ids)
            batch_ids = sample_ids[min_ind:max_ind]

            response = index.fetch(ids=batch_ids)["vectors"]
            curr_found_ids = list(response.keys())
            curr_missing_ids = list(set(batch_ids).difference(curr_found_ids))
            missing_ids.append(curr_missing_ids)

            curr_found_embeddings = [
                response[k]["values"] for k in curr_found_ids
            ]

            found_sample_ids.append(curr_found_ids)
            found_embeddings.append(curr_found_embeddings)

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

        return found_embeddings, found_sample_ids, found_label_ids

    def _get_patch_embeddings_from_label_ids(
        self,
        label_ids,
        batch_size=1000,
        allow_missing=True,
        warn_missing=False,
    ):
        found_sample_ids, found_label_ids, found_embeddings = [], [], []
        index = self.index

        missing_ids = []

        num_sample_ids = len(label_ids)
        num_batches = np.ceil(num_sample_ids / batch_size).astype(int)

        for batch in range(num_batches):
            min_ind = batch * batch_size
            max_ind = min((batch + 1) * batch_size, num_sample_ids)
            batch_ids = label_ids[min_ind:max_ind]

            response = index.fetch(ids=batch_ids)["vectors"]
            curr_found_label_ids = list(response.keys())
            curr_missing_ids = list(
                set(batch_ids).difference(curr_found_label_ids)
            )
            missing_ids.append(curr_missing_ids)

            curr_found_embeddings = [
                response[k]["values"] for k in curr_found_label_ids
            ]

            curr_found_sample_ids = [
                response[k]["metadata"]["sample_id"]
                for k in curr_found_label_ids
            ]

            found_label_ids.append(curr_found_label_ids)
            found_sample_ids.append(curr_found_sample_ids)
            found_embeddings.append(curr_found_embeddings)

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

        return found_embeddings, found_sample_ids, found_label_ids

    def _get_patch_embeddings_from_sample_ids(
        self,
        sample_ids,
        batch_size=100,
        allow_missing=True,
        warn_missing=False,
    ):

        found_sample_ids, found_label_ids, found_embeddings = [], [], []

        index = self.index
        query_vector = [0.0] * self.index.describe_index_stats().dimension

        num_sample_ids = len(sample_ids)
        num_batches = np.ceil(num_sample_ids / batch_size).astype(int)

        for batch in range(num_batches):
            min_ind = batch * batch_size
            max_ind = min((batch + 1) * batch_size, num_sample_ids)
            batch_ids = sample_ids[min_ind:max_ind]
            _filter = {
                "sample_id": {"$in": batch_ids},
            }
            response = index.query(
                vector=query_vector,
                filter=_filter,
                top_k=min(batch_size, self.config.max_k),
                include_values=True,
                include_metadata=True,
            )

            for res in response["matches"]:
                found_label_ids.append(res["id"])
                found_sample_ids.append(res["metadata"]["sample_id"])
                found_embeddings.append(res["values"])

        missing_ids = list(set(sample_ids).difference(set(found_sample_ids)))

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

        return found_embeddings, found_sample_ids, found_label_ids

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
            return self._get_patch_embeddings_from_sample_ids(
                sample_ids,
                allow_missing=allow_missing,
                warn_missing=warn_missing,
            )
        elif self.config.patches_field is not None:
            return self._get_patch_embeddings_from_label_ids(
                label_ids,
                allow_missing=allow_missing,
                warn_missing=warn_missing,
            )
        else:
            return self._get_sample_embeddings(
                sample_ids,
                allow_missing=allow_missing,
                warn_missing=warn_missing,
            )

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

        if k is None or k > self.config.max_k:
            raise ValueError("Pincone requires k <= 10000")

        if aggregation not in (None, "mean"):
            raise ValueError("Unsupported aggregation '%s'" % aggregation)

        index = self.index

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
