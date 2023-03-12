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

_SUPPORTED_METRICS = ("cosine", "dotproduct", "euclidean")


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
        index_name ("fiftyone-index"): the name of a Pinecone index to use or
            create
        index_type (None): the index type to use when creating a new index
        namespace (None): a namespace under which to store vectors added to the
            index
        metric (None): the embedding distance metric to use when creating a
            new index. Supported values are
            ``("cosine", "dotproduct", "euclidean")``
        replicas (None): an optional number of replicas when creating a new
            index
        shards (None): an optional number of shards when creating a new index
        pods (None): an optional number of pods when creating a new index
        pod_type (None): an optional pod type when creating a new index
        environment (None): a Pinecone environment to use
        api_key (None): a Pinecone API key to use
    """

    def __init__(
        self,
        embeddings_field=None,
        model=None,
        patches_field=None,
        supports_prompts=None,
        index_name="fiftyone-index",
        index_type=None,
        namespace=None,
        metric=None,
        replicas=None,
        shards=None,
        pods=None,
        pod_type=None,
        environment=None,
        api_key=None,
        **kwargs,
    ):
        if metric is not None and metric not in _SUPPORTED_METRICS:
            raise ValueError(
                "Unsupported metric '%s'. Supported values are %s"
                % (metric, _SUPPORTED_METRICS)
            )

        super().__init__(
            embeddings_field=embeddings_field,
            model=model,
            patches_field=patches_field,
            supports_prompts=supports_prompts,
            **kwargs,
        )

        self.index_name = index_name
        self.index_type = index_type
        self.namespace = namespace
        self.metric = metric
        self.replicas = replicas
        self.shards = shards
        self.pods = pods
        self.pod_type = pod_type

        # store privately so these aren't serialized
        self._environment = environment
        self._api_key = api_key

    @property
    def method(self):
        return "pinecone"

    @property
    def environment(self):
        return self._environment

    @environment.setter
    def environment(self, value):
        self._environment = value

    @property
    def api_key(self):
        return self._api_key

    @api_key.setter
    def api_key(self, value):
        self._api_key = value

    @property
    def max_k(self):
        return 10000  # Pinecone limit

    @property
    def supports_least_similarity(self):
        return False

    @property
    def supported_aggregations(self):
        return ("mean",)

    def load_credentials(self, environment=None, api_key=None):
        self._load_parameters(environment=environment, api_key=api_key)


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

        self._index = None
        self._pinecone_index_size = None
        self._pinecone_missing_size = None

        self._initialize()

    def _initialize(self):
        pinecone.init(
            environment=self.config.environment,
            api_key=self.config.api_key,
        )

        try:
            index_names = pinecone.list_indexes()
        except Exception as e:
            # @todo update help link once integration docs are available
            raise ValueError(
                "Failed to connect to Pinecone backend at environment '%s'. "
                "Refer to https://docs.voxel51.com for more information"
                % self.config.environment
            ) from e

        if self.config.index_name in index_names:
            index = pinecone.Index(self.config.index_name)
        else:
            index = None

        self._index = index

    def _create_index(self, dimension):
        kwargs = dict(
            index_type=self.config.index_type,
            metric=self.config.metric,
            replicas=self.config.replicas,
            shards=self.config.shards,
            pods=self.config.pods,
            pod_type=self.config.pod_type,
        )
        kwargs = {k: v for k, v in kwargs.items() if v is not None}

        pinecone.create_index(
            self.config.index_name,
            dimension=dimension,
            **kwargs,
        )

        self._index = pinecone.Index(self.config.index_name)

    @property
    def index(self):
        """The ``pinecone.Index`` instance for this index."""
        return self._index

    @property
    def total_index_size(self):
        if self._index is None:
            return None

        index_stats = self._index.describe_index_stats()
        return index_stats["total_vector_count"]

    @property
    def index_size(self):
        if self._index is None:
            return None

        if self._pinecone_index_size is None:
            self._compute_current_index_size()

        return self._pinecone_index_size

    @property
    def missing_size(self):
        if self._index is None:
            return None

        if self._pinecone_missing_size is None:
            self._compute_current_index_size()

        return self._pinecone_missing_size

    def add_to_index(
        self,
        embeddings,
        sample_ids,
        label_ids=None,
        overwrite=True,
        allow_existing=True,
        warn_existing=False,
        batch_size=100,
        namespace=None,
    ):
        if namespace is None:
            namespace = self.config.namespace

        if self._index is None:
            self._create_index(embeddings.shape[1])

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
        num_batches = np.ceil(num_vectors / batch_size).astype(int)

        num_existing_ids = 0

        for i in range(num_batches):
            min_ind = batch_size * i
            max_ind = min(batch_size * (i + 1), num_vectors)

            if overwrite and allow_existing and not warn_existing:
                # Simplest case
                self._index.upsert(
                    index_vectors[min_ind:max_ind], namespace=namespace
                )
            else:
                curr_index_vectors = index_vectors[min_ind:max_ind]
                curr_index_vector_ids = [r[0] for r in curr_index_vectors]
                response = self._index.fetch(curr_index_vector_ids)
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

                self._index.upsert(curr_index_vectors, namespace=namespace)

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
        if label_ids is not None:
            ids = label_ids
        else:
            ids = sample_ids

        if not allow_missing or warn_missing:
            existing_ids = self._index.fetch(ids).vectors.keys()
            missing_ids = set(existing_ids) - set(ids)
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

        self._index.delete(ids=ids)

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
                embeddings,
                sample_ids,
                label_ids,
                missing_ids,
            ) = self._get_patch_embeddings_from_sample_ids(sample_ids)
        elif self.config.patches_field is not None:
            (
                embeddings,
                sample_ids,
                label_ids,
                missing_ids,
            ) = self._get_patch_embeddings_from_label_ids(label_ids)
        else:
            (
                embeddings,
                sample_ids,
                label_ids,
                missing_ids,
            ) = self._get_sample_embeddings(sample_ids)

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

    def use_view(self, *args, **kwargs):
        self._pinecone_index_size = None
        self._pinecone_missing_size = None
        return super().use_view(*args, **kwargs)

    def _compute_current_index_size(self):
        if self.config.patches_field is not None:
            index_ids = self.current_label_ids
        else:
            index_ids = self.current_sample_ids

        index_size = 0
        missing_size = 0
        for batch_ids in fou.iter_batches(index_ids, 1000):
            num_found = len(self._index.fetch(ids=batch_ids)["vectors"])
            index_size += num_found
            missing_size += len(batch_ids) - num_found

        self._pinecone_index_size = index_size
        self._pinecone_missing_size = missing_size

    def _get_sample_embeddings(self, sample_ids, batch_size=1000):
        found_embeddings = []
        found_sample_ids = []
        missing_ids = []

        for batch_ids in fou.iter_batches(sample_ids, batch_size):
            response = self._index.fetch(ids=batch_ids)["vectors"]

            curr_found_ids = list(response.keys())
            curr_found_embeddings = [
                response[k]["values"] for k in curr_found_ids
            ]
            curr_missing_ids = list(set(batch_ids) - set(curr_found_ids))

            found_embeddings.append(curr_found_embeddings)
            found_sample_ids.append(curr_found_ids)
            missing_ids.extend(curr_missing_ids)

        return found_embeddings, found_sample_ids, None, missing_ids

    def _get_patch_embeddings_from_label_ids(self, label_ids, batch_size=1000):
        found_embeddings = []
        found_sample_ids = []
        found_label_ids = []
        missing_ids = []

        for batch_ids in fou.iter_batches(label_ids, batch_size):
            response = self._index.fetch(ids=batch_ids)["vectors"]

            curr_found_label_ids = list(response.keys())
            curr_found_embeddings = [
                response[k]["values"] for k in curr_found_label_ids
            ]
            curr_found_sample_ids = [
                response[k]["metadata"]["sample_id"]
                for k in curr_found_label_ids
            ]
            curr_missing_ids = list(set(batch_ids) - set(curr_found_label_ids))
            missing_ids.extend(curr_missing_ids)

            found_embeddings.extend(curr_found_embeddings)
            found_sample_ids.extend(curr_found_sample_ids)
            found_label_ids.extend(curr_found_label_ids)

        return found_embeddings, found_sample_ids, found_label_ids, missing_ids

    def _get_patch_embeddings_from_sample_ids(
        self, sample_ids, batch_size=100
    ):
        found_embeddings = []
        found_sample_ids = []
        found_label_ids = []

        query_vector = [0.0] * self._index.describe_index_stats().dimension
        top_k = min(batch_size, self.config.max_k)

        for batch_ids in fou.iter_batches(sample_ids, batch_size):
            response = self._index.query(
                vector=query_vector,
                filter={"sample_id": {"$in": batch_ids}},
                top_k=top_k,
                include_values=True,
                include_metadata=True,
            )

            for r in response["matches"]:
                found_embeddings.append(r["values"])
                found_sample_ids.append(r["metadata"]["sample_id"])
                found_label_ids.append(r["id"])

        missing_ids = list(set(sample_ids) - set(found_sample_ids))

        return found_embeddings, found_sample_ids, found_label_ids, missing_ids

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

        if reverse is True:
            raise ValueError(
                "Pinecone does not support least similarity queries"
            )

        if k is None or k > self.config.max_k:
            raise ValueError("Pincone requires k<=%s" % self.config.max_k)

        if aggregation not in (None, "mean"):
            raise ValueError("Unsupported aggregation '%s'" % aggregation)

        query = self._parse_neighbors_query(query)
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

        ids = []
        dists = []
        for q in query:
            response = self._index.query(
                vector=q.tolist(), top_k=k, filter=_filter
            )
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

        # Query by ID(s)
        response = self._index.fetch(query_ids)["vectors"]
        query = np.array([response[_id]["values"] for _id in query_ids])

        if single_query:
            query = query[0, :]

        return query

    @classmethod
    def _from_dict(cls, d, samples, config):
        return cls(samples, config)
