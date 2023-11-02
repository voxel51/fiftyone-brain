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

_SUPPORTED_METRICS = ("cosine", "dotproduct", "euclidean")


class PineconeSimilarityConfig(SimilarityConfig):
    """Configuration for the Pinecone similarity backend.

    Args:
        embeddings_field (None): the sample field containing the embeddings,
            if one was provided
        model (None): the :class:`fiftyone.core.models.Model` or name of the
            zoo model that was used to compute embeddings, if known
        patches_field (None): the sample field defining the patches being
            analyzed, if any
        supports_prompts (None): whether this run supports prompt queries
        index_name (None): the name of a Pinecone index to use or create. If
            none is provided, a new index will be created
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
        api_key (None): a Pinecone API key to use
        environment (None): a Pinecone environment to use
        project_name (None): a Pinecone project to use
    """

    def __init__(
        self,
        embeddings_field=None,
        model=None,
        patches_field=None,
        supports_prompts=None,
        index_name=None,
        index_type=None,
        namespace=None,
        metric=None,
        replicas=None,
        shards=None,
        pods=None,
        pod_type=None,
        api_key=None,
        environment=None,
        project_name=None,
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
        self._api_key = api_key
        self._environment = environment
        self._project_name = project_name

    @property
    def method(self):
        return "pinecone"

    @property
    def api_key(self):
        return self._api_key

    @api_key.setter
    def api_key(self, value):
        self._api_key = value

    @property
    def environment(self):
        return self._environment

    @environment.setter
    def environment(self, value):
        self._environment = value

    @property
    def project_name(self):
        return self._project_name

    @project_name.setter
    def project_name(self, value):
        self._project_name = value

    @property
    def max_k(self):
        return 10000  # Pinecone limit

    @property
    def supports_least_similarity(self):
        return False

    @property
    def supported_aggregations(self):
        return ("mean",)

    def load_credentials(
        self, api_key=None, environment=None, project_name=None
    ):
        self._load_parameters(
            api_key=api_key, environment=environment, project_name=project_name
        )


class PineconeSimilarity(Similarity):
    """Pinecone similarity factory.

    Args:
        config: a :class:`PineconeSimilarityConfig`
    """

    def ensure_requirements(self):
        fou.ensure_package("pinecone-client")

    def ensure_usage_requirements(self):
        fou.ensure_package("pinecone-client")

    def initialize(self, samples, brain_key):
        return PineconeSimilarityIndex(
            samples, self.config, brain_key, backend=self
        )


class PineconeSimilarityIndex(SimilarityIndex):
    """Class for interacting with Pinecone similarity indexes.

    Args:
        samples: the :class:`fiftyone.core.collections.SampleCollection` used
        config: the :class:`PineconeSimilarityConfig` used
        brain_key: the brain key
        backend (None): a :class:`PineconeSimilarity` instance
    """

    def __init__(self, samples, config, brain_key, backend=None):
        super().__init__(samples, config, brain_key, backend=backend)
        self._index = None
        self._initialize()

    def _initialize(self):
        pinecone.init(
            api_key=self.config.api_key,
            environment=self.config.environment,
            project_name=self.config.project_name,
        )

        try:
            index_names = pinecone.list_indexes()
        except Exception as e:
            raise ValueError(
                "Failed to connect to Pinecone backend at environment '%s'. "
                "Refer to https://docs.voxel51.com/integrations/pinecone.html "
                "for more information" % self.config.environment
            ) from e

        if self.config.index_name is None:
            root = "fiftyone-" + fou.to_slug(self.samples._root_dataset.name)
            index_name = fbu.get_unique_name(root, index_names)

            self.config.index_name = index_name
            self.save_config()

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
            return 0

        return self._index.describe_index_stats()["total_vector_count"]

    def add_to_index(
        self,
        embeddings,
        sample_ids,
        label_ids=None,
        overwrite=True,
        allow_existing=True,
        warn_existing=False,
        reload=True,
        batch_size=100,
        namespace=None,
    ):
        if namespace is None:
            namespace = self.config.namespace

        if self._index is None:
            self._create_index(embeddings.shape[1])

        if label_ids is not None:
            ids = label_ids
        else:
            ids = sample_ids

        if warn_existing or not allow_existing or not overwrite:
            existing_ids = self._get_existing_ids(ids)
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
            del_inds = [i for i, _id in enumerate(ids) if _id in existing_ids]

            embeddings = np.delete(embeddings, del_inds)
            sample_ids = np.delete(sample_ids, del_inds)
            if label_ids is not None:
                label_ids = np.delete(label_ids, del_inds)

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
            _id_dicts = [
                {"id": _id, "sample_id": _sid}
                for _id, _sid in zip(_ids, _sample_ids)
            ]
            self._index.upsert(
                list(zip(_ids, _embeddings, _id_dicts)),
                namespace=namespace,
            )

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

        if reload:
            self.reload()

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

    def cleanup(self):
        pinecone.delete_index(self.config.index_name)
        self._index = None

    def _get_existing_ids(self, ids, batch_size=1000):
        existing_ids = set()
        for batch_ids in fou.iter_batches(ids, batch_size):
            response = self._index.fetch(ids=list(batch_ids))["vectors"]
            existing_ids.update(response.keys())

        return existing_ids

    def _get_sample_embeddings(self, sample_ids, batch_size=1000):
        found_embeddings = []
        found_sample_ids = []

        if sample_ids is None:
            raise ValueError(
                "Pinecone does not support retrieving all vectors in an index"
            )

        for batch_ids in fou.iter_batches(sample_ids, batch_size):
            response = self._index.fetch(ids=list(batch_ids))["vectors"]

            for r in response.values():
                found_embeddings.append(r["values"])
                found_sample_ids.append(r["id"])

        missing_ids = list(set(sample_ids) - set(found_sample_ids))

        return found_embeddings, found_sample_ids, None, missing_ids

    def _get_patch_embeddings_from_label_ids(self, label_ids, batch_size=1000):
        found_embeddings = []
        found_sample_ids = []
        found_label_ids = []

        if label_ids is None:
            raise ValueError(
                "Pinecone does not support retrieving all vectors in an index"
            )

        for batch_ids in fou.iter_batches(label_ids, batch_size):
            response = self._index.fetch(ids=list(batch_ids))["vectors"]

            for r in response.values():
                found_embeddings.append(r["values"])
                found_sample_ids.append(r["metadata"]["sample_id"])
                found_label_ids.append(r["id"])

        missing_ids = list(set(label_ids) - set(found_label_ids))

        return found_embeddings, found_sample_ids, found_label_ids, missing_ids

    def _get_patch_embeddings_from_sample_ids(
        self, sample_ids, batch_size=100
    ):
        found_embeddings = []
        found_sample_ids = []
        found_label_ids = []

        query_vector = [0.0] * self._get_dimension()
        top_k = min(batch_size, self.config.max_k)

        for batch_ids in fou.iter_batches(sample_ids, batch_size):
            response = self._index.query(
                vector=query_vector,
                filter={"sample_id": {"$in": list(batch_ids)}},
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

        if self.has_view:
            if self.config.patches_field is not None:
                index_ids = self.current_label_ids
            else:
                index_ids = self.current_sample_ids

            _filter = {"id": {"$in": list(index_ids)}}
        else:
            _filter = None

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

    def _get_dimension(self):
        if self._index is None:
            return None

        return self._index.describe_index_stats().dimension

    @classmethod
    def _from_dict(cls, d, samples, config, brain_key):
        return cls(samples, config, brain_key)
