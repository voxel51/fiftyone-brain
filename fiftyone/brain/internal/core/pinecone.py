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
    SimilarityResults,
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
            values are "euclidean", "cosine", and "dotproduct".
        index_name (None): the name of the Pinecone Index to use. If None, the
            name "fiftyone-index" will be used.
    """

    def __init__(
        self,
        embeddings_field=None,
        model=None,
        patches_field=None,
        supports_prompts=None,
        metric="euclidean",
        index_name=None,
        dimension=None,
        pod_type="p1",
        pods=1,
        replicas=1,
        api_key=None,
        environment=None,
        namespace=None,
        **kwargs,
    ):
        if metric not in  _METRICS:
            raise ValueError(
                "metric must be one of {}".format(_METRICS)
            )
        
        super().__init__(
            embeddings_field=embeddings_field,
            model=model,
            patches_field=patches_field,
            supports_prompts=supports_prompts,
            **kwargs,
        )

        index_name = "fiftyone-index" if index_name is None else index_name

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
        return 10000


class PineconeSimilarity(Similarity):
    """Pinecone similarity factory.

    Args:
        config: an :class:`PineconeSimilarityConfig`
    """

    def ensure_requirements(self):
        fou.ensure_package("pinecone-client")

    def initialize(self, samples):
        return PineconeSimilarityResults(samples, self.config, backend=self)

    def cleanup(self, samples, brain_key):
        pass


class PineconeSimilarityResults(SimilarityResults):
    """Class for interacting with Pinecone similarity indexes.

    Args:
        samples: the :class:`fiftyone.core.collections.SampleCollection` used
        config: the :class:`SimilarityConfig` used
        embeddings (None): a ``num_embeddings x num_dims`` array of embeddings
        sample_ids (None): a ``num_embeddings`` array of sample IDs
        label_ids (None): a ``num_embeddings`` array of label IDs, if
            applicable
        backend (None): a :class:`PineconeSimilarity` instance
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
        self._index_name = config.index_name
        self._pod_type = config.pod_type
        self._pods = config.pods
        self._replicas = config.replicas
        self._metric = config.metric
        self._api_key = config.api_key
        self._environment = config.environment
        self._namespace = config.namespace
        self._max_k = config.max_k

        print("Initializing pinecone index")
        self._initialize_connection()
        if self._index_name not in pinecone.list_indexes():
            print("Creating pinecone index")
            pinecone.create_index(
                self._index_name,
                dimension=self._dimension,
                metric=self._metric,
                pod_type=self._pod_type,
                pods=self._pods,
                replicas=self._replicas,
                namespace=self._namespace,
            )

        self._neighbors_helper = None

        super().__init__(samples, config, backend=backend)

    @property
    def sample_ids(self):
        """The sample IDs of the full index."""
        return self._sample_ids

    @property
    def label_ids(self):
        """The label IDs of the full index, or ``None`` if not applicable."""
        return self._label_ids
    
    def _initialize_connection(self):
        pinecone.init(self._api_key, self._environment)
    
    def connect_to_api(self):
        """Direct access to Pinecone API.
        """
        self._initialize_connection()
        return pinecone

    @property
    def index_size(self):
        """The number of vectors in the index."""
        return self.describe_index_stats()['total_vector_count']

    def describe_index_stats(self):
        """Direct API access to Pinecone's describe_index_stats method.
        """
        self._initialize_connection()
        index = pinecone.Index(self._index_name)
        index.describe_index_stats()

    def remove_from_index(
        self,
        sample_ids=None,
        label_ids=None,
        allow_missing=True,
        warn_missing=False,
    ):

        self._initialize_connection()
        index = pinecone.Index(self._index_name)

        if self._label_ids is not None:
            self._label_ids = [
                lid for lid in self._label_ids if lid not in label_ids
            ]
            index.delete(ids=label_ids)
        elif self._sample_ids is not None:
            self._sample_ids = [
                sid for sid in self._sample_ids if sid not in sample_ids
            ]
            index.delete(ids=sample_ids)

    def _sort_by_similarity(
        self, query, k, reverse, aggregation, return_dists
    ):
        if reverse == True:
            raise ValueError(
                "Pinecone backend does not support reverse sorting"
            )

        if k is None:
            raise ValueError(
                "k must be provided when using pinecone similarity"
            )

        if k > 10000:
            raise ValueError(
                "k cannot be greater than 10000 when using pinecone similarity"
            )

        if query is None:
            raise ValueError(
                "A query must be provided when using pinecone similarity"
            )

        if aggregation is not None:
            print("Pinecone backend does not support aggregation.")
            print("Falling back to default.")

        sample_ids = self.current_sample_ids
        label_ids = self.current_label_ids

        self._initialize_connection()
        index = pinecone.Index(self._index_name)
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
        if not return_dists:
            return ids
        else:
            dists = [r["score"] for r in response["matches"]]
            return ids, dists

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

        self._initialize_connection()
        index = pinecone.Index(self._index_name)

        num_existing_ids = 0

        for i in range(num_steps):
            min_ind = upsert_pagination * i
            max_ind = min(upsert_pagination * (i + 1), num_vectors)

            ## simplest case
            if overwrite and allow_existing and not warn_existing:
                index.upsert(
                    index_vectors[min_ind:max_ind],
                    namespace=namespace
                    )
            else:
                curr_index_vectors = index_vectors[min_ind:max_ind]
                curr_index_vector_ids = [r[0] for r in curr_index_vectors]
                response = index.fetch(curr_index_vector_ids)
                curr_existing_ids = list(response.vectors.keys())
                num_existing_ids += len(curr_existing_ids)
                if num_existing_ids > 0 and not allow_existing:
                    raise ValueError(
                        "existing ids were found in the index, but allow_existing=False"
                    )
                elif not overwrite: ## pick out non-existing vectors to add
                    curr_index_vectors = [
                        civ for civ in curr_index_vectors if civ[0] not in curr_existing_ids
                        ]
                    
                index.upsert(
                    curr_index_vectors,
                    namespace=namespace,
                )

        if warn_existing and num_existing_ids > 0:
            print(
                f"Warning: {num_existing_ids} vectors already exist in the index."
            )
            


    def attributes(self):
        attrs = super().attributes()

        if self.config.embeddings_field is not None:
            attrs = [
                attr
                for attr in attrs
                if attr not in ("embeddings", "sample_ids", "label_ids")
            ]

        return attrs

    def _parse_dimension(self, embeddings, config):
        if config.dimension is not None:
            return int(config.dimension)
        elif embeddings is not None:
            return int(embeddings.shape[1])
        return 0

    def _radius_neighbors(self, query=None, thresh=None, return_dists=False):
        raise ValueError(
                "Pinecone backend does not support score thresholding."
            )
        pass

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
            "index_name",
            "pod_type",
            "pods",
            "replicas",
            "namespace",
            "metric",
            "api_key",
            "environment",
        ]

        for attr in config_attrs:
            if attr in d:
                value = d.get("index_name", None)
                if value is not None:
                    config[attr] = value

        return cls(
            samples,
            config,
            embeddings=embeddings,
            sample_ids=sample_ids,
            label_ids=label_ids,
        )
