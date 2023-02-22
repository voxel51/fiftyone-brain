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
    """Configuration for the Qdrant similarity backend.

    Args:
        embeddings_field (None): the sample field containing the embeddings,
            if one was provided
        model (None): the :class:`fiftyone.core.models.Model` or class name of
            the model that was used to compute embeddings, if one was provided
        patches_field (None): the sample field defining the patches being
            analyzed, if any
        supports_prompts (False): whether this run supports prompt queries
        collection_name ("fiftyone-collection"): the name of the Qdrant Index to use
        metric ("euclidean"): the embedding distance metric to use. Supported
            values are ``("euclidean", "cosine", and "dotproduct")``
    """
    def __init__(
        self,
        embeddings_field=None,
        model=None,
        patches_field=None,
        supports_prompts=None,
        metric="euclidean",
        collection_name="fiftyone-collection",
        dimension=None,
        replication_factor=1,
        shard_number=1,
        host='localhost',
        # port=6333,
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
        self.replication_factor = replication_factor
        self.shard_number = shard_number
        self.host = host
        # self.port = port

    @property
    def method(self):
        return "qdrant"

    @property
    def max_k(self):
        return None

    @property
    def supports_least_similarity(self):
        return False

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
        self._host = config.host
        # self._port = config.port

        self._initialize_index()
    
    def _initialize_index(self):
        self._client = qdrant.QdrantClient(host=self._host)

        self._client.recreate_collection(
            collection_name=self._collection_name,
            vectors_config=qmodels.VectorParams(
                size = self._dimension,
                distance = self._metric,
            )
        )

    def _reload_index(
        self,
        scroll_pagination=100
        ):

        offset = 0
        self._client = qdrant.QdrantClient(host=self._host)
        self._fiftyone_ids = []

        while offset is not None:
            response = self._client.scroll(
                collection_name=self._collection_name,
                offset=offset,
                limit=scroll_pagination,
                with_payload=True,
                with_vectors=False,
            )

            for doc in response[0]:
                self._fiftyone_ids.append(
                    self._convert_qdrant_id_to_fiftyone_id(
                        doc.id
                        )
                    )

            offset = response[-1]

    @property
    def total_index_size(self):
        return self._client.count(self._collection_name).count

    def connect_to_api(self):
        return self._client
    
    def _get_collection(self):
        return self._client.get_collection(
            collection_name=self._collection_name
            )

    def _convert_fiftyone_id_to_qdrant_id(self, fo_id):
        ### generate UUID
        return fo_id + '0'*8

    def _convert_fiftyone_ids_to_qdrant_ids(self, fo_ids):
        return [
            self._convert_fiftyone_id_to_qdrant_id(fo_id)
            for fo_id in fo_ids
        ]
    
    def _convert_qdrant_id_to_fiftyone_id(self, qdrant_id):
        return qdrant_id.replace("-", "")[:-8]
    
    def _convert_qdrant_ids_to_fiftyone_ids(self, qdrant_ids):
        return [
            self._convert_qdrant_id_to_fiftyone_id(qdrant_id)
            for qdrant_id in qdrant_ids
        ]

    def add_to_index(
        self,
        embeddings,
        sample_ids,
        label_ids=None,
        overwrite=True,
        allow_existing=True,
        warn_existing=False,
        upsert_pagination=None,
    ):
        embeddings_list = [arr.tolist() for arr in embeddings]
        fo_ids = label_ids if label_ids is not None else sample_ids

        num_vectors = embeddings.shape[0]
        num_intersection_ids = 0

        if upsert_pagination is not None:
            num_steps = int(np.ceil(num_vectors / upsert_pagination))
        else:
            num_steps = 1
            upsert_pagination = num_vectors

        ## only scroll through and reload if necessary
        if warn_existing or not allow_existing or not overwrite:
            self._reload_index()
            existing_ids = set(self._fiftyone_ids)
            new_ids = set(fo_ids)
            intersection_ids = existing_ids.intersection(new_ids)
            num_intersection_ids = len(intersection_ids)

        if num_intersection_ids > 0:
            if not allow_existing:
                raise ValueError(
                    "Found %d IDs (eg %s) that already exist in the index"
                    % (num_intersection_ids, intersection_ids[0])
                )
            if warn_existing and overwrite:
                logger.warning(
                    "Overwriting %d IDs that already exist in the index",
                    num_intersection_ids,
                )
            if warn_existing and not overwrite:
                logger.warning(
                    "Skipping %d IDs that already exist in the index",
                    num_intersection_ids,
                )

        for i in range(num_steps):
            min_ind = upsert_pagination * i
            max_ind = min(upsert_pagination * (i + 1), num_vectors)

            curr_fo_ids = fo_ids[min_ind:max_ind]
            curr_embeddings = embeddings_list[min_ind:max_ind]

            if not overwrite:
                curr_fo_ids, curr_embeddings = list(zip(*[
                    (fo_id, embedding) 
                    for fo_id, embedding in zip(curr_fo_ids, curr_embeddings)
                    if fo_id not in intersection_ids
                ]))
                curr_fo_ids = list(curr_fo_ids)
                curr_embeddings = list(curr_embeddings)
            
            qids = self._convert_fiftyone_ids_to_qdrant_ids(curr_fo_ids)
            self._client.upsert(
                collection_name=self._collection_name,
                points=qmodels.Batch(
                    ids=qids,
                    vectors=curr_embeddings,
                )
            )

    def _retrieve_points(
        self,
        qids,
        with_vectors=True,
    ):
        response = self._client.retrieve(
            collection_name=self._collection_name,
            ids=qids,
            with_vectors=with_vectors,
        )
        return response

    def remove_from_index(
        self,
        sample_ids=None,
        label_ids=None,
        allow_missing=True,
        warn_missing=False,
    ):
        fo_ids = label_ids if label_ids is not None else sample_ids
        qids = self._convert_fiftyone_ids_to_qdrant_ids(fo_ids)

        if warn_missing or not allow_missing:
            response = self._retrieve_points(
                qids,
                with_vectors=False,
            )

            existing_qids = [record.id for record in response]
            existing_fo_ids = self._convert_qdrant_ids_to_fiftyone_ids(
                existing_qids
            )

            missing_ids = list(set(fo_ids).difference(set(existing_fo_ids)))
            num_missing_ids = len(missing_ids)
            if num_missing_ids > 0:
                if not allow_missing:
                    raise ValueError(
                        "Found %d IDs (eg %s) that do not exist in the index"
                        % (num_missing_ids, missing_ids[0])
                    )
                if warn_missing and not allow_missing:
                    logger.warning(
                        "Skipping %d IDs that do not exist in the index",
                        num_missing_ids,
                    )
        self._client.delete(
            collection_name=self._collection_name,
            points_selector=qmodels.PointIdsList(
                points=qids,
            )
        )

    def _parse_query(self, query):
        fo_query_ids = None

        if etau.is_str(query):
            fo_query_ids = [query]
        else:
            if etau.is_container(query) and type(query[0]) in (str, np.str_):
                fo_query_ids = query
        
        if fo_query_ids is not None:
            query_ids = self._convert_fiftyone_ids_to_qdrant_ids(fo_query_ids)
            response = self._retrieve_points(
                query_ids,
                with_vectors=True
            )
            query = [record.vector for record in response]

        query = np.asarray(query)
        return query
        
    def _kneighbors(
        self,
        query=None,
        k=None,
        reverse=False,
        aggregation=None,
        return_dists=False,
    ):
        if reverse == True:
            raise ValueError(
                "Qdrant does not support least similarity queries"
            )
        
        if k is None:
            raise ValueError("k required for querying with Qdrant backend")

        if aggregation not in (None, "mean"):
            raise ValueError("Unsupported aggregation '%s'" % aggregation)
        
        if self.config.patches_field is not None:
            fo_ids = self.current_label_ids
        else:
            fo_ids = self.current_sample_ids

        qids = self._convert_fiftyone_ids_to_qdrant_ids(fo_ids)
        _filter=qmodels.Filter(must=[qmodels.HasIdCondition(has_id=qids)]),

        query = self._parse_query(query)
        if aggregation == "mean" and query.ndim == 2:
            query = query.mean(axis=0)
        
        # search_results = self._client.search(
        #     collection_name=self._collection_name,
        #     query_vector=query,
        #     with_payload=False,
        #     limit=k,
        #     query_filter=_filter
        # )
        search_results = self._client.search(
            collection_name=self._collection_name,
            query_vector=query,
            with_payload=False,
            limit=k,
        )

        ids = self._convert_qdrant_ids_to_fiftyone_ids(
            [res.id for res in search_results]
        )
        if return_dists:
            dists = [res.score for res in search_results]
            return ids, dists
        else:
            return ids
        
    @classmethod
    def _from_dict(cls, d, samples, config):
        return cls(samples, config)
