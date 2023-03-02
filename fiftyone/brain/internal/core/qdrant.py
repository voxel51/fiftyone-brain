"""
Qdrant similarity backend.

| Copyright 2017-2023, Voxel51, Inc.
| `voxel51.com <https://voxel51.com/>`_
|
"""
import logging

from bson import ObjectId
import numpy as np

import eta.core.utils as etau

import fiftyone.core.utils as fou
from fiftyone.brain.similarity import (
    SimilarityConfig,
    Similarity,
    SimilarityIndex,
)

fbu = fou.lazy_import("fiftyone.brain.internal.core.utils")

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
        collection_name ("fiftyone-collection"): the name of the Qdrant Index to
            use
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
        write_consistency_factor=None,
        hnsw_config=None,
        optimizers_config=None,
        wal_config=None,
        host="localhost",
        port=6333,
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
        self.write_consistency_factor = write_consistency_factor
        self.hnsw_config = hnsw_config
        self.optimizers_config = optimizers_config
        self.wal_config = wal_config
        self.host = host
        self.port = port

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
        self._collection_name = config.collection_name
        self._replication_factor = config.replication_factor
        self._shard_number = config.shard_number
        self._write_consistency_factor = config.write_consistency_factor
        self._host = config.host
        self._port = config.port

        self._initialize_index(config)

    def _construct_hnsw_config(self, config):
        chc = config.hnsw_config
        self._hnsw_config = qmodels.HnswConfig(
            m=chc["m"],
            ef_construct=chc["ef_construct"],
            full_scan_threshold=chc["full_scan_threshold"],
            max_indexing_threads=chc["max_indexing_threads"],
            on_disk=chc["on_disk"],
            payload_m=chc["payload_m"],
        )

    def _construct_optimizers_config(self, config):
        coc = config.optimizers_config
        self._optimizers_config = qmodels.OptimizersConfig(
            deleted_threshold=coc["deleted_threshold"],
            vacuum_min_vector_number=coc["vacuum_min_vector_number"],
            default_segment_number=coc["default_segment_number"],
            max_segment_size=coc["max_segment_size"],
            memmap_threshold=coc["memmap_threshold"],
            indexing_threshold=coc["indexing_threshold"],
            flush_interval_sec=coc["flush_interval_sec"],
            max_optimization_threads=coc["max_optimization_threads"],
        )

    def _construct_wal_config(self, config):
        cwc = config.wal_config
        self._wal_config = qmodels.WalConfig(
            wal_capacity_mb=cwc["wal_capacity_mb"],
            wal_segments_ahead=cwc["wal_segments_ahead"],
        )

    def _create_collection(self, size):
        self._client.recreate_collection(
            collection_name=self._collection_name,
            vectors_config=qmodels.VectorParams(
                size=size,
                distance=self._metric,
            ),
            shard_number=self._shard_number,
            replication_factor=self._replication_factor,
            hnsw_config=self._hnsw_config,
            optimizers_config=self._optimizers_config,
            wal_config=self._wal_config,
        )

    def _initialize_index(self, config):
        self._client = qdrant.QdrantClient(host=self._host)
        self._construct_hnsw_config(config)
        self._construct_optimizers_config(config)
        self._construct_wal_config(config)

        if config.dimension is not None:
            self._create_collection(config.dimension)

    def _reload_index(self, scroll_pagination=100):

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
                    self._convert_qdrant_id_to_fiftyone_id(doc.id)
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
        return fo_id + "0" * 8

    def _convert_fiftyone_ids_to_qdrant_ids(self, fo_ids):
        return [
            self._convert_fiftyone_id_to_qdrant_id(fo_id) for fo_id in fo_ids
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
        collections = self._client.get_collections().collections
        collection_names = [c.name for c in collections]
        if self._collection_name not in collection_names:
            size = embeddings.shape[1]
            self._create_collection(size)

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
            curr_sample_ids = sample_ids[min_ind:max_ind]

            if not overwrite:
                curr_fo_ids, curr_sample_ids, curr_embeddings = list(
                    zip(
                        *[
                            (fo_id, curr_sample_ids, embedding)
                            for fo_id, sid, embedding in zip(
                                curr_fo_ids, curr_sample_ids, curr_embeddings
                            )
                            if fo_id not in intersection_ids
                        ]
                    )
                )
                curr_fo_ids = list(curr_fo_ids)
                curr_sample_ids = list(curr_sample_ids)
                curr_embeddings = list(curr_embeddings)

            qids = self._convert_fiftyone_ids_to_qdrant_ids(curr_fo_ids)
            payloads = [{"sample_id": sid} for sid in curr_sample_ids]

            self._client.upsert(
                collection_name=self._collection_name,
                points=qmodels.Batch(
                    ids=qids,
                    payloads=payloads,
                    vectors=curr_embeddings,
                ),
            )

    def _retrieve_points(
        self,
        qids,
        with_vectors=True,
        with_payload=True,
    ):
        response = self._client.retrieve(
            collection_name=self._collection_name,
            ids=qids,
            with_vectors=with_vectors,
            with_payload=with_payload,
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
            ),
        )

    def _get_patch_embeddings_from_sample_ids(
        self, sample_ids, allow_missing=True, warn_missing=False
    ):
        _filter = qmodels.Filter(
            should=[
                qmodels.FieldCondition(
                    key="sample_id", match=qmodels.MatchValue(value=sid)
                )
                for sid in sample_ids
            ]
        )

        response = self._client.scroll(
            collection_name=self._collection_name,
            scroll_filter=_filter,
            with_vectors=True,
            with_payload=True,
        )[0]
        found_sample_ids = [record.payload["sample_id"] for record in response]
        found_label_ids = [record.id for record in response]
        found_embeddings = [record.vector for record in response]

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

    def _get_patch_embeddings_from_label_ids(
        self, label_ids, allow_missing=True, warn_missing=False
    ):
        qdrant_query_ids = self._convert_fiftyone_ids_to_qdrant_ids(label_ids)

        response = self._retrieve_points(
            qdrant_query_ids, with_vectors=True, with_payload=True
        )

        found_qids = [record.id for record in response]
        found_label_ids = self._convert_qdrant_ids_to_fiftyone_ids(found_qids)
        found_sample_ids = [record.payload["sample_id"] for record in response]
        found_embeddings = [record.vector for record in response]

        missing_ids = list(set(label_ids).difference(set(found_label_ids)))
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

    def _get_sample_embeddings(
        self, sample_ids, allow_missing=True, warn_missing=False
    ):
        found_label_ids = []

        qdrant_query_ids = self._convert_fiftyone_ids_to_qdrant_ids(sample_ids)
        response = self._retrieve_points(
            qdrant_query_ids,
            with_vectors=True,
        )
        found_qids = [record.id for record in response]
        found_sample_ids = self._convert_qdrant_ids_to_fiftyone_ids(found_qids)
        found_embeddings = [record.vector for record in response]

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

    def _parse_query(self, query):
        if query is None:
            raise ValueError("At least one query must be provided")

        if isinstance(query, np.ndarray):
            # Query by vector(s)
            if query.size == 0:
                raise ValueError("At least one query vector must be provided")

            return query

        if etau.is_str(query):
            query = [query]
        else:
            query = list(query)

        if not query:
            raise ValueError("At least one query must be provided")

        if etau.is_numeric(query[0]):
            return np.asarray(query)

        try:
            ObjectId(query[0])
            is_prompts = False
        except:
            is_prompts = True

        if is_prompts:
            if not self.config.supports_prompts:
                raise ValueError(
                    "Invalid query '%s'; this model does not support prompts"
                    % query[0]
                )

            model = self.get_model()
            return model.embed_prompts(query)

        query_ids = self._convert_fiftyone_ids_to_qdrant_ids(query)
        response = self._retrieve_points(query_ids, with_vectors=True)

        query = np.asarray([record.vector for record in response])
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

        query = self._parse_query(query)
        if aggregation == "mean" and query.ndim == 2:
            query = query.mean(axis=0)

        search_results = self._client.search(
            collection_name=self._collection_name,
            query_vector=query,
            query_filter=qmodels.Filter(
                must=[qmodels.HasIdCondition(has_id=qids)]
            ),
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
