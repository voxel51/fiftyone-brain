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
import fiftyone.brain.internal.core.utils as fbu

qdrant = fou.lazy_import("qdrant_client")
qmodels = fou.lazy_import("qdrant_client.http.models")


logger = logging.getLogger(__name__)

_SUPPORTED_METRICS = {
    "cosine": qmodels.Distance.COSINE,
    "dotproduct": qmodels.Distance.DOT,
    "euclidean": qmodels.Distance.EUCLID,
}


class QdrantSimilarityConfig(SimilarityConfig):
    """Configuration for the Qdrant similarity backend.

    Args:
        embeddings_field (None): the sample field containing the embeddings,
            if one was provided
        model (None): the :class:`fiftyone.core.models.Model` or name of the
            zoo model that was used to compute embeddings, if known
        patches_field (None): the sample field defining the patches being
            analyzed, if any
        supports_prompts (None): whether this run supports prompt queries
        collection_name (None): the name of a Qdrant collection to use or
            create. If none is provided, a new collection will be created
        metric (None): the embedding distance metric to use when creating a
            new index. Supported values are
            ``("cosine", "dotproduct", "euclidean")``
        replication_factor (None): an optional replication factor to use when
            creating a new index
        shard_number (None): an optional number of shards to use when creating
            a new index
        write_consistency_factor (None): an optional write consistsency factor
            to use when creating a new index
        hnsw_config (None): an optional dict of HNSW config parameters to use
            when creating a new index
        optimizers_config (None): an optional dict of optimizer parameters to
            use when creating a new index
        wal_config (None): an optional dict of WAL config parameters to use
            when creating a new index
        url (None): a Qdrant server URL to use
        api_key (None): a Qdrant API key to use
        grpc_port (None): Port of Qdrant gRPC interface
        prefer_grpc (None): If `true`, use gRPC interface when possible
    """

    def __init__(
        self,
        embeddings_field=None,
        model=None,
        patches_field=None,
        supports_prompts=None,
        collection_name=None,
        metric=None,
        replication_factor=None,
        shard_number=None,
        write_consistency_factor=None,
        hnsw_config=None,
        optimizers_config=None,
        wal_config=None,
        url=None,
        api_key=None,
        grpc_port=None,
        prefer_grpc=None,
        **kwargs,
    ):
        if metric is not None and metric not in _SUPPORTED_METRICS:
            raise ValueError(
                "Unsupported metric '%s'. Supported values are %s"
                % (metric, tuple(_SUPPORTED_METRICS.keys()))
            )

        super().__init__(
            embeddings_field=embeddings_field,
            model=model,
            patches_field=patches_field,
            supports_prompts=supports_prompts,
            **kwargs,
        )

        self.collection_name = collection_name
        self.metric = metric
        self.replication_factor = replication_factor
        self.shard_number = shard_number
        self.write_consistency_factor = write_consistency_factor
        self.hnsw_config = hnsw_config
        self.optimizers_config = optimizers_config
        self.wal_config = wal_config

        # store privately so these aren't serialized
        self._url = url
        self._api_key = api_key
        self._grpc_port = grpc_port
        self._prefer_grpc = prefer_grpc

    @property
    def method(self):
        return "qdrant"

    @property
    def url(self):
        return self._url

    @url.setter
    def url(self, value):
        self._url = value

    @property
    def api_key(self):
        return self._api_key

    @api_key.setter
    def api_key(self, value):
        self._api_key = value

    @property
    def grpc_port(self):
        return self._grpc_port

    @grpc_port.setter
    def grpc_port(self, value):
        self._grpc_port = value

    @property
    def prefer_grpc(self):
        return self._prefer_grpc

    @prefer_grpc.setter
    def prefer_grpc(self, value):
        self._prefer_grpc = value

    @property
    def max_k(self):
        return None

    @property
    def supports_least_similarity(self):
        return False

    @property
    def supported_aggregations(self):
        return ("mean",)

    def load_credentials(
        self, url=None, api_key=None, grpc_port=None, prefer_grpc=None
    ):
        self._load_parameters(
            url=url,
            api_key=api_key,
            grpc_port=grpc_port,
            prefer_grpc=prefer_grpc,
        )


class QdrantSimilarity(Similarity):
    """Qdrant similarity factory.

    Args:
        config: a :class:`QdrantSimilarityConfig`
    """

    def ensure_requirements(self):
        fou.ensure_package("qdrant-client")

    def ensure_usage_requirements(self):
        fou.ensure_package("qdrant-client")

    def initialize(self, samples, brain_key):
        return QdrantSimilarityIndex(
            samples, self.config, brain_key, backend=self
        )


class QdrantSimilarityIndex(SimilarityIndex):
    """Class for interacting with Qdrant similarity indexes.

    Args:
        samples: the :class:`fiftyone.core.collections.SampleCollection` used
        config: the :class:`QdrantSimilarityConfig` used
        brain_key: the brain key
        backend (None): a :class:`QdrantSimilarity` instance
    """

    def __init__(self, samples, config, brain_key, backend=None):
        super().__init__(samples, config, brain_key, backend=backend)
        self._client = None
        self._initialize()

    def _initialize(self):
        # QdrantClient does not appear to like passing None as defaults
        grpc_port = (
            self.config.grpc_port
            if self.config.grpc_port is not None
            else 6334
        )
        prefer_grpc = (
            self.config.prefer_grpc
            if self.config.prefer_grpc is not None
            else False
        )

        self._client = qdrant.QdrantClient(
            url=self.config.url,
            api_key=self.config.api_key,
            grpc_port=grpc_port,
            prefer_grpc=prefer_grpc,
        )

        try:
            collection_names = self._get_collection_names()
        except Exception as e:
            raise ValueError(
                "Failed to connect to Qdrant backend at URL '%s'. Refer to "
                "https://docs.voxel51.com/integrations/qdrant.html for more "
                "information" % self.config.url
            ) from e

        if self.config.collection_name is None:
            root = "fiftyone-" + fou.to_slug(self.samples._root_dataset.name)
            collection_name = fbu.get_unique_name(root, collection_names)

            self.config.collection_name = collection_name
            self.save_config()

    def _get_collection_names(self):
        return [c.name for c in self._client.get_collections().collections]

    def _create_collection(self, dimension):
        if self.config.metric:
            metric = self.config.metric
        else:
            metric = "cosine"

        vectors_config = qmodels.VectorParams(
            size=dimension,
            distance=_SUPPORTED_METRICS[metric],
        )

        if self.config.hnsw_config:
            hnsw_config = qmodels.HnswConfig(**self.config.hnsw_config)
        else:
            hnsw_config = None

        if self.config.optimizers_config:
            optimizers_config = qmodels.OptimizersConfig(
                **self.config.optimizers_config
            )
        else:
            optimizers_config = None

        if self.config.wal_config:
            wal_config = qmodels.WalConfig(**self.config.wal_config)
        else:
            wal_config = None

        self._client.recreate_collection(
            collection_name=self.config.collection_name,
            vectors_config=vectors_config,
            shard_number=self.config.shard_number,
            replication_factor=self.config.replication_factor,
            hnsw_config=hnsw_config,
            optimizers_config=optimizers_config,
            wal_config=wal_config,
        )

    def _get_index_ids(self, batch_size=1000):
        ids = []

        offset = 0
        while offset is not None:
            response = self._client.scroll(
                collection_name=self.config.collection_name,
                offset=offset,
                limit=batch_size,
                with_payload=True,
                with_vectors=False,
            )
            ids.extend([self._to_fiftyone_id(r.id) for r in response[0]])
            offset = response[-1]

        return ids

    @property
    def total_index_size(self):
        try:
            return self._client.count(self.config.collection_name).count
        except:
            return 0

    @property
    def client(self):
        """The ``qdrant.QdrantClient`` instance for this index."""
        return self._client

    def add_to_index(
        self,
        embeddings,
        sample_ids,
        label_ids=None,
        overwrite=True,
        allow_existing=True,
        warn_existing=False,
        reload=True,
        batch_size=1000,
    ):
        if self.config.collection_name not in self._get_collection_names():
            self._create_collection(embeddings.shape[1])

        if label_ids is not None:
            ids = label_ids
        else:
            ids = sample_ids

        if warn_existing or not allow_existing or not overwrite:
            index_ids = self._get_index_ids()

            existing_ids = set(ids) & set(index_ids)
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
            self._client.upsert(
                collection_name=self.config.collection_name,
                points=qmodels.Batch(
                    ids=self._to_qdrant_ids(_ids),
                    payloads=[{"sample_id": _id} for _id in _sample_ids],
                    vectors=_embeddings,
                ),
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

        qids = self._to_qdrant_ids(ids)

        if warn_missing or not allow_missing:
            response = self._retrieve_points(qids, with_vectors=False)

            existing_ids = self._to_fiftyone_ids([r.id for r in response])
            missing_ids = list(set(ids) - set(existing_ids))
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
            collection_name=self.config.collection_name,
            points_selector=qmodels.PointIdsList(points=qids),
        )

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
        self._client.delete_collection(self.config.collection_name)

    def _retrieve_points(self, qids, with_vectors=True, with_payload=True):
        # @todo add batching?
        return self._client.retrieve(
            collection_name=self.config.collection_name,
            ids=qids,
            with_vectors=with_vectors,
            with_payload=with_payload,
        )

    def _get_sample_embeddings(self, sample_ids):
        if sample_ids is None:
            sample_ids = self._get_index_ids()

        response = self._retrieve_points(
            self._to_qdrant_ids(sample_ids),
            with_vectors=True,
        )

        found_embeddings = [r.vector for r in response]
        found_sample_ids = self._to_fiftyone_ids([r.id for r in response])
        missing_ids = list(set(sample_ids) - set(found_sample_ids))

        return found_embeddings, found_sample_ids, None, missing_ids

    def _get_patch_embeddings_from_label_ids(self, label_ids):
        if label_ids is None:
            label_ids = self._get_index_ids()

        response = self._retrieve_points(
            self._to_qdrant_ids(label_ids),
            with_vectors=True,
            with_payload=True,
        )

        found_embeddings = [r.vector for r in response]
        found_sample_ids = [r.payload["sample_id"] for r in response]
        found_label_ids = self._to_fiftyone_ids([r.id for r in response])
        missing_ids = list(set(label_ids) - set(found_label_ids))

        return found_embeddings, found_sample_ids, found_label_ids, missing_ids

    def _get_patch_embeddings_from_sample_ids(self, sample_ids):
        _filter = qmodels.Filter(
            should=[
                qmodels.FieldCondition(
                    key="sample_id", match=qmodels.MatchValue(value=sid)
                )
                for sid in sample_ids
            ]
        )

        response = self._client.scroll(
            collection_name=self.config.collection_name,
            scroll_filter=_filter,
            with_vectors=True,
            with_payload=True,
        )[0]

        found_embeddings = [r.vector for r in response]
        found_sample_ids = [r.payload["sample_id"] for r in response]
        found_label_ids = [r.id for r in response]
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
            raise ValueError("Qdrant does not support full index neighbors")

        if reverse is True:
            raise ValueError(
                "Qdrant does not support least similarity queries"
            )

        if aggregation not in (None, "mean"):
            raise ValueError("Unsupported aggregation '%s'" % aggregation)

        if k is None:
            k = self.index_size

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

            _filter = qmodels.Filter(
                must=[
                    qmodels.HasIdCondition(
                        has_id=self._to_qdrant_ids(index_ids)
                    )
                ]
            )
        else:
            _filter = None

        ids = []
        dists = []
        for q in query:
            results = self._client.search(
                collection_name=self.config.collection_name,
                query_vector=q,
                query_filter=_filter,
                with_payload=False,
                limit=k,
            )

            ids.append(self._to_fiftyone_ids([r.id for r in results]))
            if return_dists:
                dists.append([r.score for r in results])

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
        qids = self._to_qdrant_ids(query_ids)
        response = self._retrieve_points(qids, with_vectors=True)
        query = np.array([r.vector for r in response])

        if single_query:
            query = query[0, :]

        return query

    def _to_qdrant_id(self, _id):
        return _id + "00000000"

    def _to_qdrant_ids(self, ids):
        return [self._to_qdrant_id(_id) for _id in ids]

    def _to_fiftyone_id(self, qid):
        return qid.replace("-", "")[:-8]

    def _to_fiftyone_ids(self, qids):
        return [self._to_fiftyone_id(qid) for qid in qids]

    @classmethod
    def _from_dict(cls, d, samples, config, brain_key):
        return cls(samples, config, brain_key)
