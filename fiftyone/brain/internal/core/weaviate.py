"""
Weaviate similarity backend.

| Copyright 2017-2024, Voxel51, Inc.
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

weaviate = fou.lazy_import("weaviate")
wvc = fou.lazy_import("weaviate.classes")


logger = logging.getLogger(__name__)

_SUPPORTED_METRICS = {
    "cosine": wvc.VectorDistance.COSINE,
    "dotproduct": wvc.VectorDistance.DOT,
}

_SUPPORTED_INST_METHODS = [
    "embedded",
    "local",
]


class WeaviateSimilarityConfig(SimilarityConfig):
    """Configuration for the Weaviate similarity backend.

    Args:
        embeddings_field (None): the sample field containing the embeddings,
            if one was provided
        model (None): the :class:`fiftyone.core.models.Model` or name of the
            zoo model that was used to compute embeddings, if known
        patches_field (None): the sample field defining the patches being
            analyzed, if any
        supports_prompts (None): whether this run supports prompt queries
        collection_name (None): the name of a Weaviate collection to use or
            create. If none is provided, a new collection will be created
        metric (None): the embedding distance metric to use when creating a
            new index. Supported values are
            ``("cosine", "dotproduct")``
        replication_factor (None): an optional replication factor to use when
            creating a new index
        shard_number (None): an optional number of shards to use when creating
            a new index
        write_consistency_factor (None): an optional write consistsency factor
            to use when creating a new index
        hnsw_config (None): an optional dict of HNSW config parameters to use
            when creating a new index
        url (None): a Weaviate server URL to use
        api_key (None): a Weaviate API key to use
        grpc_port (None): Port of Weaviate gRPC interface
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
        properties=None,
        inst_method=None,
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
        self.properties = properties
        self.inst_method = inst_method

        # store privately so these aren't serialized
        self._url = url
        self._api_key = api_key
        self._grpc_port = grpc_port
        self._prefer_grpc = prefer_grpc

    @property
    def method(self):
        return "weaviate"

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


class WeaviateSimilarity(Similarity):
    """Weaviate similarity factory.

    Args:
        config: a :class:`WeaviateSimilarityConfig`
    """

    def ensure_requirements(self):
        fou.ensure_package("weaviate-client")

    def ensure_usage_requirements(self):
        fou.ensure_package("weaviate-client")

    def initialize(self, samples, brain_key):
        return WeaviateSimilarityIndex(
            samples, self.config, brain_key, backend=self
        )


class WeaviateSimilarityIndex(SimilarityIndex):
    """Class for interacting with Weaviate similarity indexes.

    Args:
        samples: the :class:`fiftyone.core.collections.SampleCollection` used
        config: the :class:`WeaviateSimilarityConfig` used
        brain_key: the brain key
        backend (None): a :class:`WeaviateSimilarity` instance
    """

    def __init__(self, samples, config, brain_key, backend=None):
        super().__init__(samples, config, brain_key, backend=backend)
        self._client = None
        self._collection = None
        self._initialize()

    def _initialize(self):
        # WeaviateClient does not appear to like passing None as defaults
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

        if (
            self.config.inst_method is None
            or self.config.inst_method == "embedded"
        ):
            self._client = weaviate.connect_to_embedded(port=8787)
            self._client.is_ready()

        try:
            collection_names = self._get_collection_names()
        except Exception as e:
            raise ValueError(
                "Failed to connect to Weaviate backend at URL '%s'. Refer to "
                "https://docs.voxel51.com/integrations/weaviate.html for more "
                "information" % self.config.url
            ) from e

        if self.config.collection_name is None:
            root = "fiftyone_" + fou.to_slug(self.samples._root_dataset.name)
            collection_name = fbu.get_unique_name(
                root, collection_names
            ).replace("-", "_")

            self.config.collection_name = collection_name
            self.save_config()

    def _get_collection_names(self):
        return list(self._client.collections.list_all().keys())

    def _create_collection(self, dimension):
        if self.config.metric:
            metric = self.config.metric
        else:
            metric = "cosine"

        vectorizer_config = (
            wvc.Configure.Vectorizer.none(),
        )  # No vectorizer needed - as the pre-vectorized em

        if self.config.hnsw_config:
            hnsw_config = wvc.Configure.VectorIndex.hnsw(
                **self.config.hnsw_config
            )
        else:
            hnsw_config = wvc.Configure.VectorIndex.hnsw(
                distance_metric=_SUPPORTED_METRICS[metric],
            )

        if self.config.properties:
            properties = self.config.properties
        else:
            properties = (
                properties
            ) = [  # defining properties (data schema) is optional
                wvc.Property(name="sample_id", data_type=wvc.DataType.TEXT),
            ]
        if self._client.collections.exists(self.config.collection_name):  # FIX
            self._collection = self._client.collections.delete(
                self.config.collection_name
            )
            self._client.collections.create(
                name=self.config.collection_name,
                vectorizer_config=wvc.Configure.Vectorizer.none(),  # No vectorizer needed - as the pre-vectorized embeddings will be passed in
                vector_index_config=hnsw_config,
                properties=properties,
            )
            self._collection = self._client.collections.get(
                self.config.collection_name
            )
        else:

            self._client.collections.create(
                name=self.config.collection_name,
                vectorizer_config=wvc.Configure.Vectorizer.none(),  # No vectorizer needed - as the pre-vectorized embeddings will be passed in
                vector_index_config=hnsw_config,
                properties=properties,
            )
            self._collection = self._client.collections.get(
                self.config.collection_name
            )

    # FIX
    def _get_index_ids(self, batch_size=1000):
        ids = []

        for item in self._collection.iterator():
            ids.append(self._to_fiftyone_id(item.uuid))

        return ids

    @property
    def total_index_size(self):
        try:
            return self._collection.aggregate.over_all().total_count
        except:
            return 0

    @property
    def client(self):
        """The ``weaviate.WeaviateClient`` instance for this index."""
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
            weaviate_add_list = []
            for embedding, _id, _sample_id in zip(
                _embeddings, _ids, _sample_ids
            ):
                if _id in existing_ids:
                    self._collection.data.update(
                        properties={
                            "sample_id": _sample_id,
                        },
                        uuid=self._to_weaviate_id(_id),
                        vector=embedding,
                    )
                else:
                    weaviate_obj = wvc.DataObject(
                        properties={
                            "sample_id": _sample_id,
                        },
                        uuid=self._to_weaviate_id(_id),
                        vector=embedding,  # Vector embedding of the image
                    )
                    weaviate_add_list.append(weaviate_obj)
            self._collection.data.insert_many(weaviate_add_list)

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

        if warn_missing or not allow_missing:
            index_ids = self._get_index_ids()

            existing_ids = set(ids) & set(index_ids)

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

        self._collection.data.delete_many(
            where=wvc.Filter("id").contains_any(self._to_weaviate_ids(ids))
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
        self._client.collections.delete(self.config.collection_name)

    def _retrieve_points(self, ids, with_vectors=True):
        # @todo add batching?
        points = []
        for id in ids:
            point = self._collection.query.fetch_object_by_id(
                id, include_vector=with_vectors
            )
            points.append(point)
        return points

    def _get_sample_embeddings(self, sample_ids):
        if sample_ids is None:
            sample_ids = self._get_index_ids()

        response = self._retrieve_points(
            self._to_weaviate_ids(sample_ids),
            with_vectors=True,
        )

        found_embeddings = [r.vector for r in response]
        found_sample_ids = self._to_fiftyone_ids(
            [str(r.uuid) for r in response]
        )
        missing_ids = list(set(sample_ids) - set(found_sample_ids))

        return found_embeddings, found_sample_ids, None, missing_ids

    def _get_patch_embeddings_from_label_ids(self, label_ids):
        if label_ids is None:
            label_ids = self._get_index_ids()

        response = self._retrieve_points(
            self._to_weaviate_ids(label_ids),
            with_vectors=True,
        )

        found_embeddings = [r.vector for r in response]
        found_sample_ids = [r.properties["sample_id"] for r in response]
        found_label_ids = self._to_fiftyone_ids(
            [str(r.uuid) for r in response]
        )
        missing_ids = list(set(label_ids) - set(found_label_ids))

        return found_embeddings, found_sample_ids, found_label_ids, missing_ids

    def _get_patch_embeddings_from_sample_ids(self, sample_ids):

        response = self._collection.query.fetch_objects(
            filters=wvc.Filter("sample_id").contains_any(sample_ids),
            include_vector=True,
            return_properties=["sample_id"],
        ).objects

        found_embeddings = [r.vector for r in response]
        found_sample_ids = [r.properties["sample_id"] for r in response]
        found_label_ids = self._to_fiftyone_ids(
            [str(r.uuid) for r in response]
        )
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
            raise ValueError("Weaviate does not support full index neighbors")

        if reverse is True:
            raise ValueError(
                "Weaviate does not support least similarity queries"
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

            _filter = wvc.Filter("sample_id").contains_any(
                self._to_weaviate_ids(index_ids)
            )
        else:
            _filter = None

        ids = []
        dists = []
        for q in query:
            results = self._collection.query.near_vector(  # ADD A FILTER
                near_vector=q,
                return_properties=["sample_id"],
                return_metadata=wvc.MetadataQuery(
                    distance=True,
                ),
                limit=k,
            ).objects

            ids.append(self._to_fiftyone_ids([str(r.uuid) for r in results]))
            print(ids)
            if return_dists:
                dists.append([r.metadata.distance for r in results])

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
        qids = self._to_weaviate_ids(query_ids)
        response = self._retrieve_points(qids, with_vectors=True)
        query = np.array([r.vector for r in response])

        if single_query:
            query = query[0, :]

        return query

    def _to_weaviate_id(self, _id):
        # Remove single quotes and add 8 zeros
        formatted_str = "00000000" + _id

        # Format the string as XXXXXXXX-XXXX-XXXX-XXXX-XXXXXXXXXXXX
        wid = f"{formatted_str[0:8]}-{formatted_str[8:12]}-{formatted_str[12:16]}-{formatted_str[16:20]}-{formatted_str[20:]}"

        return wid

    def _to_weaviate_ids(self, ids):
        return [self._to_weaviate_id(_id) for _id in ids]

    def _to_fiftyone_id(self, wid):
        # Remove hyphens from the formatted UUID
        stripped_uuid = wid.replace("-", "")

        # Remove the added zeros
        _id = stripped_uuid[8:]

        return _id

    def _to_fiftyone_ids(self, qids):
        return [self._to_fiftyone_id(qid) for qid in qids]

    @classmethod
    def _from_dict(cls, d, samples, config, brain_key):
        return cls(samples, config, brain_key)
