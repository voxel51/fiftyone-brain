"""
Milvus similarity backend.

| Copyright 2017-2023, Voxel51, Inc.
| `voxel51.com <https://voxel51.com/>`_
|
"""
import logging

import numpy as np
from uuid import uuid4

import eta.core.utils as etau

import fiftyone.core.utils as fou
from fiftyone.brain.similarity import (
    SimilarityConfig,
    Similarity,
    SimilarityIndex,
)
import fiftyone.brain.internal.core.utils as fbu

pymilvus = fou.lazy_import("pymilvus")


logger = logging.getLogger(__name__)

_SUPPORTED_METRICS = {
    "dotproduct": "IP",
    "euclidean": "L2",
}


class MilvusSimilarityConfig(SimilarityConfig):
    """Configuration for the Milvus similarity backend.

    Args:
        embeddings_field (None): the sample field containing the embeddings,
            if one was provided
        model (None): the :class:`fiftyone.core.models.Model` or name of the
            zoo model that was used to compute embeddings, if known
        patches_field (None): the sample field defining the patches being
            analyzed, if any
        supports_prompts (None): whether this run supports prompt queries
        collection_name (None): the name of a Milvus collection to use or
            create. If none is provided, a new collection will be created
        metric ("dotproduct"): the embedding distance metric to use when
            creating a new index. Supported values are
            ``("dotproduct", "euclidean")``
        consistency_level ("Session"): the consistency level to use. Supported
            values are ``("Session", "Strong", "Bounded", "Eventually")``
        uri (None): a full Milvus server address to use
        user (None): a username to use
        password (None): a password to use
    """

    def __init__(
        self,
        embeddings_field=None,
        model=None,
        patches_field=None,
        supports_prompts=None,
        collection_name=None,
        metric="dotproduct",
        consistency_level="Session",
        uri=None,
        user=None,
        password=None,
        **kwargs,
    ):
        if metric not in _SUPPORTED_METRICS:
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
        self.consistency_level = consistency_level

        # store privately so these aren't serialized
        self._uri = uri
        self._user = user
        self._password = password

    @property
    def method(self):
        return "milvus"

    @property
    def uri(self):
        return self._uri

    @uri.setter
    def uri(self, value):
        self._uri = value

    @property
    def user(self):
        return self._user

    @user.setter
    def user(self, value):
        self._user = value

    @property
    def password(self):
        return self._password

    @password.setter
    def password(self, value):
        self._password = value

    @property
    def max_k(self):
        return 16384

    @property
    def supports_least_similarity(self):
        return False

    @property
    def supported_aggregations(self):
        return ("mean",)

    @property
    def index_params(self):
        return {
            "metric_type": _SUPPORTED_METRICS[self.metric],
            "index_type": "HNSW",
            "params": {"M": 8, "efConstruction": 64},
        }

    @property
    def search_params(self):
        return {
            "HNSW": {
                "metric_type": _SUPPORTED_METRICS[self.metric],
                "params": {"ef": 10},
            },
        }

    def load_credentials(self, uri=None, user=None, password=None):
        self._load_parameters(uri=uri, user=user, password=password)


class MilvusSimilarity(Similarity):
    """Milvus similarity factory.

    Args:
        config: a :class:`MilvusSimilarityConfig`
    """

    def ensure_requirements(self):
        fou.ensure_package("pymilvus")

    def ensure_usage_requirements(self):
        fou.ensure_package("pymilvus")

    def initialize(self, samples, brain_key):
        return MilvusSimilarityIndex(
            samples, self.config, brain_key, backend=self
        )


class MilvusSimilarityIndex(SimilarityIndex):
    """Class for interacting with Milvus similarity indexes.

    Args:
        samples: the :class:`fiftyone.core.collections.SampleCollection` used
        config: the :class:`MilvusSimilarityConfig` used
        brain_key: the brain key
        backend (None): a :class:`MilvusSimilarity` instance
    """

    def __init__(self, samples, config, brain_key, backend=None):
        super().__init__(samples, config, brain_key, backend=backend)
        self._alias = None
        self._collection = None
        self._initialize()

    def _initialize(self):
        kwargs = {}
        if self.config.uri:
            kwargs["uri"] = self.config.uri

        if self.config.user:
            kwargs["user"] = self.config.user

        if self.config.password:
            kwargs["password"] = self.config.password

        alias = uuid4().hex if kwargs else "default"

        try:
            pymilvus.connections.connect(alias=alias, **kwargs)
        except pymilvus.MilvusException as e:
            raise ValueError(
                "Failed to connect to Milvus backend at URI '%s'. Refer to "
                "https://docs.voxel51.com/integrations/milvus.html for more "
                "information" % self.config.uri
            ) from e

        collection_names = pymilvus.utility.list_collections(using=alias)

        if self.config.collection_name is None:
            # Milvus only supports numbers, letters and underscores
            root = "fiftyone-" + fou.to_slug(self.samples._root_dataset.name)
            root = root.replace("-", "_")
            collection_name = fbu.get_unique_name(root, collection_names)
            collection_name = collection_name.replace("-", "_")

            self.config.collection_name = collection_name
            self.save_config()

        if self.config.collection_name in collection_names:
            collection = pymilvus.Collection(
                self.config.collection_name, using=alias
            )
            collection.load()
        else:
            collection = None

        self._alias = alias
        self._collection = collection

    def _create_collection(self, dimension):
        schema = pymilvus.CollectionSchema(
            [
                pymilvus.FieldSchema(
                    "pk",
                    pymilvus.DataType.VARCHAR,
                    is_primary=True,
                    auto_id=False,
                    max_length=64000,
                ),
                pymilvus.FieldSchema(
                    "vector", pymilvus.DataType.FLOAT_VECTOR, dim=dimension
                ),
                pymilvus.FieldSchema(
                    "sample_id", pymilvus.DataType.VARCHAR, max_length=64000
                ),
            ]
        )

        collection = pymilvus.Collection(
            self.config.collection_name,
            schema,
            consistency_level=self.config.consistency_level,
            using=self._alias,
        )
        collection.create_index(
            "vector", index_params=self.config.index_params
        )
        collection.load()

        self._collection = collection

    @property
    def collection(self):
        """The ``pymilvus.Collection`` instance for this index."""
        return self._collection

    @property
    def total_index_size(self):
        if self._collection is None:
            return 0

        return self._collection.num_entities

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
    ):
        if self._collection is None:
            self._create_collection(embeddings.shape[1])

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

        elif existing_ids and overwrite:
            self._delete_ids(existing_ids)

        embeddings = [e.tolist() for e in embeddings]
        sample_ids = list(sample_ids)
        ids = list(ids)

        for _embeddings, _ids, _sample_ids in zip(
            fou.iter_batches(embeddings, batch_size),
            fou.iter_batches(ids, batch_size),
            fou.iter_batches(sample_ids, batch_size),
        ):
            insert_data = [
                list(_ids),
                list(_embeddings),
                list(_sample_ids),
            ]
            self._collection.insert(insert_data)

        self._collection.flush()

        if reload:
            self.reload()

    def _get_existing_ids(self, ids):
        ids = ['"' + str(entry) + '"' for entry in ids]
        expr = f"""pk in [{','.join(ids)}]"""
        return self._collection.query(expr)

    def _delete_ids(self, ids):
        ids = ['"' + str(entry) + '"' for entry in ids]
        expr = f"""pk in [{','.join(ids)}]"""
        self._collection.delete(expr)
        self._collection.flush()

    def _get_embeddings(self, ids):
        ids = ['"' + str(entry) + '"' for entry in ids]
        expr = f"""pk in [{','.join(ids)}]"""
        return self._collection.query(
            expr, output_fields=["pk", "sample_id", "vector"]
        )

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
            existing_ids = self._get_existing_ids(ids)
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

        self._delete_ids(ids=ids)

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
        pymilvus.utility.drop_collection(
            self.config.collection_name, using=self._alias
        )
        self._collection = None

    def _get_sample_embeddings(self, sample_ids, batch_size=1000):
        found_embeddings = []
        found_sample_ids = []

        if sample_ids is None:
            raise ValueError(
                "Milvus does not support retrieving all vectors in an index"
            )

        for batch_ids in fou.iter_batches(sample_ids, batch_size):
            response = self._get_embeddings(list(batch_ids))

            for r in response:
                found_embeddings.append(r["vector"])
                found_sample_ids.append(r["sample_id"])

        missing_ids = list(set(sample_ids) - set(found_sample_ids))

        return found_embeddings, found_sample_ids, None, missing_ids

    def _get_patch_embeddings_from_label_ids(self, label_ids, batch_size=1000):
        found_embeddings = []
        found_sample_ids = []
        found_label_ids = []

        if label_ids is None:
            raise ValueError(
                "Milvus does not support retrieving all vectors in an index"
            )

        for batch_ids in fou.iter_batches(label_ids, batch_size):
            response = self._get_embeddings(list(batch_ids))

            for r in response:
                found_embeddings.append(r["vector"])
                found_sample_ids.append(r["sample_id"])
                found_label_ids.append(r["pk"])

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
            ids = ['"' + str(entry) + '"' for entry in batch_ids]
            expr = f"""pk in [{','.join(ids)}]"""
            response = self._collection.search(
                data=[query_vector],
                anns_field="vector",
                param=self.config.search_params,
                expr=expr,
                limit=top_k,
            )
            ids = [x.id for x in response[0]]
            response = self._get_embeddings(ids)
            for r in response:
                found_embeddings.append(r["vector"])
                found_sample_ids.append(r["sample_id"])
                found_label_ids.append(r["pk"])

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
            raise ValueError("Milvus does not support full index neighbors")

        if reverse is True:
            raise ValueError(
                "Milvus does not support least similarity queries"
            )

        if k is None or k > self.config.max_k:
            raise ValueError("Milvus requires k<=%s" % self.config.max_k)

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

            expr = ['"' + str(entry) + '"' for entry in index_ids]
            expr = f"""pk in [{','.join(expr)}]"""
        else:
            expr = None

        ids = []
        dists = []
        for q in query:
            response = self._collection.search(
                data=[q.tolist()],
                anns_field="vector",
                limit=k,
                expr=expr,
                param=self.config.search_params,
            )
            ids.append([r.id for r in response[0]])
            if return_dists:
                dists.append([r.score for r in response[0]])

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
        response = self._get_embeddings(query_ids)
        query = np.array([x["vector"] for x in response])

        if single_query:
            query = query[0, :]

        return query

    def _get_dimension(self):
        if self._collection is None:
            return None

        for field in self._collection.describe()["fields"]:
            if field["name"] == "vector":
                return field["params"]["dim"]

    @classmethod
    def _from_dict(cls, d, samples, config, brain_key):
        return cls(samples, config, brain_key)
