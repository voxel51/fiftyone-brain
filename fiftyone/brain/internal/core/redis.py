"""
Redis similarity backend.

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

redis = fou.lazy_import("redis")


logger = logging.getLogger(__name__)

_SUPPORTED_METRICS = {
    "cosine": "COSINE",
    "dotproduct": "IP",
    "euclidean": "L2",
}


class RedisSimilarityConfig(SimilarityConfig):
    """Configuration for the Redis similarity backend.

    Args:
        embeddings_field (None): the sample field containing the embeddings,
            if one was provided
        model (None): the :class:`fiftyone.core.models.Model` or name of the
            zoo model that was used to compute embeddings, if known
        patches_field (None): the sample field defining the patches being
            analyzed, if any
        supports_prompts (None): whether this run supports prompt queries
        index_name (None): the name of a Redis index to use or create. If none
            is provided, a new index will be created
        metric ("cosine"): the embedding distance metric to use when creating a
            new index. Supported values are
            ``("cosine", "dotproduct", "euclidean")``
        algorithm ("FLAT"): the search algorithm to use. The supported values
            are ``("FLAT", "HNSW")``
        host ("localhost"): the host to use
        port (6379): the port to use
        db (0): the database to use
        username (None): a username to use
        password (None): a password to use
    """

    def __init__(
        self,
        embeddings_field=None,
        model=None,
        patches_field=None,
        supports_prompts=None,
        index_name=None,
        metric="cosine",
        algorithm="FLAT",
        host="localhost",
        port=6379,
        db=0,
        username=None,
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

        self.index_name = index_name
        self.metric = metric
        self.algorithm = algorithm

        # store privately so these aren't serialized
        self._host = host
        self._port = port
        self._db = db
        self._username = username
        self._password = password

    @property
    def method(self):
        return "redis"

    @property
    def host(self):
        return self._host

    @host.setter
    def host(self, value):
        self._host = value

    @property
    def port(self):
        return self._port

    @port.setter
    def port(self, value):
        self._port = value

    @property
    def db(self):
        return self._db

    @db.setter
    def db(self, value):
        self._db = value

    @property
    def username(self):
        return self._username

    @username.setter
    def username(self, value):
        self._username = value

    @property
    def password(self):
        return self._password

    @password.setter
    def password(self, value):
        self._password = value

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
        self, host=None, port=None, db=None, username=None, password=None
    ):
        self._load_parameters(
            host=host, port=port, db=db, username=username, password=password
        )


class RedisSimilarity(Similarity):
    """Redis similarity factory.

    Args:
        config: a :class:`RedisSimilarityConfig`
    """

    def ensure_requirements(self):
        fou.ensure_package("redis")

    def ensure_usage_requirements(self):
        fou.ensure_package("redis")

    def initialize(self, samples, brain_key):
        return RedisSimilarityIndex(
            samples, self.config, brain_key, backend=self
        )


class RedisSimilarityIndex(SimilarityIndex):
    """Class for interacting with Redis similarity indexes.

    Args:
        samples: the :class:`fiftyone.core.collections.SampleCollection` used
        config: the :class:`RedisSimilarityConfig` used
        brain_key: the brain key
        backend (None): a :class:`RedisSimilarity` instance
    """

    def __init__(self, samples, config, brain_key, backend=None):
        super().__init__(samples, config, brain_key, backend=backend)
        self._client = None
        self._index = None
        self._initialize()

    def _initialize(self):
        client = redis.Redis(
            host=self.config.host,
            port=self.config.port,
            db=self.config.db,
            username=self.config.username,
            password=self.config.password,
            decode_responses=True,
        )

        if self.config.index_name is None:

            def index_exists(index_name):
                try:
                    client.ft(index_name).info()
                    return True
                except:
                    return False

            root = "fiftyone-" + fou.to_slug(self._samples._root_dataset.name)
            index_name = fbu.get_unique_name(root, index_exists)

            self.config.index_name = index_name
            self.save_config()

        try:
            index = client.ft(self.config.index_name)
            index.info()
        except:
            index = None

        self._client = client
        self._index = index

    def _create_index(self, dimension):
        from redis.commands.search.field import TagField, VectorField
        from redis.commands.search.indexDefinition import (
            IndexDefinition,
            IndexType,
        )

        schema = (
            TagField("$.foid", as_name="foid"),
            TagField("$.sample_id", as_name="sample_id"),
            VectorField(
                "$.vector",
                self.config.algorithm,
                {
                    "TYPE": "FLOAT32",
                    "DIM": dimension,
                    "DISTANCE_METRIC": _SUPPORTED_METRICS[self.config.metric],
                },
                as_name="vector",
            ),
        )
        definition = IndexDefinition(
            prefix=[self.config.index_name + ":"],
            index_type=IndexType.JSON,
        )
        index = self._client.ft(self.config.index_name)
        index.create_index(fields=schema, definition=definition)

        self._index = index

    @property
    def client(self):
        """The ``redis.client.Redis`` instance for this index."""
        return self._client

    @property
    def total_index_size(self):
        try:
            return int(self._index.info()["num_docs"])
        except:
            return 0

    def add_to_index(
        self,
        embeddings,
        sample_ids,
        label_ids=None,
        overwrite=True,
        allow_existing=True,
        warn_existing=False,
        reload=True,
    ):
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

        elif existing_ids and overwrite:
            self._delete_ids(existing_ids)

        pipeline = self._client.pipeline()
        for e, id, sample_id in zip(embeddings, ids, sample_ids):
            key = f"{self.config.index_name}:{id}"
            d = {
                "foid": id,
                "sample_id": sample_id,
                "vector": e.astype(np.float32).tolist(),
            }
            pipeline.json().set(key, "$", d)

        pipeline.execute()

        if reload:
            self.reload()

    def _get_existing_ids(self, ids):
        return [d["foid"] for d in self._get_values(ids)]

    def _delete_ids(self, ids):
        keys = [f"{self.config.index_name}:{id}" for id in ids]
        self._client.delete(*keys)

    def _get_values(self, ids):
        pipeline = self._client.pipeline()
        for id in ids:
            pipeline.json().get(f"{self.config.index_name}:{id}")

        return [d for d in pipeline.execute() if d is not None]

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
        if self._index is None:
            return

        self._index.dropindex(delete_documents=True)
        self._index = None

    def _get_sample_embeddings(self, sample_ids, batch_size=1000):
        found_embeddings = []
        found_sample_ids = []

        if sample_ids is None:
            get_id = lambda key: key.rsplit(":", 1)[1]
            keys = self._client.keys(f"{self.config.index_name}:*")
            sample_ids = map(get_id, keys)

        for batch_ids in fou.iter_batches(sample_ids, batch_size):
            for d in self._get_values(batch_ids):
                found_embeddings.append(d["vector"])
                found_sample_ids.append(d["sample_id"])

        missing_ids = list(set(sample_ids) - set(found_sample_ids))

        return found_embeddings, found_sample_ids, None, missing_ids

    def _get_patch_embeddings_from_label_ids(self, label_ids, batch_size=1000):
        found_embeddings = []
        found_sample_ids = []
        found_label_ids = []

        if label_ids is None:
            get_id = lambda key: key.rsplit(":", 1)[1]
            keys = self._client.keys(f"{self.config.index_name}:*")
            label_ids = map(get_id, keys)

        for batch_ids in fou.iter_batches(label_ids, batch_size):
            for d in self._get_values(batch_ids):
                found_embeddings.append(d["vector"])
                found_sample_ids.append(d["sample_id"])
                found_label_ids.append(d["foid"])

        missing_ids = list(set(label_ids) - set(found_label_ids))

        return found_embeddings, found_sample_ids, found_label_ids, missing_ids

    def _get_patch_embeddings_from_sample_ids(
        self, sample_ids, batch_size=100
    ):
        from redis.commands.search.query import Query

        found_embeddings = []
        found_sample_ids = []
        found_label_ids = []

        for batch_ids in fou.iter_batches(sample_ids, batch_size):
            filter = "@sample_id:{ " + " | ".join(batch_ids) + " }"
            query = Query(filter).dialect(2)
            for doc in self._index.search(query).docs:
                found_embeddings.append(doc.embeddings)
                found_sample_ids.append(doc.sample_id)
                found_label_ids.append(doc.foid)

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
        from redis.commands.search.query import Query

        if query is None:
            raise ValueError("Redis does not support full index neighbors")

        if reverse is True:
            raise ValueError("Redis does not support least similarity queries")

        if k is None:
            k = self.index_size

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
                index_ids = list(self.current_label_ids)
            else:
                index_ids = list(self.current_sample_ids)

            filter = "@foid:{ " + " | ".join(index_ids) + " }"
        else:
            filter = "*"

        ids = []
        dists = []
        for q in query:
            _query = (
                Query(f"({filter})=>[KNN {k} @vector $query AS score]")
                .sort_by("score")
                .return_fields("score", "foid")
                .dialect(2)
            )
            _q = q.astype(np.float32).tobytes()
            docs = self._index.search(_query, {"query": _q}).docs

            ids.append([doc.foid for doc in docs])
            if return_dists:
                dists.append([doc.score for doc in docs])

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
        dicts = self._get_values(query_ids)
        if not dicts:
            raise ValueError(
                "Query IDs %s do not exist in this index" % query_ids
            )

        query = np.array([d["vector"] for d in dicts])

        if single_query:
            query = query[0, :]

        return query

    @classmethod
    def _from_dict(cls, d, samples, config, brain_key):
        return cls(samples, config, brain_key)
