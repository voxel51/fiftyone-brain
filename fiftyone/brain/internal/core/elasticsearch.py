"""
Elastisearch similarity backend.

| Copyright 2017-2024, Voxel51, Inc.
| `voxel51.com <https://voxel51.com/>`_
|
"""
import logging

import numpy as np

import eta.core.utils as etau

from fiftyone import ViewField as F
import fiftyone.core.utils as fou
import fiftyone.brain.internal.core.utils as fbu
from fiftyone.brain.similarity import (
    SimilarityConfig,
    Similarity,
    SimilarityIndex,
)

es = fou.lazy_import("elasticsearch")


logger = logging.getLogger(__name__)

_SUPPORTED_METRICS = {
    "cosine": "cosine",
    "dotproduct": "dot_product",
    "euclidean": "l2_norm",
    "innerproduct": "max_inner_product",
}


class ElasticsearchSimilarityConfig(SimilarityConfig):
    """Configuration for a Elasticsearch similarity instance.

    Args:
        embeddings_field (None): the sample field containing the embeddings
        model (None): the :class:`fiftyone.core.models.Model` or name of the
            zoo model that was used to compute embeddings, if known
        patches_field (None): the sample field defining the patches being
            analyzed, if any
        supports_prompts (None): whether this run supports prompt queries
        index_name (None): the name of the Elasticsearch index to use or
            create. If none is provided, a new index will be created
        metric ("cosine"): the embedding distance metric to use when creating a
            new index. Supported values are
            ``("cosine", "dotproduct", "euclidean", "innerproduct")``
        hosts (None): the full Elasticsearch server address(es) to use. Can be
            a string or list of strings
        cloud_id (None): the Cloud ID of an Elastic Cloud to connect to
        username (None): a username to use
        password (None): a password to use
        api_key (None): an API key to use
        ca_certs (None): a path to a CA certificate
        bearer_auth (None): a bearer token to use
        ssl_assert_fingerprint (None): a SHA256 fingerprint to use
        verify_certs (None): whether to verify SSL certificates
        **kwargs: keyword arguments for :class:`SimilarityConfig`
    """

    def __init__(
        self,
        embeddings_field=None,
        model=None,
        patches_field=None,
        supports_prompts=None,
        index_name=None,
        metric="cosine",
        hosts=None,
        cloud_id=None,
        username=None,
        password=None,
        api_key=None,
        ca_certs=None,
        bearer_auth=None,
        ssl_assert_fingerprint=None,
        verify_certs=None,
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

        self._hosts = hosts
        self._cloud_id = cloud_id
        self._username = username
        self._password = password
        self._api_key = api_key
        self._ca_certs = ca_certs
        self._bearer_auth = bearer_auth
        self._ssl_assert_fingerprint = ssl_assert_fingerprint
        self._verify_certs = verify_certs

    @property
    def method(self):
        return "elasticsearch"

    @property
    def hosts(self):
        return self._hosts

    @hosts.setter
    def hosts(self, value):
        self._hosts = value

    @property
    def cloud_id(self):
        return self._cloud_id

    @cloud_id.setter
    def cloud_id(self, value):
        self._cloud_id = value

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
    def api_key(self):
        return self._api_key

    @api_key.setter
    def api_key(self, value):
        self._api_key = value

    @property
    def ca_certs(self):
        return self._ca_certs

    @ca_certs.setter
    def ca_certs(self, value):
        self._ca_certs = value

    @property
    def bearer_auth(self):
        return self._bearer_auth

    @bearer_auth.setter
    def bearer_auth(self, value):
        self._bearer_auth = value

    @property
    def ssl_assert_fingerprint(self):
        return self._ssl_assert_fingerprint

    @ssl_assert_fingerprint.setter
    def ssl_assert_fingerprint(self, value):
        self._ssl_assert_fingerprint = value

    @property
    def verify_certs(self):
        return self._verify_certs

    @verify_certs.setter
    def verify_certs(self, value):
        self._verify_certs = value

    @property
    def max_k(self):
        return 10000  # Elasticsearch limit

    @property
    def supports_least_similarity(self):
        return False

    @property
    def supported_aggregations(self):
        return ("mean",)

    def load_credentials(
        self,
        hosts=None,
        cloud_id=None,
        username=None,
        password=None,
        api_key=None,
        ca_certs=None,
        bearer_auth=None,
        ssl_assert_fingerprint=None,
        verify_certs=None,
    ):
        self._load_parameters(
            hosts=hosts,
            cloud_id=cloud_id,
            username=username,
            password=password,
            api_key=api_key,
            ca_certs=ca_certs,
            bearer_auth=bearer_auth,
            ssl_assert_fingerprint=ssl_assert_fingerprint,
            verify_certs=verify_certs,
        )


class ElasticsearchSimilarity(Similarity):
    """Elasticsearch similarity factory.

    Args:
        config: a :class:`ElasticsearchSimilarityConfig`
    """

    def ensure_requirements(self):
        fou.ensure_package("elasticsearch")

    def ensure_usage_requirements(self):
        fou.ensure_package("elasticsearch")

    def initialize(self, samples, brain_key):
        return ElasticsearchSimilarityIndex(
            samples, self.config, brain_key, backend=self
        )


class ElasticsearchSimilarityIndex(SimilarityIndex):
    """Class for interacting with Elasticsearch similarity indexes.

    Args:
        samples: the :class:`fiftyone.core.collections.SampleCollection` used
        config: the :class:`ElasticsearchSimilarityConfig` used
        brain_key: the brain key
        backend (None): a :class:`ElasticsearchSimilarity` instance
    """

    def __init__(self, samples, config, brain_key, backend=None):
        super().__init__(samples, config, brain_key, backend=backend)
        self._client = None
        self._metric = None
        self._initialize()

    @property
    def total_index_size(self):
        try:
            return self._client.count(index=self.config.index_name)["count"]
        except:
            return 0

    @property
    def client(self):
        """The ``elasticsearch.Elasticsearch`` instance for this index."""
        return self._client

    def _initialize(self):
        kwargs = {}

        for key in (
            "hosts",
            "cloud_id",
            "username",
            "password",
            "api_key",
            "ca_certs",
            "bearer_auth",
            "ssl_assert_fingerprint",
            "verify_certs",
        ):
            value = getattr(self.config, key, None)
            if value is not None:
                kwargs[key] = value

        username = kwargs.pop("username", None)
        password = kwargs.pop("password", None)
        if username is not None and password is not None:
            kwargs["basic_auth"] = (username, password)

        try:
            self._client = es.Elasticsearch(**kwargs)
        except Exception as e:
            raise ValueError(
                "Failed to connect to Elasticsearch backend. Refer to "
                "https://docs.voxel51.com/integrations/elasticsearch.html for more "
                "information"
            ) from e

        if self.config.index_name is None:
            root = "fiftyone-" + fou.to_slug(self.samples._root_dataset.name)
            index_name = fbu.get_unique_name(root, self._get_index_names())

            self.config.index_name = index_name
            self.save_config()

    def _get_index_names(self):
        return self._client.indices.get_alias().keys()

    def _get_index_ids(self, batch_size=1000):
        sample_ids = []
        label_ids = []
        for batch in range(0, self.total_index_size, batch_size):
            response = self._client.search(
                index=self.config.index_name,
                body={
                    "fields": ["sample_id"],
                    "from": batch,
                    "query": {
                        "bool": {
                            "must": [
                                {"exists": {"field": "vector"}},
                                {"exists": {"field": "sample_id"}},
                            ]
                        }
                    },
                },
                source=False,
                size=batch_size,
            )
            for doc in response["hits"]["hits"]:
                sample_id = doc["fields"]["sample_id"][0]
                sample_or_label_id = doc["_id"]
                sample_ids.append(sample_id)
                label_ids.append(sample_or_label_id)

        return sample_ids, label_ids

    def _get_dimension(self):
        if self.total_index_size == 0:
            return None

        if self.config.patches_field is not None:
            embeddings, _, _ = self.get_embeddings(
                label_ids=self._label_ids[:1]
            )
        else:
            embeddings, _, _ = self.get_embeddings(
                sample_ids=self._sample_ids[:1]
            )

        return embeddings.shape[1]

    def _get_metric(self):
        if self._metric is None:
            try:
                # We must ask ES rather than using `self.config.metric` because
                # we may be working with a preexisting index
                self._metric = self._client.indices.get_mapping(
                    index=self.config.index_name
                )[self.config.index_name]["mappings"]["properties"]["vector"][
                    "similarity"
                ]
            except:
                logger.warning(
                    "Failed to infer similarity metric from index '%s'",
                    self.config.index_name,
                )

        return self._metric

    def _index_exists(self):
        if self.config.index_name is None:
            return False

        return self.config.index_name in self._get_index_names()

    def _create_index(self, dimension):
        metric = _SUPPORTED_METRICS[self.config.metric]
        mappings = {
            "properties": {
                "vector": {
                    "type": "dense_vector",
                    "dims": dimension,
                    "index": "true",
                    "similarity": metric,
                }
            }
        }
        self._client.indices.create(
            index=self.config.index_name, mappings=mappings
        )
        self._metric = metric

    def _get_existing_ids(self, ids):
        docs = [{"_index": self.config.index_name, "_id": i} for i in ids]
        resp = self._client.mget(docs=docs)
        return [d["_id"] for d in resp["docs"] if d["found"]]

    def add_to_index(
        self,
        embeddings,
        sample_ids,
        label_ids=None,
        overwrite=True,
        allow_existing=True,
        warn_existing=False,
        reload=True,
        batch_size=500,
    ):
        if not self._index_exists():
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
            embeddings = np.delete(embeddings, del_inds, axis=0)
            sample_ids = np.delete(sample_ids, del_inds)
            if label_ids is not None:
                label_ids = np.delete(label_ids, del_inds)

        if self._get_metric() == _SUPPORTED_METRICS["dotproduct"]:
            embeddings /= np.linalg.norm(embeddings, axis=1)[:, np.newaxis]

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
            operations = []
            for _e, _id, _sid in zip(_embeddings, _ids, _sample_ids):
                operations.append(
                    {"index": {"_index": self.config.index_name, "_id": _id}}
                )
                operations.append({"sample_id": _sid, "vector": _e})

            self._client.bulk(
                index=self.config.index_name,
                operations=operations,
                refresh=True,
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
            existing_ids = self._get_existing_ids(ids)
            missing_ids = set(ids) - set(existing_ids)
            num_missing = len(missing_ids)

            if num_missing > 0:
                if not allow_missing:
                    raise ValueError(
                        "Found %d IDs (eg %s) that are not present in the "
                        "index" % (num_missing, next(iter(missing_ids)))
                    )

                if warn_missing:
                    logger.warning(
                        "Ignoring %d IDs that are not present in the index",
                        num_missing,
                    )

                ids = existing_ids

        operations = [
            {"delete": {"_index": self.config.index_name, "_id": i}}
            for i in ids
        ]
        self._client.bulk(body=operations, refresh=True)

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

    def _parse_embeddings_response(self, response, label_id=True):
        found_embeddings = []
        found_sample_ids = []
        found_label_ids = []
        for r in response:
            if r.get("found", True):
                found_embeddings.append(r["_source"]["vector"])
                if label_id:
                    found_sample_ids.append(r["_source"]["sample_id"])
                    found_label_ids.append(r["_id"])
                else:
                    found_sample_ids.append(r["_id"])

        return found_embeddings, found_sample_ids, found_label_ids

    def _get_sample_embeddings(self, sample_ids, batch_size=1000):
        found_embeddings = []
        found_sample_ids = []

        if sample_ids is None:
            sample_ids, label_ids = self._get_index_ids(batch_size=batch_size)

        for batch_ids in fou.iter_batches(sample_ids, batch_size):
            response = self._client.mget(
                index=self.config.index_name, ids=batch_ids, source=True
            )

            (
                _found_embeddings,
                _found_sample_ids,
                _,
            ) = self._parse_embeddings_response(
                response["docs"], label_id=False
            )
            found_embeddings += _found_embeddings
            found_sample_ids += _found_sample_ids

        missing_ids = list(set(sample_ids) - set(found_sample_ids))

        return found_embeddings, found_sample_ids, None, missing_ids

    def _get_patch_embeddings_from_label_ids(self, label_ids, batch_size=1000):
        found_embeddings = []
        found_sample_ids = []
        found_label_ids = []

        if label_ids is None:
            sample_ids, label_ids = self._get_index_ids(batch_size=batch_size)

        for batch_ids in fou.iter_batches(label_ids, batch_size):
            response = self._client.mget(
                index=self.config.index_name, ids=batch_ids, source=True
            )

            (
                _found_embeddings,
                _found_sample_ids,
                _found_label_ids,
            ) = self._parse_embeddings_response(response["docs"])
            found_embeddings += _found_embeddings
            found_sample_ids += _found_sample_ids
            found_label_ids += _found_label_ids

        missing_ids = list(set(label_ids) - set(found_label_ids))

        return found_embeddings, found_sample_ids, found_label_ids, missing_ids

    def _get_patch_embeddings_from_sample_ids(
        self, sample_ids, batch_size=100
    ):
        found_embeddings = []
        found_sample_ids = []
        found_label_ids = []

        if sample_ids is None:
            sample_ids, label_ids = self._get_index_ids(batch_size=batch_size)

        for batch_ids in fou.iter_batches(sample_ids, batch_size):
            response = self._client.search(
                index=self.config.index_name,
                body={"query": {"terms": {"sample_id": sample_ids}}},
            )

            (
                _found_embeddings,
                _found_sample_ids,
                _found_label_ids,
            ) = self._parse_embeddings_response(response["hits"]["hits"])
            found_embeddings += _found_embeddings
            found_sample_ids += _found_sample_ids
            found_label_ids += _found_label_ids

        missing_ids = list(set(sample_ids) - set(found_sample_ids))

        return found_embeddings, found_sample_ids, found_label_ids, missing_ids

    def cleanup(self):
        self._client.indices.delete(
            index=self.config.index_name, ignore_unavailable=True
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
            raise ValueError(
                "Elasticsearch does not support full index neighbors"
            )

        if reverse is True:
            raise ValueError(
                "Elasticsearch does not support least similarity queries"
            )

        if aggregation not in (None, "mean"):
            raise ValueError(
                f"Elasticsearch does not support {aggregation} aggregation"
            )

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

            _filter = {"terms": {"_id": list(index_ids)}}
        else:
            _filter = None

        ids = []
        dists = []
        for q in query:
            if self._get_metric() == _SUPPORTED_METRICS["dotproduct"]:
                q /= np.linalg.norm(q)

            knn = {
                "field": "vector",
                "query_vector": q.tolist(),
                "k": k,
                "num_candidates": 10 * k,
            }
            if _filter:
                knn["filter"] = _filter

            response = self._client.search(
                index=self.config.index_name,
                knn=knn,
                size=k,
            )
            ids.append([r["_id"] for r in response["hits"]["hits"]])
            if return_dists:
                dists.append([r["_score"] for r in response["hits"]["hits"]])

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
        response = self._client.mget(
            index=self.config.index_name, ids=query_ids, source=True
        )
        query = np.array(
            [r["_source"]["vector"] for r in response["docs"] if r["found"]]
        )

        if single_query:
            query = query[0, :]

        return query

    @classmethod
    def _from_dict(cls, d, samples, config, brain_key):
        return cls(samples, config, brain_key)
