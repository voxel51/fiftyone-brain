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
        key_field ("filepath"): the name of the FiftyOne sample field used as
            the unique identifier to match elastic documents
        patch_key_field ("id"): the name of the FiftyOne patch attribute
            field used as the unique identifier to match elastic documents, if
            ``patches_field`` is provided
        backend_key_field ("fiftyone_sample"): the name of the elastic
            document source field used as the unique identifier to match
            embeddings with FiftyOne samples
        backend_patch_key_field ("fiftyone_patch"): the name of the elastic
            document source field used to match to a patch, if
            ``patches_field`` is provided
        backend_vector_field ("vector"): the name of the elastic doc source field
            storing the embedding vector
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
        key_field="filepath",
        patch_key_field="id",
        backend_key_field="fiftyone_sample",
        backend_patch_key_field="fiftyone_patch",
        backend_vector_field="vector",
        **kwargs,
    ):
        if metric not in _SUPPORTED_METRICS:
            raise ValueError(
                "Unsupported metric '%s'. Supported values are %s"
                % (metric, tuple(_SUPPORTED_METRICS.keys()))
            )

        if (
            backend_key_field == backend_patch_key_field
            and patches_field is not None
        ):
            raise ValueError(
                "The backend_key_field and backend_patch_key_field cannot have"
                " the same value '%s'" % backend_key_field
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
        self.key_field = key_field
        self.patch_key_field = patch_key_field
        self.backend_key_field = backend_key_field
        self.backend_patch_key_field = backend_patch_key_field
        self.backend_vector_field = backend_vector_field

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
    def is_patch_index(self):
        return self.config.patches_field is not None

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

    def _get_dimension(self):
        if self.total_index_size == 0:
            return None

        if self.is_patch_index:
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
                )[self.config.index_name]["mappings"]["properties"][
                    self.config.backend_vector_field
                ][
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
                self.config.backend_vector_field: {
                    "type": "dense_vector",
                    "dims": dimension,
                    "index": "true",
                    "similarity": metric,
                },
            }
        }
        if self.config.backend_key_field != "_id":
            mappings["properties"][self.config.backend_key_field] = {
                "type": "keyword"
            }
        if (
            self.is_patch_index
            and self.config.backend_patch_key_field != "_id"
        ):
            mappings["properties"][self.config.backend_patch_key_field] = {
                "type": "keyword"
            }

        self._client.indices.create(
            index=self.config.index_name, mappings=mappings
        )
        self._metric = metric

    def _get_existing_ids(self, ids, is_labels=None):
        return_labels = False
        if self.is_patch_index and is_labels:
            key_field = self.config.backend_patch_key_field
            return_labels = True
        else:
            key_field = self.config.backend_key_field

        sample_ids, label_ids, index_ids, _ = self._get_docs(
            values=ids, key_field=key_field, include_embeddings=False
        )
        if return_labels:
            return label_ids, index_ids

        return sample_ids, index_ids

    def _remap_ids_to_keys(self, ids, key_field, incoming_key_field):
        unwind = False
        if key_field == self.config.patch_key_field:
            unwind = True
        if ids is not None and key_field != incoming_key_field:
            value_map = dict(
                zip(
                    *self.samples.values(
                        [incoming_key_field, key_field], unwind=unwind
                    )
                )
            )
            _ids = [value_map[i] for i in ids]
        else:
            _ids = ids
        return _ids

    def _remap_keys_to_ids(self, keys, key_field, outgoing_key_field):
        unwind = False
        if key_field == self.config.patch_key_field:
            unwind = True
        if keys is not None and key_field != outgoing_key_field:
            value_map = dict(
                zip(
                    *self.samples.values(
                        [key_field, outgoing_key_field], unwind=unwind
                    )
                )
            )
            _ids = [value_map.get(i, None) for i in keys]
        else:
            _ids = keys
        return _ids

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
        _incoming_key_field="id",
        _incoming_patch_key_field="id",
    ):
        if self.is_patch_index and label_ids is None:
            raise ValueError(
                "Label IDs are required to add embeddings to a patch index but"
                " none were provided. The patch field for this index is: %s"
                % self.config.patches_field
            )
        sample_ids = self._remap_ids_to_keys(
            sample_ids, self.config.key_field, _incoming_key_field
        )
        label_ids = self._remap_ids_to_keys(
            label_ids, self.config.patch_key_field, _incoming_patch_key_field
        )

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
                skip_key_field = False
                skip_patch_key_field = False

                op1 = {"index": {"_index": self.config.index_name}}
                if self.config.backend_key_field == "_id":
                    op1["index"]["_id"] = _sid
                    skip_key_field = True
                if (
                    self.is_patch_index
                    and self.config.backend_patch_key_field == "_id"
                ):
                    op1["index"]["_id"] = _id
                    skip_patch_key_field = True

                op2 = {self.config.backend_vector_field: _e}
                if not skip_key_field:
                    op2[self.config.backend_key_field] = _sid
                if self.is_patch_index and not skip_patch_key_field:
                    op2[self.config.backend_patch_key_field] = _id

                operations.append(op1)
                operations.append(op2)

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
        _incoming_key_field="id",
        _incoming_patch_key_field="id",
    ):
        sample_ids = self._remap_ids_to_keys(
            sample_ids, self.config.key_field, _incoming_key_field
        )
        label_ids = self._remap_ids_to_keys(
            label_ids, self.config.patch_key_field, _incoming_patch_key_field
        )
        is_labels = False
        if label_ids is not None:
            ids = label_ids
            is_labels = True
        else:
            ids = sample_ids

        existing_ids, index_ids = self._get_existing_ids(
            ids, is_labels=is_labels
        )
        if not allow_missing or warn_missing:
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

        operations = [
            {"delete": {"_index": self.config.index_name, "_id": i}}
            for i in index_ids
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
        _incoming_key_field="id",
        _incoming_patch_key_field="id",
    ):
        sample_ids = self._remap_ids_to_keys(
            sample_ids, self.config.key_field, _incoming_key_field
        )
        label_ids = self._remap_ids_to_keys(
            label_ids, self.config.patch_key_field, _incoming_patch_key_field
        )

        ids = sample_ids
        is_labels = False
        if label_ids is not None:
            ids = label_ids
            is_labels = True
            if not self.is_patch_index:
                # This is an index initially created on full sample
                # embeddings, but patches are attempting to be
                # accessed
                raise ValueError("This index does not support label IDs")
            if sample_ids is not None:
                logger.warning(
                    "Ignoring sample IDs when label IDs are provided"
                )

        (
            sample_ids,
            label_ids,
            embeddings,
            missing_ids,
        ) = self._get_embeddings(ids=ids, is_labels=is_labels)

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

        sample_ids = self._remap_keys_to_ids(
            sample_ids, self.config.key_field, _incoming_key_field
        )
        embeddings = np.array(embeddings)
        sample_ids = np.array(sample_ids)
        if label_ids is not None:
            label_ids = self._remap_keys_to_ids(
                label_ids,
                self.config.patch_key_field,
                _incoming_patch_key_field,
            )
            label_ids = np.array(label_ids)

        return embeddings, sample_ids, label_ids

    def _get_embeddings(self, ids=None, batch_size=1000, is_labels=False):
        key_field = None
        missing_ids = []
        if ids is not None:
            key_field = (
                self.config.backend_patch_key_field
                if is_labels
                else self.config.backend_key_field
            )
        (
            found_sample_ids,
            found_label_ids,
            _,
            found_embeddings,
        ) = self._get_docs(
            values=ids, key_field=key_field, batch_size=batch_size
        )

        if ids is not None:
            found_ids = found_label_ids if is_labels else found_sample_ids
            missing_ids = list(set(ids) - set(found_ids))

        return found_sample_ids, found_label_ids, found_embeddings, missing_ids

    def _get_docs(
        self,
        values=None,
        key_field=None,
        batch_size=1000,
        include_embeddings=True,
    ):
        must_filter = [{"exists": {"field": self.config.backend_vector_field}}]
        if key_field:
            must_filter.append({"exists": {"field": key_field}})

        if (
            self.is_patch_index
            and key_field != self.config.backend_patch_key_field
        ):
            must_filter.append(
                {"exists": {"field": self.config.backend_patch_key_field}}
            )
        elif key_field != self.config.backend_key_field:
            must_filter.append(
                {"exists": {"field": self.config.backend_key_field}}
            )

        fields = [
            self.config.backend_key_field,
            self.config.backend_patch_key_field,
        ]
        if include_embeddings:
            fields.append(self.config.backend_vector_field)

        if values is not None and key_field is not None:
            hits = self._get_docs_query(
                values, key_field, fields, must_filter, batch_size=batch_size
            )
        else:
            hits = self._get_docs_all(
                fields, must_filter, batch_size=batch_size
            )

        return self._parse_hits(hits)

    def _get_docs_query(
        self, values, key_field, fields, must_filter, batch_size=1000
    ):
        query_field = self._parse_query_field(key_field)

        hits = []
        for batch_ids in fou.iter_batches(values, batch_size):
            terms = {query_field: batch_ids}
            response = self._client.search(
                index=self.config.index_name,
                body={
                    "fields": fields,
                    "size": batch_size,
                    "_source": False,
                    "query": {
                        "bool": {"must": [*must_filter, {"terms": terms}]},
                    },
                },
            )
            hits.extend(response["hits"]["hits"])
        return hits

    def _get_docs_all(self, fields, must_filter, batch_size=1000):
        hits = []
        for batch in range(0, self.total_index_size, batch_size):
            response = self._client.search(
                index=self.config.index_name,
                body={
                    "fields": fields,
                    "from": batch,
                    "size": batch_size,
                    "_source": False,
                    "query": {"bool": {"must": must_filter}},
                },
            )
            hits.extend(response["hits"]["hits"])
        return hits

    def _parse_hits(self, hits):
        sample_ids = []
        label_ids = []
        index_ids = []
        embeddings = []

        for hit in hits:
            if hit.get("found", True):
                sample_id, label_id, vector_id, embedding = self._parse_hit(
                    hit
                )
                sample_ids.append(sample_id)
                label_ids.append(label_id)
                index_ids.append(vector_id)
                embeddings.append(embedding)

        return sample_ids, label_ids, index_ids, embeddings

    def _parse_hit(self, hit):
        label_id = None
        source_field = "_source" if "_source" in hit else "fields"

        if self.is_patch_index:
            if self.config.backend_patch_key_field == "_id":
                label_id = hit["_id"]
            else:
                label_id = hit[source_field].get(
                    self.config.backend_patch_key_field, None
                )
            if isinstance(label_id, list) and len(label_id) > 0:
                label_id = label_id[0]

        if self.config.backend_key_field == "_id":
            sample_id = hit["_id"]
        else:
            sample_id = hit[source_field].get(
                self.config.backend_key_field, None
            )
        if isinstance(sample_id, list) and len(sample_id) > 0:
            sample_id = sample_id[0]

        embedding = hit[source_field].get(
            self.config.backend_vector_field, None
        )
        vector_id = hit["_id"]

        return sample_id, label_id, vector_id, embedding

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
            if self.is_patch_index:
                key_field = self.config.patch_key_field
                backend_key_field = self.config.backend_patch_key_field
                current_ids = self.current_label_ids

            else:
                key_field = self.config.key_field
                backend_key_field = self.config.backend_key_field
                current_ids = self.current_sample_ids

            index_ids = self._remap_ids_to_keys(current_ids, key_field, "id")
            filter_field = self._parse_query_field(backend_key_field)
            _filter = {"terms": {filter_field: list(index_ids)}}
        else:
            _filter = None

        sample_ids = []
        label_ids = [] if self.is_patch_index else None
        dists = []
        for q in query:
            if self._get_metric() == _SUPPORTED_METRICS["dotproduct"]:
                q /= np.linalg.norm(q)

            knn = {
                "field": self.config.backend_vector_field,
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
                fields=[
                    self.config.backend_key_field,
                    self.config.backend_patch_key_field,
                ],
            )

            _sample_ids, _label_ids, _, _ = self._parse_hits(
                response["hits"]["hits"]
            )
            _dists = [r["_score"] for r in response["hits"]["hits"]]

            _sample_ids = self._remap_keys_to_ids(
                _sample_ids, self.config.key_field, "id"
            )
            missing_inds = [
                ind for ind, _id in enumerate(_sample_ids) if _id is None
            ]
            if _label_ids is not None and self.is_patch_index:
                _label_ids = self._remap_keys_to_ids(
                    _label_ids, self.config.patch_key_field, "id"
                )
                missing_inds.extend(
                    [ind for ind, _id in enumerate(_label_ids) if _id is None]
                )

            for i in sorted(set(missing_inds), reverse=True):
                del _sample_ids[i]
                if return_dists:
                    del _dists[i]
                if label_ids is not None:
                    del _label_ids[i]

            if return_dists:
                dists.append(_dists)
            sample_ids.append(_sample_ids)
            if self.is_patch_index:
                label_ids.append(_label_ids)

        if single_query:
            sample_ids = sample_ids[0]
            if label_ids is not None:
                label_ids = label_ids[0]
            if return_dists:
                dists = dists[0]

        if self.is_patch_index:
            ids = label_ids
        else:
            ids = sample_ids

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
        if self.is_patch_index:
            key_field = self.config.patch_key_field
            backend_key_field = self.config.backend_patch_key_field
        else:
            key_field = self.config.key_field
            backend_key_field = self.config.backend_key_field

        query_ids = self._remap_ids_to_keys(query_ids, key_field, "id")
        _, _, _, embeddings = self._get_docs(
            values=query_ids,
            key_field=backend_key_field,
            include_embeddings=True,
        )
        query = np.array(embeddings)

        if single_query:
            query = query[0, :]

        return query

    def _parse_query_field(self, key_field):
        # Text fields in elastic need to have `.keyword` appended to them to
        # use `terms` search
        mapping = self._client.indices.get_mapping(
            index=self.config.index_name
        )
        properties = mapping[self.config.index_name]["mappings"]["properties"]
        if key_field not in properties:
            raise ValueError(
                "Field %s not found in elastic index %s"
                % (key_field, self.config.index_name)
            )
        field_type = properties[key_field]["type"]
        if field_type == "text":
            return key_field + ".keyword"
        else:
            return key_field

    @classmethod
    def _from_dict(cls, d, samples, config, brain_key):
        return cls(samples, config, brain_key)
