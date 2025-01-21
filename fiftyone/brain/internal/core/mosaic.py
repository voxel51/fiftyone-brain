"""
Redis similarity backend.

| Copyright 2017-2025, Voxel51, Inc.
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

vector_search_client = fou.lazy_import("databricks.vector_search.client")


logger = logging.getLogger(__name__)

# Todo: add in required for arguments that are necessary to create the index table
class MosaicSimilarityConfig(SimilarityConfig):
    """Configuration for the Mosaic similarity backend.

    Args:
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
        **kwargs: keyword arguments for
            :class:`fiftyone.brain.similarity.SimilarityConfig`
    """

    def __init__(
        self,
        endpoint_name=None,
        workspace_url=None,
        catalog_name=None,
        schema_name=None,
        index_name=None,
        service_principal_client_id=None,
        service_principal_client_secret=None,
        personal_access_token=None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.index_name = index_name
        self.endpoint_name = endpoint_name
        self.catalog_name = catalog_name
        self.schema_name = schema_name

        # store privately so these aren't serialized
        self._workspace_url = workspace_url
        self._service_principal_client_id = service_principal_client_id
        self._service_principal_client_secret = service_principal_client_secret
        self._personal_access_token = personal_access_token

    @property
    def method(self):
        return "mosaic"

    @property
    def workspace_url(self):
        return self._workspace_url

    @workspace_url.setter
    def workspace_url(self, workspace_url):
        self._workspace_url = workspace_url

    @property
    def service_principal_client_id(self):
        return self._service_principal_client_id

    @service_principal_client_id.setter
    def service_principal_client_id(self, service_principal_client_id):
        self._service_principal_client_id = service_principal_client_id

    @property
    def service_principal_client_secret(self):
        return self._service_principal_client_secret

    @service_principal_client_secret.setter
    def service_principal_client_secret(self, service_principal_client_secret):
        self._service_principal_client_secret = service_principal_client_secret

    @property
    def personal_access_token(self):
        return self._personal_access_token

    @personal_access_token.setter
    def personal_access_token(self, personal_access_token):
        self._personal_access_token = personal_access_token

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
        self,
        workspace_url=None,
        service_principal_client_id=None,
        service_principal_client_secret=None,
        personal_access_token=None,
    ):
        self._load_parameters(
            workspace_url=workspace_url,
            service_principal_client_id=service_principal_client_id,
            service_principal_client_secret=service_principal_client_secret,
            personal_access_token=personal_access_token,
        )


class MosaicSimilarity(Similarity):
    """Mosaic similarity factory.

    Args:
        config: a :class:`RedisSimilarityConfig`
    """

    def ensure_requirements(self):
        fou.ensure_package("databricks-vectorsearch")

    def ensure_usage_requirements(self):
        fou.ensure_package("databricks-vectorsearch")

    def initialize(self, samples, brain_key):
        return MosaicSimilarityIndex(
            samples, self.config, brain_key, backend=self
        )


class MosaicSimilarityIndex(SimilarityIndex):
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
        self._client = vector_search_client.VectorSearchClient(
            workspace_url=self.config.workspace_url,
            service_principal_client_id=self.config.service_principal_client_id,
            service_principal_client_secret=self.config.service_principal_client_secret,
            personal_access_token=self.config.personal_access_token,
        )

        try:
            index_names_result = self._client.list_indexes(
                self.config.endpoint_name
            )
        except Exception as e:
            raise ValueError(
                f"Failed to list indexes from endpoint :{self.config.endpoint_name}"
            ) from e

        index_prefix = f"{self.config.catalog_name}.{self.config.schema_name}."
        index_names = [
            ind["name"].replace(index_prefix, "")
            for ind in index_names_result["vector_indexes"]
            if ind["name"].startswith(index_prefix)
        ]

        if self.config.index_name is None:
            root = "fiftyone-" + fou.to_slug(self._samples._root_dataset.name)
            index_name = fbu.get_unique_name(root, index_names)

            self.config.index_name = index_name
            self.save_config()

        if self.config.index_name in index_names:
            index = self._client.get_index(
                endpoint_name=self.config.endpoint_name,
                index_name=f"{index_prefix}{self.config.index_name}",
            )
        else:
            index = None

        self._index = index

    def _create_index(self, dimension):
        self._index = self._client.create_direct_access_index(
            endpoint_name=self.config.endpoint_name,
            index_name=f"{self.config.catalog_name}.{self.config.schema_name}.{self.config.index_name}",
            primary_key="foid",
            embedding_dimension=dimension,
            embedding_vector_column="embedding_vector",
            schema={
                "foid": "string",
                "sample_id": "string",
                "embedding_vector": "array<float>",
            },
        )

    @property
    def client(self):
        """The ``databricks.vector_search.client.VectorSearchClient`` instance for this index."""
        return self._client

    @property
    def total_index_size(self):
        # TEST THIS
        return self._index.describe()["status"]["indexed_row_count"]

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
        if self._index is None:
            self._create_index(embeddings.shape[1])

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
            embeddings = np.delete(embeddings, del_inds, axis=0)
            sample_ids = np.delete(sample_ids, del_inds)
            if label_ids is not None:
                label_ids = np.delete(label_ids, del_inds)

        for _embeddings, _ids, _sample_ids in zip(
            fou.iter_batches(embeddings, batch_size),
            fou.iter_batches(ids, batch_size),
            fou.iter_batches(sample_ids, batch_size),
        ):
            result = [
                {"foid": f, "sample_id": s, "embedding_vector": list(e)}
                for s, f, e in zip(_ids, _sample_ids, _embeddings)
            ]
            self._index.upsert(result)

        if reload:
            self.reload()

    def _get_index_ids(self, batch_size=1000):
        ids = set()
        last_primary_key = None
        result = self._index.scan(num_results=batch_size)
        while len(result) > 0:
            ids.update(
                [
                    doc["fields"][0]["value"]["string_value"]
                    for doc in result["data"]
                ]
            )
            last_primary_key = result["last_primary_key"]
            result = self._index.scan(
                num_results=batch_size, last_primary_key=last_primary_key
            )
        return list(ids)

    def _get_values(self, ids, batch_size=1000):

        embeddings = []
        last_primary_key = None
        result = self._index.scan(num_results=batch_size)
        while len(result) > 0:
            for doc in result["data"]:
                foid = doc["fields"][0]["value"]["string_value"]
                if foid in ids:
                    embedding = [
                        d["number_value"]
                        for d in doc["fields"][2]["value"]["list_value"][
                            "values"
                        ]
                    ]
                    embeddings.append(embedding)
            last_primary_key = result["last_primary_key"]
            result = self._index.scan(
                num_results=batch_size, last_primary_key=last_primary_key
            )

        return embeddings

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
            existing_ids = self._get_index_ids()
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

        self._index.delete(ids)

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

    # Note: might be an arg in delete_brain_run?
    def cleanup(self):
        self._client.delete_index(self.config.index_name)

    def _get_sample_embeddings(self, sample_ids, batch_size=1000):
        found_embeddings = []
        found_sample_ids = []

        if sample_ids is None:
            sample_ids = self._get_index_ids()

        last_primary_key = None
        result = self._index.scan(num_results=batch_size)
        while len(result) > 0:
            for doc in result["data"]:
                sample_id = doc["fields"][1]["value"]["string_value"]
                if sample_id in sample_ids:
                    embedding = [
                        d["number_value"]
                        for d in doc["fields"][2]["value"]["list_value"][
                            "values"
                        ]
                    ]
                    found_embeddings.append(embedding)
                    found_sample_ids.append(sample_id)
            last_primary_key = result["last_primary_key"]
            result = self._index.scan(
                num_results=batch_size, last_primary_key=last_primary_key
            )

        missing_ids = list(set(sample_ids) - set(found_sample_ids))

        return found_embeddings, found_sample_ids, None, missing_ids

    def _get_patch_embeddings_from_label_ids(self, label_ids, batch_size=1000):
        found_embeddings = []
        found_sample_ids = []
        found_label_ids = []

        if label_ids is None:
            label_ids = self._get_index_ids()

        last_primary_key = None
        result = self._index.scan(num_results=batch_size)
        while len(result) > 0:
            for doc in result["data"]:
                label_id = doc["fields"][0]["value"]["string_value"]
                if label_id in label_ids:
                    embedding = [
                        d["number_value"]
                        for d in doc["fields"][2]["value"]["list_value"][
                            "values"
                        ]
                    ]
                    found_embeddings.append(embedding)
                    found_label_ids.append(label_id)
                    found_sample_ids.append(
                        doc["fields"][1]["value"]["string_value"]
                    )
            last_primary_key = result["last_primary_key"]
            result = self._index.scan(
                num_results=batch_size, last_primary_key=last_primary_key
            )

        missing_ids = list(set(label_ids) - set(found_label_ids))

        return found_embeddings, found_sample_ids, found_label_ids, missing_ids

    def _get_patch_embeddings_from_sample_ids(
        self, sample_ids, batch_size=100
    ):
        found_embeddings = []
        found_sample_ids = []
        found_label_ids = []

        last_primary_key = None
        result = self._index.scan(num_results=batch_size)
        while len(result) > 0:
            for doc in result["data"]:
                sample_id = doc["fields"][1]["value"]["string_value"]
                if sample_id in sample_ids:
                    embedding = [
                        d["number_value"]
                        for d in doc["fields"][2]["value"]["list_value"][
                            "values"
                        ]
                    ]
                    found_embeddings.append(embedding)
                    found_sample_ids.append(sample_id)
                    found_label_ids.append(
                        doc["fields"][0]["value"]["string_value"]
                    )
            last_primary_key = result["last_primary_key"]
            result = self._index.scan(
                num_results=batch_size, last_primary_key=last_primary_key
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
            raise ValueError("Mosaic does not support full index neighbors")

        if reverse is True:
            raise ValueError(
                "Mosaic does not support least similarity queries"
            )

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

            _filter = {"foid": list(index_ids)}
        else:
            _filter = None

        ids = []
        dists = []
        for q in query:
            results = self._index.similarity_search(
                columns=["foid"],
                query_vector=list(q),
                filters=_filter,
                num_results=k,
            )

            ids.append([res[0] for res in results["result"]["data_array"]])
            if return_dists:
                dists.append(
                    [res[1] for res in results["result"]["data_array"]]
                )

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
        embeddings = self._get_values(query_ids)
        if len(embeddings) == 0:
            raise ValueError(
                "Query IDs %s do not exist in this index" % query_ids
            )
        query = np.array(embeddings)

        if single_query:
            query = query[0, :]

        return query

    @classmethod
    def _from_dict(cls, d, samples, config, brain_key):
        return cls(samples, config, brain_key)
