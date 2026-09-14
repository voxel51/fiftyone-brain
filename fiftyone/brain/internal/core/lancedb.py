"""
LanceDB similarity backend.

| Copyright 2017-2026, Voxel51, Inc.
| `voxel51.com <https://voxel51.com/>`_
|
"""
import logging
from copy import deepcopy

import numpy as np

import eta.core.utils as etau

import fiftyone.core.storage as fos
import fiftyone.core.utils as fou
import fiftyone.brain.internal.core.utils as fbu
from fiftyone.brain.similarity import (
    SimilarityConfig,
    Similarity,
    SimilarityIndex,
)

lancedb = fou.lazy_import("lancedb")
pa = fou.lazy_import("pyarrow")


_SUPPORTED_METRICS = {
    "cosine": "cosine",
    "euclidean": "l2",
}

logger = logging.getLogger(__name__)


class LanceDBSimilarityConfig(SimilarityConfig):
    """Configuration for a LanceDB similarity instance.

    Args:
        table_name (None): the name of the LanceDB table to use. If none is
            provided, a new table will be created
        metric ("cosine"): the embedding distance metric to use when creating a
            new index. Supported values are ``("cosine", "euclidean")``
        uri ("/tmp/lancedb"): the database URI to use
        storage_options (None): a dict of storage options for the object store
            backing ``uri``, eg credentials for a cloud bucket. Passed through
            to ``lancedb.connect()``. Like ``uri``, this is not serialized, so
            it must be supplied again each time the index is loaded. A value
            given here replaces any configured for the backend rather than
            merging with it
        **kwargs: keyword arguments for :class:`SimilarityConfig`
    """

    def __init__(
        self,
        table_name=None,
        metric="cosine",
        uri="/tmp/lancedb",
        storage_options=None,
        **kwargs,
    ):
        if metric not in _SUPPORTED_METRICS:
            raise ValueError(
                "Unsupported metric '%s'. Supported values are %s"
                % (metric, tuple(_SUPPORTED_METRICS.keys()))
            )

        super().__init__(**kwargs)

        self.table_name = table_name
        self.metric = metric

        # store privately so these aren't serialized
        self._uri = uri
        self.storage_options = storage_options

    @property
    def method(self):
        return "lancedb"

    @property
    def uri(self):
        return self._uri

    @uri.setter
    def uri(self, value):
        self._uri = value

    @property
    def storage_options(self):
        return self._storage_options

    @storage_options.setter
    def storage_options(self, value):
        # Copy: `_load_parameters` hands over the dict owned by the global
        # brain config, and refreshing a credential in place would rewrite it
        # for every other index in the process
        self._storage_options = deepcopy(value)

    @property
    def max_k(self):
        return None

    @property
    def supports_least_similarity(self):
        return False

    @property
    def supported_aggregations(self):
        return ("mean",)

    def load_credentials(self, uri=None, storage_options=None):
        self._load_parameters(uri=uri, storage_options=storage_options)


class LanceDBSimilarity(Similarity):
    """LanceDB similarity factory.

    Args:
        config: a :class:`LanceDBSimilarityConfig`
    """

    def ensure_requirements(self):
        fou.ensure_package("lancedb")

    def ensure_usage_requirements(self):
        fou.ensure_package("lancedb")

    def initialize(self, samples, brain_key):
        return LanceDBSimilarityIndex(
            samples, self.config, brain_key, backend=self
        )


class LanceDBSimilarityIndex(SimilarityIndex):
    """Class for interacting with LanceDB similarity indexes.

    Args:
        samples: the :class:`fiftyone.core.collections.SampleCollection` used
        config: the :class:`LanceDBSimilarityConfig` used
        brain_key: the brain key
        backend (None): a :class:`LanceDBSimilarity` instance
    """

    def __init__(self, samples, config, brain_key, backend=None):
        super().__init__(samples, config, brain_key, backend=backend)
        self._table = None
        self._db = None
        self._initialize()

    def _initialize(self):
        # Only pass storage options when there are some: no minimum lancedb
        # version is declared, and older releases have no such parameter
        connect_kwargs = {}
        if self.config.storage_options is not None:
            connect_kwargs["storage_options"] = self.config.storage_options

        try:
            db = lancedb.connect(self.config.uri, **connect_kwargs)
        except Exception as e:
            raise ValueError(
                "Failed to connect to LanceDB backend at URI '%s'. Refer to "
                "https://docs.voxel51.com/integrations/lancedb.html for more "
                "information" % self.config.uri
            ) from e

        table_names = db.table_names()

        if self.config.table_name is None:
            root = "fiftyone-" + fou.to_slug(self.samples._root_dataset.name)
            table_name = fbu.get_unique_name(root, table_names)

            self.config.table_name = table_name
            self.save_config()

        if self.config.table_name in table_names:
            table = db.open_table(self.config.table_name)
        else:
            table = None

        # Storage options given to `connect()` reach `create_table()` and
        # `open_table()` through the connection, so those calls need none of
        # their own. Keep the connection private: its own `serialize()`
        # includes the storage options, so a public name here would write the
        # credential into the brain document
        self._db = db
        self._table = table

    @property
    def table(self):
        """The ``lancedb.LanceTable`` instance for this index."""
        return self._table

    @property
    def total_index_size(self):
        if self._table is None:
            return 0

        return len(self._table)

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
        if self._table is None:
            pa_table = pa.Table.from_arrays(
                [[], [], []], names=["id", "sample_id", "vector"]
            )
        else:
            pa_table = self._table.to_arrow()

        if label_ids is not None:
            ids = label_ids
        else:
            ids = sample_ids

        if warn_existing or not allow_existing or not overwrite:
            existing_ids = set(pa_table["id"].to_pylist()) & set(ids)
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

        if label_ids is not None:
            ids = list(label_ids)
        else:
            ids = list(sample_ids)

        dim = embeddings.shape[1]

        if self._table:
            prev_embeddings = np.concatenate(
                pa_table["vector"].to_numpy()
            ).reshape(-1, dim)
            embeddings = np.concatenate([prev_embeddings, embeddings])
            ids = pa_table["id"].to_pylist() + ids
            sample_ids = pa_table["sample_id"].to_pylist() + sample_ids

        embeddings = pa.array(embeddings.reshape(-1), type=pa.float32())
        embeddings = pa.FixedSizeListArray.from_arrays(embeddings, dim)
        sample_ids = list(sample_ids)
        pa_table = pa.Table.from_arrays(
            [ids, sample_ids, embeddings], names=["id", "sample_id", "vector"]
        )
        self._table = self._db.create_table(
            self.config.table_name, pa_table, mode="overwrite"
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
            existing_ids = list(self._index.fetch(ids).vectors.keys())
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

        df = self._table.to_pandas()
        df = df[~df["id"].isin(ids)]
        self._table = self._db.create_table(
            self.config.table_name, df, mode="overwrite"
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

        df = self._table.to_pandas()

        found_embeddings = []
        found_sample_ids = []
        found_label_ids = []
        missing_ids = []

        if sample_ids is not None and self.config.patches_field is not None:
            df.set_index("sample_id", drop=False, inplace=True)

            if not etau.is_container(sample_ids):
                sample_ids = [sample_ids]

            for sample_id in sample_ids:
                if sample_id in df.index:
                    found_embeddings.append(df.loc[sample_id]["vector"])
                    found_sample_ids.append(sample_id)
                    found_label_ids.append(df.loc[sample_id]["id"])
                else:
                    missing_ids.append(sample_id)
        elif self.config.patches_field is not None:
            df.set_index("id", drop=False, inplace=True)

            if label_ids is None:
                label_ids = list(df.index)
            elif not etau.is_container(label_ids):
                label_ids = [label_ids]

            for label_id in label_ids:
                if label_id in df.index:
                    found_embeddings.append(df.loc[label_id]["vector"])
                    found_sample_ids.append(df.loc[label_id]["sample_id"])
                    found_label_ids.append(label_id)
                else:
                    missing_ids.append(label_id)
        else:
            df.set_index("id", drop=False, inplace=True)

            if sample_ids is None:
                sample_ids = list(df.index)
            elif not etau.is_container(sample_ids):
                sample_ids = [sample_ids]

            for sample_id in sample_ids:
                if sample_id in df.index:
                    found_embeddings.append(df.loc[sample_id]["vector"])
                    found_sample_ids.append(sample_id)
                else:
                    missing_ids.append(sample_id)

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

        embeddings = np.array(found_embeddings)
        sample_ids = np.array(found_sample_ids)
        if label_ids is not None:
            label_ids = np.array(found_label_ids)

        return embeddings, sample_ids, label_ids

    def cleanup(self):
        if self._db is None:
            return

        for tbl in (
            self.config.table_name,
            self.config.table_name + "_filter",
        ):
            if tbl in self._db.table_names():
                self._db.drop_table(tbl)

        self._table = None

    def _kneighbors(
        self,
        query=None,
        k=None,
        reverse=False,
        aggregation=None,
        return_dists=False,
    ):
        if query is None:
            raise ValueError("LanceDB does not support full index neighbors")

        if reverse is True:
            raise ValueError(
                "LanceDB does not support least similarity queries"
            )

        if aggregation not in (None, "mean"):
            raise ValueError(
                f"LanceDB does not support {aggregation} aggregation"
            )

        if k is None:
            k = self.index_size

        query = self._parse_neighbors_query(query)
        if aggregation == "mean" and query.ndim == 2:
            query = query.mean(axis=0)

        single_query = query.ndim == 1
        if single_query:
            query = [query]

        table = self._table

        if self.has_view:
            if self.config.patches_field is not None:
                index_ids = list(self.current_label_ids)
            else:
                index_ids = list(self.current_sample_ids)

            df = table.to_pandas()
            df = df[df["id"].isin(index_ids)]
            table = self._db.create_table(
                self.config.table_name + "_filter", df, mode="overwrite"
            )

        metric = _SUPPORTED_METRICS[self.config.metric]

        sample_ids = []
        label_ids = [] if self.config.patches_field is not None else None
        dists = []
        for q in query:
            results = table.search(q).metric(metric).limit(k).to_df()

            if self.config.patches_field is not None:
                sample_ids.append(results.sample_id.tolist())
                label_ids.append(results.id.tolist())
            else:
                sample_ids.append(results.id.tolist())

            if return_dists:
                dists.append(results._distance.tolist())

        if single_query:
            sample_ids = sample_ids[0]
            if label_ids is not None:
                label_ids = label_ids[0]
            if return_dists:
                dists = dists[0]

        if return_dists:
            return sample_ids, label_ids, dists

        return sample_ids, label_ids

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
        df = self._table.to_pandas()
        df = df[df["id"].isin(query_ids)]
        query = np.array([v for v in df["vector"]])

        if query.size == 0:
            raise ValueError(
                "Query IDs %s were not found in the index" % query_ids
            )

        if single_query:
            query = query[0, :]

        return query

    @classmethod
    def _from_dict(cls, d, samples, config, brain_key):
        return cls(samples, config, brain_key)
