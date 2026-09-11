"""
LanceDB similarity backend.

| Copyright 2017-2026, Voxel51, Inc.
| `voxel51.com <https://voxel51.com/>`_
|
"""
import logging
from collections import Counter

import numpy as np

import eta.core.utils as etau

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

# IDs per predicate. Lance parses a predicate as a single expression, so an
# unbounded `IN` list turns a large removal into a multi-megabyte string. At
# 10k the predicate is ~180 KB and an existence scan runs about 3x faster than
# it does at 1k, where the per-call overhead dominates
_ID_BATCH_SIZE = 10000

# 0.34.0 is the first release whose `create_index` accepts a `config=`, which
# is how the scalar index on `id` is built. Everything else this backend calls
# -- `list_tables`, `merge_insert`, `checkout_latest`, `list_indices` -- is
# older than that, so the config parameter sets the floor.
_LANCEDB_REQUIREMENT = "lancedb>=0.34.0"

logger = logging.getLogger(__name__)


def _table_names(db):
    """Returns the names of every table in ``db``.

    The alternative, ``table_names()``, is paged and defaults to 10, so a
    database holding more than that reports an existing table as missing --
    which would strand an index and let
    :meth:`fiftyone.brain.internal.core.utils.get_unique_name` hand out a name
    that is already taken.
    """
    names = []
    page_token = None
    while True:
        response = db.list_tables(page_token=page_token)

        # Materialize the page so the emptiness check below tests the rows
        # rather than the container, which can be truthy while yielding
        # nothing. `getattr` because a bare list is also accepted.
        page = list(getattr(response, "tables", response))
        names.extend(page)

        page_token = getattr(response, "page_token", None)

        # An empty page ends the walk even when a token comes back with it,
        # which would otherwise loop forever
        if not page_token or not page:
            return names


def _to_arrow_table(ids, sample_ids, embeddings):
    """Builds an Arrow table in the index's schema.

    Args:
        ids: an iterable of index IDs
        sample_ids: an iterable of sample IDs
        embeddings: a ``num_embeddings x num_dims`` array of embeddings

    Returns:
        a ``pyarrow.Table``
    """
    dims = embeddings.shape[1]
    vectors = pa.FixedSizeListArray.from_arrays(
        pa.array(embeddings.reshape(-1), type=pa.float32()), dims
    )
    return pa.Table.from_arrays(
        [_to_id_list(ids), _to_id_list(sample_ids), vectors],
        names=["id", "sample_id", "vector"],
    )


def _to_id_list(ids):
    """Normalizes an ID or an iterable of IDs to a list.

    Args:
        ids: an ID or an iterable of IDs

    Returns:
        a list of IDs
    """
    if ids is None:
        return []

    # A bare string is iterable, so listing it would split it into characters
    # and quietly address the wrong rows
    if not etau.is_container(ids):
        return [ids]

    return list(ids)


def _id_predicate(ids):
    """Builds a SQL predicate matching the given IDs.

    Args:
        ids: an iterable of IDs

    Returns:
        a SQL predicate string
    """
    # Doubling is how Lance's SQL parser escapes a quote inside a literal
    quoted = ", ".join("'%s'" % str(_id).replace("'", "''") for _id in ids)
    return "id IN (%s)" % quoted


class LanceDBSimilarityConfig(SimilarityConfig):
    """Configuration for a LanceDB similarity instance.

    Args:
        table_name (None): the name of the LanceDB table to use. If none is
            provided, a new table will be created
        metric ("cosine"): the embedding distance metric to use when creating a
            new index. Supported values are ``("cosine", "euclidean")``
        uri ("/tmp/lancedb"): the database URI to use. May be a local path or
            an object store prefix such as ``gs://bucket/prefix``
        storage_options (None): a dict of storage options to pass to LanceDB,
            used to authenticate against an object store. Refer to
            https://lancedb.github.io/lancedb/guides/storage/ for the keys each
            store accepts
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
        self._storage_options = storage_options

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
        self._storage_options = value

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
        fou.ensure_package(_LANCEDB_REQUIREMENT)

    def ensure_usage_requirements(self):
        fou.ensure_package(_LANCEDB_REQUIREMENT)

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
        try:
            # An object store needs credentials, which a local path does not,
            # so only pass the options through when they were supplied
            kwargs = {}
            if self.config.storage_options:
                kwargs["storage_options"] = self.config.storage_options

            db = lancedb.connect(self.config.uri, **kwargs)
        except Exception as e:
            raise ValueError(
                "Failed to connect to LanceDB backend at URI '%s'. Refer to "
                "https://docs.voxel51.com/integrations/lancedb.html for more "
                "information" % self.config.uri
            ) from e

        table_names = _table_names(db)

        if self.config.table_name is None:
            root = "fiftyone-" + fou.to_slug(self.samples._root_dataset.name)
            table_name = fbu.get_unique_name(root, table_names)

            self.config.table_name = table_name
            self.save_config()

        if self.config.table_name in table_names:
            table = db.open_table(self.config.table_name)
        else:
            table = None

        self._db = db
        self._table = table

    @property
    def table(self):
        """The ``lancedb.LanceTable`` instance for this index."""
        return self._table

    def _sync_table(self):
        """Points this index at the latest committed state of its table.

        Lance pins a handle to the version it was opened at, so another
        process's writes stay invisible for the lifetime of this object. On a
        write path that stale view is not merely out of date: an existence
        check misses rows that are present, so a merge inserts a second row
        under an ID that already exists, and ``allow_existing=False`` fails to
        raise. The refresh costs about 0.1 ms against a 2.8 ms merge.
        """
        if self._table is None:
            # The table may have been created by another writer since this
            # index opened, in which case there is a version to move to
            if self.config.table_name in _table_names(self._db):
                self._table = self._db.open_table(self.config.table_name)

            return

        self._table.checkout_latest()

    def reload(self):
        """Refreshes the index against the latest committed table version."""
        self._sync_table()
        super().reload()

    @property
    def total_index_size(self):
        if self._table is None:
            return 0

        return len(self._table)

    def _ensure_id_index(self):
        """Creates the scalar index on the ``id`` column if it is missing.

        Every merge, delete and existence check matches on ``id``, and Lance
        scans the whole column for those matches unless it is indexed, which
        makes the cost of an incremental write proportional to the size of the
        table. At a million rows and a 100-row batch, a merge measures about
        33.9 ms unindexed against 10.6 ms indexed, and the rewrite it replaces
        measures 4,152 ms.
        """
        try:
            # `create_index` replaces by default, so this guard is what stops
            # every add from rebuilding the index; it is not a
            # micro-optimization
            indexed_columns = [
                table_index.columns
                for table_index in self._table.list_indices()
            ]
            if ["id"] in indexed_columns:
                return

            # BTree rather than Bitmap: IDs are unique, so the column's
            # cardinality is its row count
            self._table.create_index("id", config=lancedb.index.BTree())
        except Exception as e:
            # The rows are committed by this point and the index only makes
            # later writes faster, so failing to build it must not fail the
            # add. Concurrent writers race to create it and Lance rejects the
            # losers with a conflict it labels retryable
            logger.warning("Failed to index the 'id' column: %s", e)

    def _get_existing_ids(self, ids):
        """Returns the subset of ``ids`` that are present in the index.

        Args:
            ids: an iterable of IDs

        Returns:
            a list of the IDs that are present
        """
        if self._table is None:
            return []

        existing_ids = []
        for batch_ids in fou.iter_batches(list(ids), _ID_BATCH_SIZE):
            # A vector search applies a default row limit but a plain scan
            # does not; asking for no limit keeps a future default from
            # turning this into a false report of missing IDs
            results = (
                self._table.search(None)
                .where(_id_predicate(batch_ids))
                .select(["id"])
                .limit(None)
                .to_arrow()
            )
            existing_ids.extend(results["id"].to_pylist())

        # A table can hold an ID twice — written outside this connector, or
        # by a merge that raced — and callers count these to report on them
        return list(dict.fromkeys(existing_ids))

    def _merge_rows(self, pa_table, overwrite):
        """Upserts the given rows into the table.

        Args:
            pa_table: a ``pyarrow.Table`` in the index's schema
            overwrite: whether to replace rows whose IDs already exist
        """
        # Lance merges in place, so this costs the size of the batch rather
        # than the size of the table
        merge = self._table.merge_insert("id")
        merge = merge.when_not_matched_insert_all()
        if overwrite:
            merge = merge.when_matched_update_all()

        merge.execute(pa_table)

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
        if label_ids is not None:
            ids = _to_id_list(label_ids)
        else:
            ids = _to_id_list(sample_ids)

        if not ids:
            # `fbu.get_embeddings()` hands back an empty array when a
            # collection yields no embeddings at all, and an empty Arrow
            # column is typed null, which no table or index can be built from
            if reload:
                self.reload()

            return

        self._sync_table()

        # A duplicate would match one target row twice, which the merge below
        # rejects, and on the insert-only path it would write two rows sharing
        # an ID. Raising with the offending ID beats either failure. The set
        # is the cheap test; the Counter only runs to name the duplicate
        if len(set(ids)) != len(ids):
            duplicate_ids = [
                _id for _id, count in Counter(ids).items() if count > 1
            ]
            raise ValueError(
                "Found %d duplicate IDs (eg %s) in the provided IDs. IDs must "
                "be unique within a single add"
                % (len(duplicate_ids), duplicate_ids[0])
            )

        # The merge honors `overwrite`, so the existing IDs are looked up
        # only for the warning and the error below
        if warn_existing or not allow_existing:
            existing_ids = self._get_existing_ids(ids)
            num_existing = len(existing_ids)

            if num_existing > 0:
                if not allow_existing:
                    raise ValueError(
                        "Found %d IDs (eg %s) that already exist in the index"
                        % (num_existing, existing_ids[0])
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

        pa_table = _to_arrow_table(ids, sample_ids, embeddings)

        if self._table is None:
            try:
                self._table = self._db.create_table(
                    self.config.table_name, pa_table
                )
            except Exception:
                if self.config.table_name not in _table_names(self._db):
                    raise

                # Another writer created the table between this index opening
                # and this add; join it rather than replacing it
                self._table = self._db.open_table(self.config.table_name)
                self._merge_rows(pa_table, overwrite)
        else:
            self._merge_rows(pa_table, overwrite)

        # Runs after the write so an existing table without the index picks
        # it up on its next add, with the new rows included
        self._ensure_id_index()

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
            ids = _to_id_list(label_ids)
        else:
            ids = _to_id_list(sample_ids)

        self._sync_table()

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

        if self._table is not None:
            # Lance records a deletion as a per-fragment sidecar, leaving the
            # data files untouched
            for batch_ids in fou.iter_batches(ids, _ID_BATCH_SIZE):
                self._table.delete(_id_predicate(batch_ids))

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
            if tbl in _table_names(self._db):
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
            results = table.search(q).metric(metric).limit(k).to_pandas()

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
