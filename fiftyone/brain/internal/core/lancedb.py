"""
LanceDB similarity backend.

| Copyright 2017-2026, Voxel51, Inc.
| `voxel51.com <https://voxel51.com/>`_
|
"""

import logging
from collections import Counter
import re
from copy import deepcopy

import numpy as np

import eta.core.utils as etau

import fiftyone.brain as fb
import fiftyone.core.utils as fou
import fiftyone.brain.internal.core.utils as fbu
from fiftyone.brain.similarity import (
    SimilarityConfig,
    Similarity,
    SimilarityIndex,
)

#: The store a run opens when neither it nor the backend names one.
DEFAULT_URI = "/tmp/lancedb"

lancedb = fou.lazy_import("lancedb")
pa = fou.lazy_import("pyarrow")
pd = fou.lazy_import("pandas")


_SUPPORTED_METRICS = {
    "cosine": "cosine",
    "euclidean": "l2",
}

# Vector index families, mapped to their `lancedb.index` class name. The
# default suits 512 dimensions; at 2048, `ivf_pq` with `num_sub_vectors=256`
# matches its recall at a ninth of the index size and builds 3.5x faster, so
# the family is configuration rather than a constant
_SUPPORTED_INDEX_TYPES = {
    "ivf_flat": "IvfFlat",
    "ivf_sq": "IvfSq",
    "ivf_pq": "IvfPq",
    "ivf_rq": "IvfRq",
    "ivf_hnsw_flat": "IvfHnswFlat",
    "ivf_hnsw_sq": "IvfHnswSq",
    "ivf_hnsw_pq": "IvfHnswPq",
}

# The column every vector index here is built on, and the one name used for
# it. `replace=True` is scoped to the index *name*, not the column: two builds
# under different names leave both indexes in place and the query keeps using
# whichever was created first, with no error and no way to select the other at
# query time. Reusing one name is what makes a rebuild a replacement
_VECTOR_COLUMN = "vector"
_VECTOR_INDEX_NAME = "vector_idx"

# Widths at or below this take IVF_RQ. Its 1-bit codes are best in class at
# 768 -- 0.9993 mean / 0.900 worst against IVF_PQ m=96's 0.9977 / 0.800, at a
# third of the latency and a 2 s build against 32 s -- and collapse at 2048
# (0.088-0.173 mean, 0.000 worst). The boundary sits at the widest measured
# good rather than anywhere interpolated: 1024 and 1536 are untested and take
# PQ deliberately.
_RQ_MAX_DIMS = 768

# Sub-vectors per PQ code, as a divisor of the width. LanceDB's own default
# is a sixteenth, which at 768 measures 0.9447 mean / 0.200 worst against an
# eighth's 0.9963 / 0.800 -- so the family default is the one setting here
# that has to be supplied rather than left alone.
_PQ_DIMS_PER_SUB_VECTOR = 8

# Partitions probed per query where the family does not escalate. LanceDB's
# implicit value is 20.
_DEFAULT_NPROBES = 25

# Rows below which an index is not worth building: an exhaustive scan of a
# small table beats an indexed query, because the index read is a fixed cost
# the scan does not pay. The crossover was measured four ways at 768
# dimensions -- ~749 rows from a laptop and ~2,049 in-region when the index
# is cache-warm, and ~29,700 and ~239,900 respectively on a cold first
# query. The default takes the warm in-region figure, which is the steady
# state a deployment runs in; the cold figures describe how much a first
# query costs, which pre-warming answers rather than skipping the index.
# Configurable because the warm crossover moves with client position, which
# the deployment knows and this library does not.
_DEFAULT_MIN_INDEX_ROWS = 2048

# Families that escalate instead of pinning. They build about twice the
# partitions of the others at the same row count, so one pinned value probes
# half the share and reads as recall decaying with scale: at 400k by 768,
# nprobes=25 reaches 26% of an RQ index against 52% of a PQ one, and RQ goes
# 0.9943/0.700 pinned against 0.9993/0.900 escalating.
_ESCALATING_FAMILIES = frozenset({"ivf_rq", "ivf_sq"})


def _default_index_type(dims):
    """The index family for embeddings of the given width.

    Args:
        dims: the embedding dimension

    Returns:
        a key of ``_SUPPORTED_INDEX_TYPES``
    """
    return "ivf_rq" if dims <= _RQ_MAX_DIMS else "ivf_pq"


def _default_index_params(index_type, dims):
    """The family's parameters when the config supplies none.

    Args:
        index_type: a key of ``_SUPPORTED_INDEX_TYPES``
        dims: the embedding dimension

    Returns:
        a dict of keyword arguments for the family
    """
    if index_type in ("ivf_pq", "ivf_hnsw_pq"):
        return {"num_sub_vectors": dims // _PQ_DIMS_PER_SUB_VECTOR}

    return {}


# IDs per predicate. Lance parses a predicate as a single expression, so an
# unbounded `IN` list turns a large removal into a multi-megabyte string. At
# 10k the predicate is ~180 KB and an existence scan runs about 3x faster than
# it does at 1k, where the per-call overhead dominates
_ID_BATCH_SIZE = 10000

# No floor, which is what the backend declared before any of this: every
# call the write path makes -- `merge_insert`, `checkout_latest`, the
# predicate delete, the unlimited scan, `list_indices`, `index_stats` --
# was verified working back to 0.30.2. Naming a version here would refuse
# to run for people the connector serves perfectly well.
_LANCEDB_REQUIREMENT = "lancedb"

# `create_index` first accepts a `config=` in 0.34.0, and that is the only
# call in this backend that needs it -- both indexes are built through it.
# Older releases skip them rather than lose the backend: writes scan the id
# column and queries scan every vector, slower and still correct.
_ID_INDEX_REQUIREMENT = "lancedb>=0.34.0"

# Page size when paginating LanceDB table listings
_DB_TABLE_PG_LIMIT = 100

logger = logging.getLogger(__name__)


def _empty_rows():
    """An empty frame in the index's schema."""
    return pd.DataFrame({"id": [], "sample_id": [], "vector": []}).astype(
        {"id": str, "sample_id": str}
    )


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


def _id_predicate(ids, column="id"):
    """Builds a SQL predicate matching the given IDs.

    Args:
        ids: an iterable of IDs
        column ("id"): the column to match against

    Returns:
        a SQL predicate string
    """
    # Doubling is how Lance's SQL parser escapes a quote inside a literal
    quoted = ", ".join("'%s'" % str(_id).replace("'", "''") for _id in ids)
    return "%s IN (%s)" % (column, quoted)


class LanceDBSimilarityConfig(SimilarityConfig):
    """Configuration for a LanceDB similarity instance.

    Args:
        table_name (None): the name of the LanceDB table to use. If none is
            provided, a new table will be created
        metric ("cosine"): the embedding distance metric to use when creating a
            new index. Supported values are ``("cosine", "euclidean")``
        uri (None): the database URI to use. Recorded on the run, so two
            runs may keep their tables in different stores. A run that names
            none opens whatever the backend is configured with, which lets a
            deployment move every unnamed run's store at once
        storage_options (None): a dict of storage options for the object store
            backing ``uri``, eg credentials for a cloud bucket. Passed through
            to ``lancedb.connect()``. Unlike ``uri`` this is not serialized,
            since it carries credentials, so it must be supplied again each
            time the index is loaded. A value given here replaces any
            configured for the backend rather than merging with it
        index_type (None): the vector index family to build. Supported
            values are ``("ivf_flat", "ivf_sq", "ivf_pq", "ivf_rq",
            "ivf_hnsw_flat", "ivf_hnsw_sq", "ivf_hnsw_pq")``. Chosen from the
            embedding width when unset: ``"ivf_rq"`` at 768 dimensions and
            below, ``"ivf_pq"`` above
        min_index_rows (2048): the row count below which no vector index is
            built, because a scan of a small table beats an indexed query.
            Measured at 768 dimensions: the cache-warm crossover is ~749
            rows from a laptop and ~2,049 in-region. Set it from where the
            queries run, which this library cannot know
        index_params (None): a dict of keyword arguments for the index
            family, such as ``num_sub_vectors`` for ``"ivf_pq"``. The PQ
            families take an eighth of the width when unset, rather than
            LanceDB's sixteenth. The distance type is set from ``metric`` and
            cannot be overridden here
        nprobes (None): the number of partitions to probe per query. Set from
            the family when unset -- the escalating families search outward
            until they have enough, and the rest take a fixed 25 -- because
            the partition count grows with the table, so one pinned value
            probes an ever-smaller share as rows arrive and recall falls for
            that reason rather than with scale. **Inert on the
            ``ivf_hnsw_*`` families**, which build a single IVF partition,
            leaving nothing to choose between: the query plan still reports
            the value and the engine ignores it. Use ``ef`` there
        ef (None): the HNSW search-list size, and the only pruning knob the
            ``ivf_hnsw_*`` families have. LanceDB suggests starting at
            ``1.5 * k`` and raising toward ``10 * k`` if recall needs it
            (https://docs.lancedb.com/performance). Must be at least
            ``k * refine_factor`` when both are set
        refine_factor (10): how many extra candidates to retrieve and
            re-rank by exact distance -- what LanceDB's guide calls pulling
            extra candidates and re-scoring them on full vectors, which the
            quantized families need. An indexed query is approximate, and for
            ``"cosine"`` it also reports distance on the scale the index was
            built on rather than the scale an unindexed query reports. The
            re-rank restores both: measured at 50k rows and 512 dimensions it
            costs 1.1 ms against a 2.2 ms indexed query, and is still 6x
            faster than the 19.9 ms an unindexed scan costs. Set to ``None``
            to skip it
        **kwargs: keyword arguments for :class:`SimilarityConfig`
    """

    def __init__(
        self,
        table_name=None,
        metric="cosine",
        uri=None,
        storage_options=None,
        index_type=None,
        min_index_rows=None,
        index_params=None,
        nprobes=None,
        ef=None,
        refine_factor=10,
        **kwargs,
    ):
        if metric not in _SUPPORTED_METRICS:
            raise ValueError(
                "Unsupported metric '%s'. Supported values are %s"
                % (metric, tuple(_SUPPORTED_METRICS.keys()))
            )

        if index_type is not None and index_type not in _SUPPORTED_INDEX_TYPES:
            raise ValueError(
                "Unsupported index type '%s'. Supported values are %s"
                % (index_type, tuple(_SUPPORTED_INDEX_TYPES.keys()))
            )

        if index_params is not None and not isinstance(index_params, dict):
            raise ValueError(
                "index_params must be a dict, found %s" % type(index_params)
            )

        # Caught here rather than at build time, where it arrives as a
        # "multiple values for keyword argument" TypeError inside the build,
        # is warned about, and leaves the table unindexed for good
        if index_params and "distance_type" in index_params:
            raise ValueError(
                "The index distance type is set from `metric`; remove "
                "'distance_type' from `index_params`"
            )

        # Non-positive knobs reach LanceDB as an OverflowError or a bare
        # "cannot be zero" from its Rust layer, at query time, in whichever
        # session loads the brain run rather than the one that set them
        for name, value in (
            ("nprobes", nprobes),
            ("ef", ef),
            ("refine_factor", refine_factor),
        ):
            if value is not None and (not isinstance(value, int) or value < 1):
                raise ValueError(
                    "%s must be a positive integer, found %r" % (name, value)
                )

        super().__init__(**kwargs)

        self.table_name = table_name
        self.metric = metric
        # Serialized, so a run keeps the store it was built in rather than
        # whatever the process reading it happens to be configured with
        self.uri = uri
        self.index_type = index_type
        self.min_index_rows = (
            _DEFAULT_MIN_INDEX_ROWS
            if min_index_rows is None
            else min_index_rows
        )
        self.index_params = index_params or {}
        self.nprobes = nprobes
        self.ef = ef
        self.refine_factor = refine_factor

        # Assigned through its setter, which copies. Private: unlike the URI
        # this carries credentials, and a run document is readable by anyone
        # who can read the dataset
        self.storage_options = storage_options

    @property
    def method(self):
        return "lancedb"

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
        # `_load_parameters` lets a configured value overwrite one already
        # assigned, so the backend is consulted only where this config
        # carries none: a bare refresh must not swap the credential the
        # caller is holding for whatever the deployment is configured with
        if storage_options is not None or self.storage_options is None:
            self._load_parameters(storage_options=storage_options)

        # Not through `_load_parameters`, which lets a configured value
        # overwrite what is already set: the run's own URI has to win over
        # the backend's, or a deployment default would move a run's table
        # out from under it
        if uri is not None:
            self.uri = uri

    def resolve_uri(self):
        """The store this run opens.

        Its own URI where it recorded one, else whatever the backend is
        configured with, else a local directory.

        The fallback is resolved here rather than assigned to :attr:`uri`,
        so a run that named no store does not silently acquire the one that
        happened to be configured the first time something read it.

        Returns:
            the database URI
        """
        if self.uri:
            return self.uri

        configured = fb.brain_config.similarity_backends.get(
            self.method, {}
        ).get("uri")

        return configured or DEFAULT_URI


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

    # Set on the instance once the release is known to reject `config=`, so
    # a long ingest does not warn about it on every add. A class attribute
    # rather than an `__init__` assignment, because an index can be built
    # without one
    _id_index_unavailable = False
    _vector_index_unavailable = False

    def __init__(self, samples, config, brain_key, backend=None):
        super().__init__(samples, config, brain_key, backend=backend)
        self._table = None
        self._db = None
        self._initialize()

    def _initialize(self):
        # Only pass storage options when there are some: no minimum lancedb
        # version is declared, and older releases have no such parameter. An
        # empty dict carries no credential, so it counts as none
        connect_kwargs = {}
        if self.config.storage_options:
            connect_kwargs["storage_options"] = self.config.storage_options

        uri = self.config.resolve_uri()
        try:
            db = lancedb.connect(uri, **connect_kwargs)
        except Exception as e:
            raise ValueError(
                "Failed to connect to LanceDB backend at URI '%s'. Refer to "
                "https://docs.voxel51.com/integrations/lancedb.html for more "
                "information" % uri
            ) from e

        if self.config.table_name is None:
            root = "fiftyone-" + fou.to_slug(self.samples._root_dataset.name)
            table_name = fbu.get_unique_name(root, _table_names(db))

            self.config.table_name = table_name
            self.save_config()

            # A name minted against the listing above names no table yet;
            # `add_to_index` is what creates it
            table = None
        else:
            table = _open_table(db, self.config.table_name)

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
            # Asked for rather than looked up in the listing: another writer
            # may have created it since this index opened, and absence is the
            # ordinary case rather than an error
            self._table = _open_table(self._db, self.config.table_name)

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

        Every merge, delete and existence check matches on ``id``, and
        LanceDB's performance guide is explicit that this index is a
        prerequisite rather than a tuning choice: a merge "has to scan
        existing data to find matches on the join key (or look them up via
        a scalar index, if one exists)", and without one "the matching step
        falls back to a full column scan, which becomes the dominant cost
        at scale". Measured at 512 dimensions and a 100-row batch, a merge
        into a 500k-row table costs 3.8 ms indexed against 13.3 ms
        unindexed, and the unindexed figure is the one that grows.

        https://docs.lancedb.com/performance
        """
        if self._id_index_unavailable:
            return

        try:
            # `create_index` replaces by default, so this guard is what stops
            # every add from rebuilding the index; it is not a
            # micro-optimization
            if self._is_indexed("id"):
                return

            # BTree rather than Bitmap: IDs are unique, so the column's
            # cardinality is its row count
            self._table.create_index("id", config=lancedb.index.BTree())
        except TypeError as e:
            # `config=` is what 0.34.0 added, and a release without it
            # rejects the keyword rather than failing the build. Say what
            # the cost is, since nothing else here is degraded -- once,
            # because no later add on this release will fare differently
            self._id_index_unavailable = True
            logger.warning(
                "Skipping the 'id' index (%s); every write will scan the id "
                "column instead. %s builds it",
                e,
                _ID_INDEX_REQUIREMENT,
            )
        except Exception as e:
            # The rows are committed by this point and the index only makes
            # later writes faster, so failing to build it must not fail the
            # add. Concurrent writers race to create it and Lance rejects the
            # losers with a conflict it labels retryable
            logger.warning("Failed to index the 'id' column: %s", e)

    def _rows(self):
        """The table's rows, or an empty frame when there is no table yet.

        An index whose adds all yielded no embeddings never creates a table:
        `fbu.get_embeddings()` hands back an empty array for a collection
        that yields none, and a run saved from it is a brain key pointing at
        nothing. Reading that as an empty index rather than dereferencing
        `None` keeps the missing-ID reporting below working, since every ID
        asked for is then correctly missing.

        Returns:
            a ``pandas.DataFrame`` in the index's schema
        """
        if self._table is None:
            return _empty_rows()

        return self._table.to_pandas()

    def _rows_for(self, ids, column="id"):
        """The rows carrying the given IDs, read by predicate.

        A whole-table ``to_pandas()`` reads every vector to answer a
        question about a handful of rows, and the gap grows with the table
        because one side reads all of it. Measured at 512 dimensions,
        fetching one row: 37 ms against 2.0 ms at 20k rows, 180 against 1.7
        at 100k, 584 against 1.4 at 300k.

        Args:
            ids: an iterable of IDs
            column ("id"): the column the IDs name

        Returns:
            a ``pandas.DataFrame`` in the index's schema
        """
        ids = list(ids)
        if self._table is None or not ids:
            return _empty_rows()

        frames = [
            self._table.search(None)
            .where(_id_predicate(batch_ids, column=column))
            .limit(None)
            .to_pandas()
            for batch_ids in fou.iter_batches(ids, _ID_BATCH_SIZE)
        ]

        return pd.concat(frames, ignore_index=True)

    def _dims(self):
        """The width of the table's vector column.

        Returns:
            the embedding dimension
        """
        return self._table.schema.field(_VECTOR_COLUMN).type.list_size

    def _is_indexed(self, column):
        """Whether the table carries an index on the given column.

        Args:
            column: a column name

        Returns:
            True/False
        """
        return [column] in [
            table_index.columns for table_index in self._table.list_indices()
        ]

    def _ensure_vector_index(self):
        """Creates the index on the ``vector`` column if it is missing.

        An unindexed query scans every vector: 3.7 ms at 1k rows, 47.3 ms
        at 100k and 417.6 ms at 1M, measured at 512 dimensions on local
        disk, which spends the whole 500 ms query budget before any network
        hop. LanceDB's guide puts the threshold at roughly 100k vectors;
        ``min_index_rows`` is lower because the crossover was measured
        directly at this width rather than taken as a general figure.

        https://docs.lancedb.com/performance

        A query over a view is answered from a per-query copy of the table
        that carries no index, so it scans regardless. Replacing that
        rewrite with a native prefilter is separate work.

        Rebuilt when the distance type stops matching ``metric``, and
        otherwise left alone. LanceDB answers a query whose metric
        disagrees with the index by scanning every vector instead, and says
        so only in a log line from its Rust layer -- no exception, no
        Python warning -- so a stale distance type looks exactly like an
        index that is merely slow. The family is not compared: a rebuild
        costs minutes at ten million rows, so changing ``index_type`` under
        an existing index is a no-op rather than a surprise that fires on
        the next add.
        """
        if self._vector_index_unavailable:
            return

        # Not a failure, so not warned: a table under the crossover is
        # answered faster by a scan, and the next add that carries it over
        # builds the index then
        if len(self._table) < self.config.min_index_rows:
            return

        distance_type = _SUPPORTED_METRICS[self.config.metric]

        try:
            if self._is_indexed(_VECTOR_COLUMN):
                stats = self._table.index_stats(_VECTOR_INDEX_NAME)
                if stats is not None and stats.distance_type == distance_type:
                    return

            index_type = self.config.index_type or _default_index_type(
                self._dims()
            )
            index_params = self.config.index_params or _default_index_params(
                index_type, self._dims()
            )
            index_class_name = _SUPPORTED_INDEX_TYPES[index_type]

            # Built outside the compatibility handler below: a bad key in
            # `index_params` reaches the family constructor as a TypeError
            # too, and treating that as an old release would report the
            # wrong cause and latch the index off for good -- correcting
            # the config would then never be retried
            index_config = getattr(lancedb.index, index_class_name)(
                distance_type=distance_type, **index_params
            )
        except Exception as e:
            logger.warning(
                "Failed to configure the %r index: %s", _VECTOR_COLUMN, e
            )
            return

        try:
            # Pinned rather than left implicit: `replace=True` is scoped to
            # the index name, so if LanceDB's default ever moved off
            # `vector_idx` a rebuild would add a second index rather than
            # replace this one, and the query would silently keep using
            # whichever was built first
            self._table.create_index(
                _VECTOR_COLUMN, config=index_config, name=_VECTOR_INDEX_NAME
            )
        except TypeError as e:
            # Same keyword, same story as the id index: a release without
            # `config=` rejects it rather than failing the build. Warned
            # once, because no later add on this release will differ
            self._vector_index_unavailable = True
            logger.warning(
                "Skipping the %r index (%s); every query will scan every "
                "vector instead. %s builds it",
                _VECTOR_COLUMN,
                e,
                _ID_INDEX_REQUIREMENT,
            )
        except Exception as e:
            # The rows are committed by this point and a query without the
            # index is slow rather than wrong, so a failure here must not fail
            # the add. Concurrent writers race to create it and Lance rejects
            # the losers
            logger.warning(
                "Failed to index the %r column: %s", _VECTOR_COLUMN, e
            )

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

    def _merge_rows(self, pa_table, *, overwrite):
        """Upserts the given rows into the table.

        Lance merges in place, so this costs the size of the batch, not the
        size of the table: 33.9 ms against 4,152 ms for a whole-table rewrite
        at 1M rows, 512 dimensions and a 100-row batch.

        Args:
            pa_table: a ``pyarrow.Table`` in the index's schema
            overwrite: whether to replace rows whose IDs already exist
        """
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
                self._merge_rows(pa_table, overwrite=overwrite)
        else:
            self._merge_rows(pa_table, overwrite=overwrite)

        # Both run after the write so an existing table without an index
        # picks one up on its next add, with the new rows included
        self._ensure_id_index()
        self._ensure_vector_index()

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

        # Which column the caller is naming, so the rows can be read by
        # predicate. Only a request for the whole index reads the whole
        # table; naming any IDs reads just those rows
        if sample_ids is not None and self.config.patches_field is not None:
            lookup_column, lookup_ids = "sample_id", sample_ids
        elif self.config.patches_field is not None:
            lookup_column, lookup_ids = "id", label_ids
        else:
            lookup_column, lookup_ids = "id", sample_ids

        if lookup_ids is None:
            df = self._rows()
        else:
            df = self._rows_for(_to_id_list(lookup_ids), column=lookup_column)

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

        # Two-dimensional even when empty, as `sklearn.py` is careful to
        # be: a reducer handed a (0,) array reports "Expected 2D array",
        # nowhere near whatever produced the empty result
        embeddings = (
            np.array(found_embeddings)
            if found_embeddings
            else np.empty((0, 0))
        )
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
            if isinstance(tbl, str) and tbl in _table_names(self._db):
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

        if table is None:
            # No table means no rows to be near. The shape has to match what
            # a populated query returns, because callers unpack it -- and
            # each slot gets its own list, because the populated path
            # returns three and `_set_list_values_by_id` takes all three
            def empty():
                return [] if single_query else [[] for _ in query]

            label_ids = (
                empty() if self.config.patches_field is not None else None
            )
            if return_dists:
                return empty(), label_ids, empty()

            return empty(), label_ids

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
            results = self._search_exactly_k(table, q, metric, k)

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

    def _search(self, table, query, metric, k, *, bypass_index=False):
        """Builds the vector query, applying whichever knobs are configured.

        Each knob is applied only when set, because LanceDB tunes ``nprobes``
        from the partition count and the partition count grows with the table.
        Pinning it probes an ever-smaller share as rows are added, which reads
        as recall decaying with scale.

        ``nprobes`` reaches the plan on every family but only changes results
        on those that build more than one IVF partition, which the
        ``ivf_hnsw_*`` families do not.

        Args:
            table: the ``lancedb.LanceTable`` to query
            query: a query vector
            metric: the LanceDB distance type to query with
            k: the number of neighbors to return
            bypass_index (False): whether to scan every vector rather than
                use the index

        Returns:
            a ``lancedb`` query builder
        """
        search = table.search(query).metric(metric).limit(k)

        if bypass_index:
            return search.bypass_vector_index()

        index_type = self.config.index_type or _default_index_type(len(query))

        if self.config.nprobes is not None:
            search = search.nprobes(self.config.nprobes)
        elif index_type in _ESCALATING_FAMILIES:
            # `maximum_nprobes` first: a minimum above the standing maximum
            # is rejected. Zero means unbounded, so the floor is where the
            # search starts rather than where it stops
            search = search.maximum_nprobes(0).minimum_nprobes(k)
        else:
            search = search.nprobes(_DEFAULT_NPROBES)

        if self.config.ef is not None:
            search = search.ef(self.config.ef)

        if self.config.refine_factor is not None:
            search = search.refine_factor(self.config.refine_factor)

        return search

    def _search_exactly_k(self, table, query, metric, k):
        """Runs the query, falling back to a scan if it comes back short.

        An indexed query only sees the rows in the partitions it probes, so a
        ``k`` approaching the table size can return fewer rows than asked --
        ``ivf_flat`` saturates near half the table at 150k rows -- and reports
        no error. Callers read that as the index holding fewer rows than it
        does, and :meth:`SimilarityIndex.sort_by_similarity` documents
        ``k=None`` as sorting every sample, so the shortfall drops samples
        from a view without saying so.

        The scan costs a second query, and only in the case that would
        otherwise lose rows: a bounded ``k`` returns in full and never
        reaches it.

        Args:
            table: the ``lancedb.LanceTable`` to query
            query: a query vector
            metric: the LanceDB distance type to query with
            k: the number of neighbors to return

        Returns:
            a ``pandas.DataFrame`` of results
        """
        results = self._search(table, query, metric, k).to_pandas()

        # A table with fewer than k rows is short for the honest reason
        if len(results) >= k or len(table) < k:
            return results

        return self._search(
            table, query, metric, k, bypass_index=True
        ).to_pandas()

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

        # Query by ID(s), read by predicate rather than by reading the
        # whole table back to keep a handful of rows
        df = self._rows_for(query_ids)
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


def _open_table(db, table_name):
    """Opens a table, or returns ``None`` when the name holds none.

    Asked for rather than looked up, because an existence check means paging
    the whole listing and a run whose table has yet to be written is the
    ordinary case.

    Args:
        db: a ``lancedb`` connection
        table_name: the name to open

    Returns:
        a ``lancedb.LanceTable``, or None if no table has that name
    """
    try:
        return db.open_table(table_name)
    except ValueError:
        # A table that is absent and one that will not open raise the same
        # type, and only the message separates them. The listing decides it
        # instead: a name that is there names a table that exists, whatever
        # is wrong with it, and calling that absent would let the next add
        # replace it -- `create_table` succeeds over a directory whose
        # manifests are gone and drops the rows still sitting in it. Going
        # through the listing also keeps this off LanceDB's wording, which
        # differs between the embedded and remote paths
        if table_name in _table_names(db):
            raise

        return None


def _legacy_table_names(db):
    """Every table name, from the listing releases before 0.27.1 offer.

    `list_tables` does not exist there; `table_names` does, on every
    release back to at least 0.20.0, and pages on the last name it
    returned rather than on a token of its own. Measured complete and
    without repeats that way from 0.20.0 through 0.39.0 -- which the
    token-based listing is not, hence the compensation in
    :func:`_table_names`.

    Args:
        db: a ``lancedb`` connection

    Returns:
        a list of table names
    """
    names = []
    page_token = None
    while True:
        page = list(
            db.table_names(page_token=page_token, limit=_DB_TABLE_PG_LIMIT)
        )
        if not page:
            return names

        names.extend(page)
        page_token = page[-1]


def _table_names(db):
    # `list_tables` caps a page at ten by default, so the whole listing has
    # to be paged for. The cursor is the token the response carries, which
    # is a storage key rather than a table name, and it is exclusive: a page
    # begins after the last name of the page before it. The response carries
    # no token once it has returned the last page
    if not hasattr(db, "list_tables"):
        return _legacy_table_names(db)

    page_token = None
    table_names = []
    seen = set()
    while True:
        response = db.list_tables(
            page_token=page_token, limit=_DB_TABLE_PG_LIMIT
        )

        # Materialized so the guard below tests the rows rather than the
        # container, which can be truthy while yielding nothing
        page = list(response.tables)
        table_names.extend(page)
        seen.update(page)

        page_token = response.page_token

        # An empty page ends the walk even when a token comes back with it,
        # which would otherwise spin
        if not page_token or not page:
            return table_names

        # Before 0.38.0 the token is the name of the next table rather than
        # the storage key of the last one returned, and the request it is
        # passed to resumes *after* it -- so that one table is never listed,
        # once per page boundary. The name is the token itself, so take it.
        # A key carries the "/" that LanceDB forbids in a table name, which
        # is what tells the two forms apart. Past 0.38.0 this never fires.
        if "/" not in page_token and page_token not in seen:
            table_names.append(page_token)
            seen.add(page_token)
