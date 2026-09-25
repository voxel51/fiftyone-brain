"""
Unit tests for the LanceDB similarity backend.

LanceDB is embedded, so these exercise a real table under ``tmp_path`` and
need no service. They are skipped where ``lancedb`` is not installed; CI
installs it. The tests that build an index over a dataset live in
``tests/intensive/test_similarity.py``, and the table listing is covered
against a real database in ``tests/test_lancedb_store.py``.

| Copyright 2017-2026, Voxel51, Inc.
| `voxel51.com <https://voxel51.com/>`_
|
"""
import contextlib
import inspect
import logging
import os
import shutil
import types
from unittest import mock

import numpy as np
import pytest

lancedb = pytest.importorskip("lancedb")
pa = pytest.importorskip("pyarrow")

# Every import below is E402 by construction: it has to follow the
# `importorskip` calls above, which decide whether this module runs at all
import fiftyone.brain as fob  # noqa: E402
from fiftyone.brain.similarity import SimilarityIndex  # noqa: E402
from fiftyone.brain.internal.core import (
    lancedb as lancedb_backend,
)  # noqa: E402
from fiftyone.brain.internal.core.lancedb import (  # noqa: E402
    LanceDBSimilarity,
    LanceDBSimilarityConfig,
    LanceDBSimilarityIndex,
    _DB_TABLE_PG_LIMIT,
    _ID_BATCH_SIZE,
    _DEFAULT_NPROBES,
    _ID_INDEX_REQUIREMENT,
    _RQ_MAX_DIMS,
    _SLICE_NAME_COLUMN,
    _SUPPORTED_INDEX_TYPES,
    _SUPPORTED_METRICS,
    _VECTOR_INDEX_NAME,
    _default_index_params,
    _id_predicate,
    _open_table,
    _table_names,
    _to_arrow_table,
    _to_id_list,
)

#: `create_index(config=)` arrives in lancedb 0.34.0, and both indexes are
#: built through it. The backend runs without them -- writes scan the id
#: column and queries scan every vector -- so the tests that assert an index
#: exists are the ones that cannot run below it
builds_indexes = pytest.mark.skipif(
    "config"
    not in inspect.signature(lancedb.table.Table.create_index).parameters,
    reason="create_index(config=) arrives in lancedb 0.34.0",
)

#: A family's parameters live in Lance's own index metadata. `index_stats`
#: carries only the row counts, the distance type and the family name, so
#: reading anything finer needs pylance -- a separate package from lancedb,
#: and not one the backend itself requires. It is left out of CI rather
#: than pinned there, since its releases track Lance's on-disk format
#: rather than lancedb's, so the parameters are pinned at the point they
#: are chosen and this reads them back only where it can
try:
    import lance  # noqa: F401

    _READS_INDEX_METADATA = True
except ImportError:
    _READS_INDEX_METADATA = False

reads_index_metadata = pytest.mark.skipif(
    not _READS_INDEX_METADATA,
    reason="reading Lance index metadata needs pylance",
)

DIMS = 8

# Enough IDs to span more than one predicate batch
BATCHED_ROWS = 2 * _ID_BATCH_SIZE + 1

# Rows past which a search stops returning them in distance order: it hands
# back batches sorted within themselves, and `k` near the table size -- what
# `k=None` asks for -- then spans several. Measured reliable at this count on
# lancedb 0.38.0 and 0.39.0, and flaky at 24576 on 0.39.0, so the margin is
# deliberate. `TestKneighborsOverAView` guards the premise rather than
# assuming it holds
UNSORTED_ROWS = 32768

# Rows a product quantizer needs before it can train, so the `ivf_pq` and
# `ivf_hnsw_pq` families cannot be exercised on the 3-row fixtures
PQ_TRAINING_ROWS = 256

# Raised by a mocked pager on the page after the walk should have stopped, so
# a non-terminating loop fails the test rather than spinning forever. A hang
# reads in CI as a job timeout with no output
_DID_NOT_STOP = AssertionError("_table_names did not stop paging")


def _random_embeddings(num_rows, seed=0):
    rng = np.random.default_rng(seed)
    return rng.random((num_rows, DIMS), dtype=np.float32)


def _constant_embeddings(num_rows, fill):
    return np.full((num_rows, DIMS), fill, dtype=np.float32)


def _row_ids(num_rows):
    """``["id-00000", ...]``, enough unique IDs for a table of that size."""
    return np.array(["id-%05d" % i for i in range(num_rows)])


def _seeded_index(tmp_path, **config_kwargs):
    """An unbound index already holding the three basis rows."""
    index = _unbound_index(tmp_path, **config_kwargs)
    index.add_to_index(
        _basis_embeddings(3), np.array(["a", "b", "c"]), reload=False
    )
    return index


def _detached_index(config):
    """An index over the given config, built without a sample collection.

    ``SimilarityIndex.__init__`` binds the index to a collection, which
    needs a database. The paths under test reach only the table, the
    connection and the config, so the instance is built without that
    binding — with the attributes that would name a collection set to
    ``None``, which is the state an unbound index is in.
    """
    index = LanceDBSimilarityIndex.__new__(LanceDBSimilarityIndex)
    index._samples = None
    index._curr_view = None
    index._config = config
    index._initialize()
    return index


def _unbound_index(tmp_path, **config_kwargs):
    """An index over a temp URI, built without a sample collection.

    The same shortcut as the ``index`` fixture, for the tests that need a
    config the fixture does not build.
    """
    config_kwargs.setdefault("metric", "euclidean")
    # These fixtures hold a handful of rows, well under the crossover the
    # default protects. The threshold has its own tests
    config_kwargs.setdefault("min_index_rows", 0)

    return _detached_index(
        LanceDBSimilarityConfig(
            table_name="test", uri=str(tmp_path), **config_kwargs
        )
    )


def _basis_embeddings(num_rows):
    """Unit basis vectors, so nearest-neighbor order is unambiguous."""
    return np.eye(num_rows, DIMS, dtype=np.float32)


@contextlib.contextmanager
def _viewing(**properties):
    """Patches the base class's view properties for the block.

    The fixtures build an index without the sample collection those
    properties are computed from, so a test that wants a view says outright
    what it holds.
    """
    with contextlib.ExitStack() as stack:
        for name, value in properties.items():
            stack.enter_context(
                mock.patch.object(
                    LanceDBSimilarityIndex,
                    name,
                    new_callable=mock.PropertyMock,
                    return_value=value,
                )
            )

        yield


def _no_view():
    """Patches out the view, so a query runs against the whole index."""
    return _viewing(has_view=False)


def _view_of(sample_ids, label_ids=None):
    """Patches in a view restricted to the given IDs."""
    return _viewing(
        has_view=True,
        current_sample_ids=sample_ids,
        current_label_ids=label_ids,
    )


def _scoped_to(index, slice_name):
    """Points the index at a view scoped to the given group slice.

    A plain value rather than a ``PropertyMock``: ``_curr_view`` is an
    attribute ``use_view`` assigns, and a grouped collection is the only
    thing that carries a slice.
    """
    return mock.patch.object(
        index, "_curr_view", types.SimpleNamespace(group_slice=slice_name)
    )


def _filled(index, ids):
    """Fills an index with one row per ID.

    Which branch :meth:`_query_filters` takes turns on how much of the
    table a view holds, so a test that wants a particular one sizes the
    table for it rather than taking whatever a fixture has.
    """
    index.add_to_index(
        _random_embeddings(len(ids)), np.array(ids), reload=False
    )


def _nearest_ids(embeddings, ids, query, k, among=None):
    """The exhaustive answer, computed here rather than asked of the index.

    Args:
        embeddings: the ``num_rows x num_dims`` array the table was built from
        ids: the IDs of those rows, in the same order
        query: a query vector
        k: how many neighbors to return
        among (None): the IDs a view restricts the answer to

    Returns:
        a list of IDs, nearest first
    """
    order = np.argsort(((embeddings - query) ** 2).sum(axis=1), kind="stable")
    ranked = [ids[i] for i in order]

    if among is not None:
        among = set(among)
        ranked = [_id for _id in ranked if _id in among]

    return ranked[:k]


def _ids(index):
    return sorted(index.table.to_arrow()["id"].to_pylist())


def _sample_ids(index):
    return sorted(index.table.to_arrow()["sample_id"].to_pylist())


def _row_for(index, _id):
    rows = index.table.to_arrow().to_pylist()
    return next(row for row in rows if row["id"] == _id)


def _indexed_columns(index):
    return [config.columns for config in index.table.list_indices()]


def _id_index(index):
    return next(
        config
        for config in index.table.list_indices()
        if config.columns == ["id"]
    )


def _lance_index_meta(index):
    """The vector index's metadata, which `index_stats` does not carry."""
    return index.table.to_lance().stats.index_stats(_VECTOR_INDEX_NAME)[
        "indices"
    ][0]


def _vector_indexes(index):
    return [
        config
        for config in index.table.list_indices()
        if config.columns == ["vector"]
    ]


def _plan(index, query, k=1, **kwargs):
    """The plan for the query the connector itself builds.

    Routed through ``_search`` rather than rebuilt here, so that a knob the
    connector stops applying shows up as a changed plan.
    """
    search = index._search(
        query, _SUPPORTED_METRICS[index.config.metric], k, **kwargs
    )
    return " ".join(search.explain_plan(True).split())


def _data_files(index):
    path = os.path.join(index.table.uri, "data")
    return sorted(os.listdir(path)) if os.path.isdir(path) else []


def _num_fragments(index):
    return index.table.stats()["fragment_stats"]["num_fragments"]


def _deletion_files(index):
    path = os.path.join(index.table.uri, "_deletions")
    return sorted(os.listdir(path)) if os.path.isdir(path) else []


@pytest.fixture(name="index")
def fixture_index(tmp_path):
    """A LanceDB index over an empty temp URI.

    ``SimilarityIndex.__init__`` binds the index to a sample collection, which
    needs a database. The write paths under test reach only the table, the
    connection and the config, so the instance is built without that binding.

    The base class's ``reload`` is patched rather than the connector's, so
    that the connector's override still runs and the view refresh underneath
    it becomes the no-op.
    """
    index = _detached_index(
        LanceDBSimilarityConfig(
            table_name="test",
            uri=str(tmp_path),
            metric="euclidean",
            min_index_rows=0,
        )
    )

    with mock.patch.object(SimilarityIndex, "reload") as reload:
        index.reload_mock = reload
        yield index


@pytest.fixture(name="populated_index")
def fixture_populated_index(index):
    """An index holding rows ``a``, ``b`` and ``c``."""
    index.add_to_index(
        _random_embeddings(3), np.array(["a", "b", "c"]), reload=False
    )
    return index


@pytest.fixture(name="basis_index")
def fixture_basis_index(index):
    """An index whose rows are unit basis vectors.

    Nearest-neighbor order is then unambiguous, so a query can assert which
    row comes back rather than only how many.
    """
    index.add_to_index(
        _basis_embeddings(3), np.array(["a", "b", "c"]), reload=False
    )
    return index


@pytest.fixture(name="sliced_index")
def fixture_sliced_index(index):
    """An index whose table carries a slice column, as a grouped one's does.

    The resolution is patched rather than driven from a dataset: what the
    table holds is what the query path reads, and building a grouped
    collection to reach it would need a database this module does without.
    """
    with mock.patch.object(
        LanceDBSimilarityIndex,
        "_resolve_slice_names",
        return_value=["left", "left", "right"],
    ):
        index.add_to_index(
            _basis_embeddings(3), np.array(["a", "b", "c"]), reload=False
        )

    return index


@pytest.fixture(name="batched_index")
def fixture_batched_index(index):
    """An index holding more rows than fit in one predicate batch."""
    ids = np.array(["id-%05d" % i for i in range(BATCHED_ROWS)])
    index.add_to_index(_random_embeddings(BATCHED_ROWS), ids, reload=False)
    return index, list(ids)


class TestPredicates:
    """Building the SQL a query and a write are restricted by."""

    @pytest.mark.parametrize(
        "ids,expected",
        [
            pytest.param(["a"], "id IN ('a')", id="single"),
            pytest.param(["a", "b"], "id IN ('a', 'b')", id="several"),
            pytest.param(
                ["o'brien"], "id IN ('o''brien')", id="quote_is_doubled"
            ),
        ],
    )
    def test_predicate(self, ids, expected):
        assert _id_predicate(ids) == expected

    def test_a_negated_predicate_matches_everything_else(self):
        assert _id_predicate(["a", "b"], negate=True) == "id NOT IN ('a', 'b')"

    def test_a_column_can_be_named(self):
        assert (
            _id_predicate(["s1"], column="sample_id") == "sample_id IN ('s1')"
        )

    def test_quoted_id_round_trips(self, index):
        index.add_to_index(
            _random_embeddings(2),
            np.array(["o'brien", "plain"]),
            reload=False,
        )

        assert index._get_existing_ids(["o'brien"]) == ["o'brien"]

    def test_predicate_shaped_id_removes_only_its_own_row(self, index):
        # An ID that closes the quote and appends a disjunction would match
        # every row, so a removal would empty the table rather than take one
        # row out of it
        injection = "x') OR ('1'='1"
        index.add_to_index(
            _random_embeddings(3),
            np.array([injection, "plain", "other"]),
            reload=False,
        )

        index.remove_from_index(sample_ids=[injection], reload=False)

        assert _ids(index) == ["other", "plain"]


class TestRequirements:
    """The declared lancedb floor."""

    @pytest.mark.parametrize(
        "method",
        [
            pytest.param("ensure_requirements", id="install"),
            pytest.param("ensure_usage_requirements", id="usage"),
        ],
    )
    def test_the_floor_is_declared_to_fiftyone(self, method):
        # 0.34.0 is the first release whose `create_index` accepts `config=`;
        # 0.33.0 raises `TypeError` on it. Declaring a bare "lancedb" would
        # satisfy this check on a version the write path cannot run on
        backend = LanceDBSimilarity(LanceDBSimilarityConfig())

        with mock.patch.object(
            lancedb_backend.fou, "ensure_package"
        ) as ensure_package:
            getattr(backend, method)()

        # The literal rather than the constant, which would compare equal to
        # itself whatever it was set to
        ensure_package.assert_called_once_with("lancedb")


class TestTableNames:
    """What the connector does with a listing longer than one page.

    That :func:`_table_names` pages correctly is covered against a real
    database in ``tests/test_lancedb_store.py``. These cover the callers --
    the paths that would strand a table if the listing came back short --
    and the two ways the walk could fail to terminate.
    """

    # More tables than one default page holds
    NUM_TABLES = 25

    def _make_tables(self, uri):
        db = lancedb.connect(uri)
        pa_table = _to_arrow_table(["a"], ["a"], _random_embeddings(1))
        names = ["table-%03d" % i for i in range(self.NUM_TABLES)]
        for name in names:
            db.create_table(name, pa_table)

        return db, names

    def test_existing_table_past_first_page_is_opened(self, tmp_path):
        _, names = self._make_tables(str(tmp_path))
        stranded = names[-1]

        index = _detached_index(
            LanceDBSimilarityConfig(table_name=stranded, uri=str(tmp_path))
        )

        # A missed table reads as a new index, which strands the rows already
        # written and makes the next add fail against the table that is there
        assert index._table is not None
        assert index._table.name == stranded

    def test_cleanup_drops_table_past_first_page(self, tmp_path):
        db, names = self._make_tables(str(tmp_path))
        stranded = names[-1]

        index = _detached_index(
            LanceDBSimilarityConfig(table_name=stranded, uri=str(tmp_path))
        )
        index.cleanup()

        # Asserted through `open_table` rather than `_table_names`: a
        # truncated listing would not contain the 25th table either, so
        # checking absence there passes whether or not the drop happened
        with pytest.raises(ValueError, match="was not found"):
            db.open_table(stranded)

    def test_a_release_without_list_tables_uses_the_older_listing(self):
        # `list_tables` arrives in lancedb 0.27.1. Before it only
        # `table_names` exists, which pages on the last name it returned
        # rather than on a token of its own
        db = mock.Mock(spec=["table_names"])
        db.table_names.side_effect = [["a", "b"], ["c"], []]

        assert _table_names(db) == ["a", "b", "c"]
        assert db.table_names.call_args_list == [
            mock.call(page_token=None, limit=_DB_TABLE_PG_LIMIT),
            mock.call(page_token="b", limit=_DB_TABLE_PG_LIMIT),
            mock.call(page_token="c", limit=_DB_TABLE_PG_LIMIT),
        ]

    def test_the_newer_listing_is_preferred_when_present(self):
        # `table_names` is deprecated from 0.38.0, so it is the fallback
        # rather than the path
        page = mock.Mock(tables=["a"], page_token=None)
        db = mock.Mock(spec=["list_tables", "table_names"])
        db.list_tables.return_value = page

        assert _table_names(db) == ["a"]
        db.table_names.assert_not_called()

    def test_pages_through_every_response(self):
        # A storage key, the form 0.38.0 and later hand back: the last row
        # returned, which has already been collected
        page_one = mock.Mock(tables=["a", "b"], page_token="b.lance/")
        page_two = mock.Mock(tables=["c"], page_token=None)
        db = mock.Mock(spec=["list_tables"])
        db.list_tables.side_effect = [page_one, page_two]

        assert _table_names(db) == ["a", "b", "c"]
        assert db.list_tables.call_args_list == [
            mock.call(page_token=None, limit=_DB_TABLE_PG_LIMIT),
            mock.call(page_token="b.lance/", limit=_DB_TABLE_PG_LIMIT),
        ]

    def test_recovers_the_name_an_older_cursor_skips(self):
        # Before 0.38.0 the token is the next table's name and the request
        # it is passed to resumes after it, so that table is never listed.
        # Taking the token is what puts it back
        page_one = mock.Mock(tables=["a", "b"], page_token="c")
        page_two = mock.Mock(tables=["d"], page_token=None)
        db = mock.Mock(spec=["list_tables"])
        db.list_tables.side_effect = [page_one, page_two]

        assert _table_names(db) == ["a", "b", "c", "d"]

    def test_a_recovered_name_is_not_repeated(self):
        # A token naming something already collected is not a skip
        page_one = mock.Mock(tables=["a", "b"], page_token="b")
        page_two = mock.Mock(tables=["c"], page_token=None)
        db = mock.Mock(spec=["list_tables"])
        db.list_tables.side_effect = [page_one, page_two]

        assert _table_names(db) == ["a", "b", "c"]

    def test_stops_on_an_empty_page(self):
        # A server that keeps handing back a token would otherwise spin here.
        # The side effect is bounded so a walk that fails to stop raises on
        # its second page instead of hanging the suite
        page = mock.Mock(tables=[], page_token="always")
        db = mock.Mock(spec=["list_tables"])
        db.list_tables.side_effect = [page, _DID_NOT_STOP]

        assert _table_names(db) == []

    def test_stops_on_a_container_that_yields_nothing(self):
        # A MagicMock is truthy but iterates empty, so a guard that tests the
        # response object rather than its rows never terminates here. Any
        # test that mocks the lancedb module reaches this path
        db = mock.MagicMock()
        db.list_tables.side_effect = [mock.MagicMock(), _DID_NOT_STOP]

        assert _table_names(db) == []


class TestStaleHandle:
    """Two index objects open on the same table.

    Lance pins a handle to the version it was opened at, so a second writer's
    commits are invisible until the handle is moved forward. On a write path
    that stale view breaks the uniqueness of ``id``.
    """

    def _handle(self, tmp_path):
        return _detached_index(
            LanceDBSimilarityConfig(
                table_name="shared", uri=str(tmp_path), metric="euclidean"
            )
        )

    def test_upsert_sees_another_writers_row(self, tmp_path):
        first = self._handle(tmp_path)
        first.add_to_index(
            _constant_embeddings(1, 1.0), np.array(["a"]), reload=False
        )

        second = self._handle(tmp_path)
        second.add_to_index(
            _constant_embeddings(1, 2.0), np.array(["b"]), reload=False
        )

        # `first` has not seen "b", so an insert-only merge would write a
        # second row under that ID rather than updating the one there
        first.add_to_index(
            _constant_embeddings(1, 3.0),
            np.array(["b"]),
            overwrite=True,
            reload=False,
        )

        assert _ids(first) == ["a", "b"]
        assert _row_for(first, "b")["vector"][0] == 3.0

    def test_existence_check_sees_another_writers_row(self, tmp_path):
        first = self._handle(tmp_path)
        first.add_to_index(
            _random_embeddings(1), np.array(["a"]), reload=False
        )

        second = self._handle(tmp_path)
        second.add_to_index(
            _random_embeddings(1), np.array(["b"]), reload=False
        )

        # A stale handle reports "b" missing, so this would raise
        first.remove_from_index(
            sample_ids=["b"], allow_missing=False, reload=False
        )

        assert _ids(first) == ["a"]

    def test_sync_opens_a_table_created_after_initialize(self, tmp_path):
        # Both handles open before the table exists, so this one has nothing
        # to move forward — it has to open the table instead
        early = self._handle(tmp_path)
        writer = self._handle(tmp_path)
        writer.add_to_index(
            _random_embeddings(1), np.array(["z"]), reload=False
        )

        assert early._table is None

        early._sync_table()

        assert early._table is not None
        assert early.total_index_size == 1


class TestToArrowTable:
    """Building the Arrow table that backs a write."""

    def test_schema_and_contents(self):
        table = _to_arrow_table(
            ["a", "b"], ["s1", "s2"], _random_embeddings(2)
        )

        assert table.column_names == ["id", "sample_id", "vector"]
        assert table["id"].to_pylist() == ["a", "b"]
        assert table["sample_id"].to_pylist() == ["s1", "s2"]
        assert table["vector"].type.list_size == DIMS

    def test_accepts_numpy_id_arrays(self):
        # `fbu.get_ids()` returns numpy arrays, so this is the production shape
        table = _to_arrow_table(
            np.array(["a", "b"]),
            np.array(["s1", "s2"]),
            _random_embeddings(2),
        )

        assert table["id"].to_pylist() == ["a", "b"]
        assert table["sample_id"].to_pylist() == ["s1", "s2"]

    def test_slice_names_become_a_column(self):
        table = _to_arrow_table(
            ["a", "b"], ["s1", "s2"], _random_embeddings(2), ["left", "right"]
        )

        assert table.column_names == [
            "id",
            "sample_id",
            "vector",
            _SLICE_NAME_COLUMN,
        ]
        assert table[_SLICE_NAME_COLUMN].to_pylist() == ["left", "right"]

    def test_unresolved_slice_names_are_still_typed(self):
        # An untyped null column will not merge into a string one, so a
        # batch whose samples are all missing has to say what it is
        table = _to_arrow_table(["a"], ["s1"], _random_embeddings(1), [None])

        assert table.schema.field(_SLICE_NAME_COLUMN).type == pa.string()

    def test_float64_embeddings_are_cast(self):
        embeddings = np.ones((2, DIMS), dtype=np.float64)

        table = _to_arrow_table(["a", "b"], ["s1", "s2"], embeddings)

        assert table["vector"].type.value_type == pa.float32()
        assert table["vector"].to_pylist()[0] == [1.0] * DIMS


class TestIdListNormalization:
    """Turning a caller's IDs into a list."""

    @pytest.mark.parametrize(
        "ids,expected",
        [
            pytest.param(["a", "b"], ["a", "b"], id="list"),
            pytest.param("abc", ["abc"], id="scalar_string_is_not_split"),
            pytest.param(None, [], id="none_is_empty"),
        ],
    )
    def test_normalizes(self, ids, expected):
        assert _to_id_list(ids) == expected

    def test_scalar_sample_id_is_not_split_by_add(self, index):
        # `sample_ids` needs the same guard as `ids`, or the Arrow table is
        # built with one ID and six sample IDs and fails on column length
        index.add_to_index(
            _random_embeddings(1), "abcdef", label_ids=["label"], reload=False
        )

        assert _sample_ids(index) == ["abcdef"]

    def test_removing_nothing_removes_nothing(self, populated_index):
        # A caller passing neither ID argument must be a no-op. Checked
        # with `allow_missing=False`: treating None as an ID removes no rows
        # either way, so the only visible difference is a bogus "not
        # present" error naming an ID the caller never supplied
        populated_index.remove_from_index(allow_missing=False, reload=False)

        assert _ids(populated_index) == ["a", "b", "c"]


class TestAddToIndex:
    """Adding embeddings to the index."""

    def test_first_add_creates_the_table(self, index):
        assert index.table is None

        index.add_to_index(
            _random_embeddings(3), np.array(["a", "b", "c"]), reload=False
        )

        assert index.total_index_size == 3
        assert _ids(index) == ["a", "b", "c"]

    @pytest.mark.parametrize(
        "new_ids",
        [
            pytest.param(["d", "e"], id="fewer_than_the_table"),
            pytest.param(["d", "e", "f"], id="same_size_as_the_table"),
        ],
    )
    def test_second_add_appends(self, populated_index, new_ids):
        # Concatenating a list with a numpy array broadcasts instead of
        # appending. At equal lengths that would rewrite every sample_id into
        # a concatenated string rather than raising
        populated_index.add_to_index(
            _random_embeddings(len(new_ids), seed=1),
            np.array(new_ids),
            reload=False,
        )

        expected = sorted(["a", "b", "c"] + new_ids)
        assert _ids(populated_index) == expected
        assert _sample_ids(populated_index) == expected

    def test_label_ids_become_the_index_ids(self, index):
        index.add_to_index(
            _random_embeddings(2),
            np.array(["s1", "s2"]),
            label_ids=np.array(["l1", "l2"]),
            reload=False,
        )

        assert _ids(index) == ["l1", "l2"]
        assert _sample_ids(index) == ["s1", "s2"]

    def test_repeated_sample_ids_are_allowed_across_label_ids(self, index):
        # A patch index holds many labels per sample, so only the index IDs
        # have to be unique
        index.add_to_index(
            _random_embeddings(3),
            np.array(["s1", "s1", "s2"]),
            label_ids=np.array(["l1", "l2", "l3"]),
            reload=False,
        )

        assert _ids(index) == ["l1", "l2", "l3"]
        assert _sample_ids(index) == ["s1", "s1", "s2"]

    def test_overwrite_replaces_the_whole_row(self, index):
        index.add_to_index(
            _random_embeddings(1),
            np.array(["s1"]),
            label_ids=np.array(["l1"]),
            reload=False,
        )

        index.add_to_index(
            _constant_embeddings(1, 1.0),
            np.array(["s2"]),
            label_ids=np.array(["l1"]),
            reload=False,
        )

        row = _row_for(index, "l1")
        assert index.total_index_size == 1
        assert row["vector"] == [1.0] * DIMS
        assert row["sample_id"] == "s2"

    def test_overwrite_updates_and_inserts_in_one_batch(self, populated_index):
        original_b = _row_for(populated_index, "b")["vector"]

        populated_index.add_to_index(
            _constant_embeddings(2, 1.0), np.array(["a", "d"]), reload=False
        )

        assert _ids(populated_index) == ["a", "b", "c", "d"]
        assert _row_for(populated_index, "a")["vector"] == [1.0] * DIMS
        assert _row_for(populated_index, "d")["vector"] == [1.0] * DIMS
        assert _row_for(populated_index, "b")["vector"] == original_b

    def test_no_overwrite_keeps_the_vector(self, populated_index):
        original = _row_for(populated_index, "a")["vector"]

        populated_index.add_to_index(
            _constant_embeddings(1, 1.0),
            np.array(["a"]),
            overwrite=False,
            reload=False,
        )

        assert populated_index.total_index_size == 3
        assert _row_for(populated_index, "a")["vector"] == original

    def test_no_overwrite_still_adds_the_new_ids(self, populated_index):
        populated_index.add_to_index(
            _constant_embeddings(2, 1.0),
            np.array(["a", "d"]),
            overwrite=False,
            reload=False,
        )

        assert _ids(populated_index) == ["a", "b", "c", "d"]
        assert _row_for(populated_index, "d")["vector"] == [1.0] * DIMS

    @pytest.mark.parametrize(
        "warn_existing",
        [
            pytest.param(False, id="without_warn_existing"),
            pytest.param(True, id="with_warn_existing"),
        ],
    )
    def test_allow_existing_false_raises(self, populated_index, warn_existing):
        # Both terms of the lookup guard matter: with `warn_existing` also
        # set, an `or` that collapsed to the wrong operand would skip the
        # lookup and upsert over the existing row instead of raising
        with pytest.raises(ValueError, match="already exist"):
            populated_index.add_to_index(
                _random_embeddings(1),
                np.array(["a"]),
                allow_existing=False,
                warn_existing=warn_existing,
                reload=False,
            )

    def test_allow_existing_false_passes_for_new_ids(self, populated_index):
        populated_index.add_to_index(
            _random_embeddings(1),
            np.array(["d"]),
            allow_existing=False,
            reload=False,
        )

        assert _ids(populated_index) == ["a", "b", "c", "d"]

    @pytest.mark.parametrize(
        "overwrite,expected",
        [
            pytest.param(
                True,
                "Overwriting 1 IDs that already exist in the index",
                id="overwrite",
            ),
            pytest.param(
                False,
                "Skipping 1 IDs that already exist in the index",
                id="skip",
            ),
        ],
    )
    def test_warn_existing_logs(
        self, populated_index, caplog, overwrite, expected
    ):
        with caplog.at_level(logging.WARNING):
            populated_index.add_to_index(
                _random_embeddings(1),
                np.array(["a"]),
                overwrite=overwrite,
                warn_existing=True,
                reload=False,
            )

        assert [r.getMessage() for r in caplog.records] == [expected]

    def test_no_warning_without_warn_existing(self, populated_index, caplog):
        with caplog.at_level(logging.WARNING):
            populated_index.add_to_index(
                _random_embeddings(1), np.array(["a"]), reload=False
            )

        assert caplog.records == []

    def test_duplicate_ids_in_one_batch_raise(self, index):
        with pytest.raises(ValueError, match="duplicate IDs"):
            index.add_to_index(
                _random_embeddings(2), np.array(["a", "a"]), reload=False
            )

    @pytest.mark.parametrize(
        "populated",
        [
            pytest.param(False, id="empty_index"),
            pytest.param(True, id="populated_index"),
        ],
    )
    def test_empty_batch_is_a_no_op(self, index, populated):
        # `fbu.get_embeddings()` returns an empty array rather than None when
        # a collection yields no embeddings, and `compute_similarity` only
        # guards against None, so this batch shape reaches the connector
        expected = 0
        if populated:
            index.add_to_index(
                _random_embeddings(2), np.array(["a", "b"]), reload=False
            )
            expected = 2

        index.add_to_index(
            np.empty((0, 0), dtype=np.float64),
            np.array([], dtype="<U24"),
            reload=False,
        )

        assert index.total_index_size == expected

    def test_joins_a_table_another_writer_created(self, index, tmp_path):
        # Two handles can both open before either has written, and the loser
        # of the race must merge into the table rather than fail on it
        other = _detached_index(
            LanceDBSimilarityConfig(
                table_name="test", uri=str(tmp_path), metric="euclidean"
            )
        )
        assert index.table is None and other.table is None

        index.add_to_index(
            _random_embeddings(1), np.array(["a"]), reload=False
        )
        other.add_to_index(
            _random_embeddings(1, seed=1), np.array(["b"]), reload=False
        )

        assert _ids(other) == ["a", "b"]

    def test_second_add_does_not_recreate_the_table(self, populated_index):
        # Recreating the table would make add cost scale with table size
        with mock.patch.object(
            type(populated_index._db), "create_table", autospec=True
        ) as create_table:
            populated_index.add_to_index(
                _random_embeddings(2, seed=1),
                np.array(["d", "e"]),
                reload=False,
            )

        create_table.assert_not_called()
        assert _ids(populated_index) == ["a", "b", "c", "d", "e"]

    def test_race_fallback_joins_a_table_created_mid_add(
        self, index, tmp_path
    ):
        # `_sync_table` opens a table another writer already committed, so it
        # handles the wide race and this narrower one is left: a table that
        # appears between that listing and `create_table`. Patched out here
        # because otherwise the fallback is unreachable
        other = lancedb.connect(str(tmp_path))
        other.create_table(
            "test", _to_arrow_table(["a"], ["a"], _random_embeddings(1))
        )

        with mock.patch.object(LanceDBSimilarityIndex, "_sync_table"):
            index.add_to_index(
                _random_embeddings(1, seed=1), np.array(["b"]), reload=False
            )

        assert _ids(index) == ["a", "b"]

    def test_race_fallback_takes_the_schema_the_winner_chose(
        self, index, tmp_path
    ):
        # A writer on a release without the slice column creates a table
        # with three, and Lance rejects a merge whose source names a
        # fourth. The rows were built before that table was known, so they
        # are rebuilt once it is
        other = lancedb.connect(str(tmp_path))
        other.create_table(
            "test", _to_arrow_table(["a"], ["a"], _random_embeddings(1))
        )

        # A real grouped collection, so the resolution under test is the
        # connector's own rather than a stand-in for it
        view = mock.MagicMock()
        view.select.return_value.values.return_value = (["b"], ["left"])
        dataset = mock.MagicMock(group_field="group")
        dataset.select_group_slices.return_value = view
        index._samples = mock.MagicMock(_root_dataset=dataset)

        with mock.patch.object(LanceDBSimilarityIndex, "_sync_table"):
            index.add_to_index(
                _random_embeddings(1, seed=1), np.array(["b"]), reload=False
            )

        assert _ids(index) == ["a", "b"]
        assert _SLICE_NAME_COLUMN not in index.table.schema.names

    def test_create_failure_propagates_when_no_table_appeared(self, index):
        # Only a lost race is recoverable; anything else must surface
        with mock.patch.object(
            type(index._db),
            "create_table",
            autospec=True,
            side_effect=RuntimeError("disk full"),
        ), pytest.raises(RuntimeError, match="disk full"):
            index.add_to_index(
                _random_embeddings(1), np.array(["a"]), reload=False
            )

    def test_add_appends_a_fragment_instead_of_rewriting(
        self, populated_index
    ):
        # Fragment count, not file presence: Lance leaves a superseded data
        # file on disk until compaction, so a whole-table rewrite also leaves
        # the original there and any assertion about files still holding it
        # passes against the write path this one exists to reject. A rewrite
        # keeps the fragment count flat; appending raises it
        before = _num_fragments(populated_index)

        populated_index.add_to_index(
            _random_embeddings(2, seed=1), np.array(["d", "e"]), reload=False
        )

        assert _num_fragments(populated_index) == before + 1


class TestIdIndex:
    """The scalar index on the ``id`` column."""

    @builds_indexes
    def test_a_release_without_config_keeps_the_backend(self, index, caplog):
        # `create_index(config=)` arrives in 0.34.0 and is the only call
        # here that needs it. Older releases reject the keyword, and the
        # rows still have to land: a write without the index scans the id
        # column, which is slower and not wrong
        with mock.patch.object(
            lancedb_backend.lancedb.index,
            "BTree",
            side_effect=TypeError(
                "LanceTable.create_index() got an unexpected keyword "
                "argument 'config'"
            ),
        ), caplog.at_level(logging.WARNING):
            index.add_to_index(
                _random_embeddings(2), np.array(["a", "b"]), reload=False
            )
            after_first = len(caplog.records)

            index.add_to_index(
                _random_embeddings(1, seed=1), np.array(["c"]), reload=False
            )

        assert _ids(index) == ["a", "b", "c"]
        assert ["id"] not in _indexed_columns(index)
        assert _ID_INDEX_REQUIREMENT in caplog.text
        # Warned once, not once per add
        assert len(caplog.records) == after_first == 1

    @builds_indexes
    def test_created_on_first_add(self, populated_index):
        assert ["id"] in _indexed_columns(populated_index)

    @builds_indexes
    def test_added_to_a_table_that_lacks_it(self, index):
        index.add_to_index(
            _random_embeddings(2), np.array(["a", "b"]), reload=False
        )

        # Named rather than taken by position: the table carries a vector
        # index too, and dropping that one would test nothing
        index.table.drop_index(_id_index(index).name)
        assert ["id"] not in _indexed_columns(index)

        index.add_to_index(
            _random_embeddings(1, seed=1), np.array(["c"]), reload=False
        )

        assert ["id"] in _indexed_columns(index)

    def test_second_add_does_not_rebuild_it(self, populated_index):
        # `create_index` replaces by default, so an add that calls it
        # unconditionally rebuilds the whole column every time, which is the
        # table-size-proportional cost the incremental write path removes
        with mock.patch.object(
            type(populated_index.table), "create_index", autospec=True
        ) as create_index:
            populated_index.add_to_index(
                _random_embeddings(1, seed=1),
                np.array(["d"]),
                reload=False,
            )

        create_index.assert_not_called()

    @builds_indexes
    def test_a_failed_index_does_not_fail_the_add(
        self, populated_index, caplog
    ):
        # The rows are committed by this point, and concurrent writers race to
        # build the index, so the loser must not take the add down with it
        with mock.patch.object(
            type(populated_index.table),
            "list_indices",
            autospec=True,
            return_value=[],
        ), mock.patch.object(
            type(populated_index.table),
            "create_index",
            autospec=True,
            side_effect=RuntimeError("retryable commit conflict"),
        ), caplog.at_level(
            logging.WARNING
        ):
            populated_index.add_to_index(
                _random_embeddings(1, seed=1),
                np.array(["d"]),
                reload=False,
            )

        assert _ids(populated_index) == ["a", "b", "c", "d"]
        assert "Failed to index the 'id' column" in caplog.text


#: Every family, and whether the connector supplies it parameters when the
#: config names none: `None` marks the width-dependent ones. Spelled out
#: rather than derived from the backend, and checked against it below, so a
#: family added there has to have its default decided here instead of
#: inheriting the empty case by being forgotten
_DEFAULTED_FAMILIES = {
    "ivf_flat": {},
    "ivf_sq": {},
    "ivf_rq": {},
    "ivf_hnsw_flat": {},
    "ivf_hnsw_sq": {},
    "ivf_pq": None,
    "ivf_hnsw_pq": None,
}


class TestDefaultIndexParams:
    """The parameters a family is given when the config names none."""

    def test_every_supported_family_is_listed(self):
        assert set(_DEFAULTED_FAMILIES) == set(_SUPPORTED_INDEX_TYPES)

    @pytest.mark.parametrize(
        "index_type",
        sorted(k for k, v in _DEFAULTED_FAMILIES.items() if v == {}),
    )
    def test_a_family_without_parameters_supplies_none(self, index_type):
        assert _default_index_params(index_type, 1024) == {}

    # Three widths rather than one: at a single width an eighth is
    # indistinguishable from the constant it happens to equal. 1000 is not a
    # multiple of 16, so a half-of-a-sixteenth spelling fails it too
    @pytest.mark.parametrize(
        "dims,expected", [(768, 96), (1000, 125), (2048, 256)]
    )
    @pytest.mark.parametrize(
        "index_type",
        sorted(k for k, v in _DEFAULTED_FAMILIES.items() if v is None),
    )
    def test_a_pq_family_splits_the_width_by_eight(
        self, index_type, dims, expected
    ):
        # LanceDB's own default is a sixteenth of the width, which measures
        # 0.9447 mean / 0.200 worst at 768 against an eighth's 0.9963 / 0.800
        assert _default_index_params(index_type, dims) == {
            "num_sub_vectors": expected
        }


class TestVectorIndex:
    """The vector index on the ``vector`` column."""

    @builds_indexes
    def test_every_supported_family_names_a_real_class(self):
        # The map holds class names rather than classes so that importing
        # this module does not import lancedb, which is optional -- the
        # cost being that a name LanceDB has renamed is not caught until
        # a build tries it, and `_ensure_vector_index` answers that with a
        # warning. Here, where lancedb is present, it is caught
        missing = [
            name
            for name in _SUPPORTED_INDEX_TYPES.values()
            if not hasattr(lancedb.index, name)
        ]

        assert missing == []

    @builds_indexes
    def test_a_table_under_the_crossover_is_left_unindexed(self, tmp_path):
        # A scan beats an indexed query on a small table, because the index
        # read is a fixed cost the scan does not pay
        index = _unbound_index(tmp_path, min_index_rows=8)
        index.add_to_index(_random_embeddings(4), _row_ids(4), reload=False)

        assert not _vector_indexes(index)

    @builds_indexes
    def test_the_add_that_crosses_it_builds_the_index(self, tmp_path):
        # Checked per add, so growth past the threshold is picked up rather
        # than settled once when the table was small
        index = _unbound_index(tmp_path, min_index_rows=8)
        index.add_to_index(_random_embeddings(4), _row_ids(4), reload=False)
        assert not _vector_indexes(index)

        index.add_to_index(
            _random_embeddings(4, seed=1),
            np.array(["late-%d" % i for i in range(4)]),
            reload=False,
        )

        assert _vector_indexes(index)[0].index_type == "IvfRq"

    @builds_indexes
    def test_created_on_first_add(self, populated_index):
        assert ["vector"] in _indexed_columns(populated_index)

    @builds_indexes
    def test_narrow_embeddings_default_to_ivf_rq(self, populated_index):
        # RQ's 1-bit codes are best in class up to 768 and collapse above it
        assert _vector_indexes(populated_index)[0].index_type == "IvfRq"

    @pytest.mark.parametrize(
        "dims,expected",
        [
            pytest.param(_RQ_MAX_DIMS, "IvfRq", id="at_the_boundary"),
            pytest.param(_RQ_MAX_DIMS + 256, "IvfPq", id="above_it"),
        ],
    )
    @builds_indexes
    def test_the_family_follows_the_width(self, tmp_path, dims, expected):
        index = _unbound_index(tmp_path)
        rows = PQ_TRAINING_ROWS
        index.add_to_index(
            np.random.default_rng(0).random((rows, dims), dtype=np.float32),
            _row_ids(rows),
            reload=False,
        )

        assert _vector_indexes(index)[0].index_type == expected

    @builds_indexes
    @reads_index_metadata
    def test_a_pq_default_supplies_its_own_sub_vectors(self, tmp_path):
        # LanceDB's own default is a sixteenth of the width, which measures
        # 0.9447 mean / 0.200 worst at 768 against an eighth's 0.9963 / 0.800
        dims = _RQ_MAX_DIMS + 256
        index = _unbound_index(tmp_path)
        index.add_to_index(
            np.random.default_rng(0).random(
                (PQ_TRAINING_ROWS, dims), dtype=np.float32
            ),
            _row_ids(PQ_TRAINING_ROWS),
            reload=False,
        )

        sub_index = _lance_index_meta(index)["sub_index"]

        assert sub_index["num_sub_vectors"] == dims // 8

    # The read above needs pylance, so this is what checks in CI that the
    # computed count is the one LanceDB is handed. `min_index_rows` keeps
    # the first add under the crossover, so the table exists to patch
    # before any index is attempted
    @builds_indexes
    @pytest.mark.parametrize("dims", [1000, 1024])
    def test_a_pq_default_reaches_the_family(self, tmp_path, dims):
        index = _unbound_index(tmp_path, index_type="ivf_pq", min_index_rows=3)
        rng = np.random.default_rng(0)
        index.add_to_index(
            rng.random((2, dims), dtype=np.float32),
            _row_ids(2),
            reload=False,
        )

        with mock.patch.object(
            type(index.table), "create_index", autospec=True
        ) as create_index:
            index.add_to_index(
                rng.random((2, dims), dtype=np.float32),
                np.array(["extra-0", "extra-1"]),
                reload=False,
            )

        config = create_index.call_args.kwargs["config"]

        assert config.num_sub_vectors == dims // 8

    @builds_indexes
    def test_an_explicit_family_overrides_the_width(self, tmp_path):
        index = _seeded_index(tmp_path, index_type="ivf_flat")

        assert _vector_indexes(index)[0].index_type == "IvfFlat"

    @builds_indexes
    def test_named_so_a_rebuild_replaces(self, populated_index):
        assert _vector_indexes(populated_index)[0].name == _VECTOR_INDEX_NAME

    @pytest.mark.parametrize(
        "index_type,index_params,num_rows,expected",
        [
            pytest.param("ivf_flat", {}, 8, "IvfFlat", id="ivf_flat"),
            pytest.param("ivf_sq", {}, 8, "IvfSq", id="ivf_sq"),
            pytest.param(
                "ivf_hnsw_flat", {}, 8, "IvfHnswFlat", id="ivf_hnsw_flat"
            ),
            pytest.param(
                "ivf_hnsw_pq",
                {"num_sub_vectors": 2},
                PQ_TRAINING_ROWS,
                "IvfHnswPq",
                id="ivf_hnsw_pq",
            ),
            # A product quantizer trains on 256 rows, so this arm is the
            # expensive one and the reason `num_rows` is a parameter
            pytest.param(
                "ivf_pq",
                {"num_sub_vectors": 2},
                PQ_TRAINING_ROWS,
                "IvfPq",
                id="ivf_pq",
            ),
            pytest.param("ivf_rq", {}, PQ_TRAINING_ROWS, "IvfRq", id="ivf_rq"),
        ],
    )
    @builds_indexes
    def test_family_is_configurable(
        self, tmp_path, index_type, index_params, num_rows, expected
    ):
        index = _unbound_index(
            tmp_path, index_type=index_type, index_params=index_params
        )
        index.add_to_index(
            _random_embeddings(num_rows), _row_ids(num_rows), reload=False
        )

        assert _vector_indexes(index)[0].index_type == expected

    @builds_indexes
    def test_unsupported_index_type_raises(self, tmp_path):
        with pytest.raises(ValueError, match="Unsupported index type"):
            LanceDBSimilarityConfig(
                table_name="test", uri=str(tmp_path), index_type="brute_force"
            )

    @builds_indexes
    def test_index_params_reach_the_family(self, tmp_path, caplog):
        # `index_params` is the only route to a family's own knobs, so a bad
        # value has to surface as a failed build rather than be dropped. 8
        # dimensions does not divide into 3 sub-vectors
        index = _unbound_index(
            tmp_path,
            index_type="ivf_pq",
            index_params={"num_sub_vectors": 3},
        )

        with caplog.at_level(logging.WARNING):
            index.add_to_index(
                _random_embeddings(PQ_TRAINING_ROWS),
                _row_ids(PQ_TRAINING_ROWS),
                reload=False,
            )

        assert not _vector_indexes(index)
        assert "Failed to index the 'vector' column" in caplog.text

    @builds_indexes
    def test_a_build_too_small_to_train_is_retried_on_the_next_add(
        self, tmp_path, caplog
    ):
        # The guard is "build when absent", so a family that cannot train on
        # the rows present yet picks its index up once enough have arrived
        # rather than losing it for the life of the table
        index = _unbound_index(
            tmp_path, index_type="ivf_pq", index_params={"num_sub_vectors": 2}
        )

        with caplog.at_level(logging.WARNING):
            index.add_to_index(
                _random_embeddings(8), _row_ids(8), reload=False
            )

        assert not _vector_indexes(index)
        assert "train" in caplog.text

        index.add_to_index(
            _random_embeddings(PQ_TRAINING_ROWS, seed=1),
            np.array(["late-%05d" % i for i in range(PQ_TRAINING_ROWS)]),
            reload=False,
        )

        assert _vector_indexes(index)[0].index_type == "IvfPq"

    @pytest.mark.parametrize(
        "metric,expected",
        [
            pytest.param("cosine", "Cosine", id="cosine"),
            pytest.param("euclidean", "L2", id="euclidean"),
        ],
    )
    @builds_indexes
    def test_built_with_the_metric_the_queries_use(
        self, tmp_path, metric, expected
    ):
        # LanceDB answers a query whose metric disagrees with the index by
        # silently scanning every vector, reporting it only as a log line from
        # its Rust layer. A mismatch here is an index that is built, stored
        # and never read
        index = _unbound_index(tmp_path, metric=metric)
        index.add_to_index(
            _basis_embeddings(3), np.array(["a", "b", "c"]), reload=False
        )

        plan = _plan(index, _basis_embeddings(3)[0])

        assert "ANNSubIndex" in plan
        assert "metric=%s" % expected in plan

    @builds_indexes
    def test_a_query_is_not_an_exhaustive_scan(self, basis_index):
        plan = _plan(basis_index, _basis_embeddings(3)[0])

        assert "ANNSubIndex: name=%s" % _VECTOR_INDEX_NAME in plan

    @builds_indexes
    def test_the_bypass_control_is_an_exhaustive_scan(self, basis_index):
        # The control for the assertion above: same table, same query, index
        # forced off. Without it, "ANNSubIndex is present" says nothing about
        # what its absence would look like
        plan = " ".join(
            basis_index.table.search(_basis_embeddings(3)[0])
            .limit(1)
            .bypass_vector_index()
            .explain_plan(True)
            .split()
        )

        assert "ANNSubIndex" not in plan
        assert "KNNVectorDistance" in plan

    @builds_indexes
    def test_building_twice_leaves_one_index(self, populated_index):
        # `replace=True` is scoped to the index name, so a second build under
        # a different name would leave both and the query would keep using the
        # first. The guard is bypassed here to reach the second build at all
        populated_index._config.index_type = "ivf_flat"
        with mock.patch.object(
            type(populated_index.table),
            "list_indices",
            autospec=True,
            return_value=[],
        ):
            populated_index._ensure_vector_index()

        vector_indexes = _vector_indexes(populated_index)

        assert len(vector_indexes) == 1
        assert vector_indexes[0].index_type == "IvfFlat"

    @builds_indexes
    def test_rebuilt_when_the_metric_stops_matching(self, tmp_path):
        # An index whose distance type disagrees with the query's metric is
        # not used: LanceDB scans every vector instead and says so only in a
        # log line from its Rust layer, which looks exactly like an index
        # that is merely slow
        index = _seeded_index(tmp_path, metric="euclidean")
        assert (
            index.table.index_stats(_VECTOR_INDEX_NAME).distance_type == "l2"
        )

        index._config.metric = "cosine"
        index.add_to_index(
            _random_embeddings(1, seed=3), np.array(["d"]), reload=False
        )

        stats = index.table.index_stats(_VECTOR_INDEX_NAME)
        assert stats.distance_type == "cosine"
        assert "ANNSubIndex" in _plan(index, _basis_embeddings(3)[0])

    @builds_indexes
    def test_the_family_is_left_alone_once_built(self, tmp_path):
        # A rebuild costs minutes at ten million rows, so a changed
        # `index_type` is a no-op rather than a surprise on the next add
        index = _seeded_index(tmp_path, index_type="ivf_flat")

        index._config.index_type = "ivf_hnsw_sq"
        index.add_to_index(
            _random_embeddings(1, seed=3), np.array(["d"]), reload=False
        )

        assert _vector_indexes(index)[0].index_type == "IvfFlat"

    @builds_indexes
    def test_second_add_does_not_rebuild_it(self, populated_index):
        # A rebuild costs seconds at a million rows and minutes at ten, so an
        # add that calls `create_index` unconditionally is not an option
        with mock.patch.object(
            type(populated_index.table), "create_index", autospec=True
        ) as create_index:
            populated_index.add_to_index(
                _random_embeddings(1, seed=1), np.array(["d"]), reload=False
            )

        create_index.assert_not_called()

    @builds_indexes
    def test_a_bad_family_parameter_is_retried_not_latched(self, tmp_path):
        # A bad key in `index_params` reaches the family constructor as a
        # TypeError, the same type a release without `config=` raises. Read
        # as the latter it would report the wrong cause and turn the index
        # off for the life of the instance, so correcting the config would
        # never take
        index = _unbound_index(
            tmp_path, index_type="ivf_flat", index_params={"nope": 1}
        )
        index.add_to_index(_random_embeddings(8), _row_ids(8), reload=False)

        assert not index._vector_index_unavailable
        assert not _vector_indexes(index)

        index._config.index_params = {}
        index.add_to_index(
            _random_embeddings(8, seed=1),
            np.array(["late-%d" % i for i in range(8)]),
            reload=False,
        )

        assert _vector_indexes(index)[0].index_type == "IvfFlat"

    @builds_indexes
    def test_a_failed_index_does_not_fail_the_add(
        self, populated_index, caplog
    ):
        # A query without the index is slow rather than wrong, and the rows
        # are committed by this point, so the add must survive
        with mock.patch.object(
            type(populated_index.table),
            "list_indices",
            autospec=True,
            return_value=[],
        ), mock.patch.object(
            type(populated_index.table),
            "create_index",
            autospec=True,
            side_effect=RuntimeError("retryable commit conflict"),
        ), caplog.at_level(
            logging.WARNING
        ):
            populated_index.add_to_index(
                _random_embeddings(1, seed=1),
                np.array(["d"]),
                reload=False,
            )

        assert _ids(populated_index) == ["a", "b", "c", "d"]
        assert "Failed to index the 'vector' column" in caplog.text


class TestQueryKnobs:
    """``nprobes``, ``ef`` and ``refine_factor`` on a vector query."""

    def _knobs(self, index, query):
        """The knob calls the connector's query makes, in order.

        Recorded onto a real builder rather than a stand-in: the knobs have
        to be observed on the object LanceDB would receive.
        """
        builder = index.table.search(query).metric("l2").limit(1)
        recorder = mock.Mock()
        for knob in (
            "nprobes",
            "minimum_nprobes",
            "maximum_nprobes",
            "ef",
            "refine_factor",
        ):
            stub = getattr(recorder, knob)
            stub.return_value = builder
            setattr(builder, knob, stub)

        with mock.patch.object(
            type(index.table), "search", autospec=True, return_value=builder
        ):
            index._search(query, "l2", 1)

        return recorder.mock_calls

    @builds_indexes
    def test_the_default_family_escalates_rather_than_pinning(
        self, basis_index
    ):
        # A narrow width defaults to RQ, which builds about twice the
        # partitions of PQ at the same row count -- so one pinned value
        # probes half the share and recall falls as the table grows. The
        # maximum goes first: a minimum above the standing maximum is
        # rejected. Zero means unbounded
        assert self._knobs(basis_index, _basis_embeddings(3)[0]) == [
            mock.call.maximum_nprobes(0),
            mock.call.minimum_nprobes(1),
            mock.call.refine_factor(10),
        ]

    @builds_indexes
    def test_a_pinning_family_takes_the_deliberate_value(self, tmp_path):
        # Not left at LanceDB's implicit 20
        index = _seeded_index(tmp_path, index_type="ivf_pq")

        assert self._knobs(index, _basis_embeddings(3)[0]) == [
            mock.call.nprobes(_DEFAULT_NPROBES),
            mock.call.refine_factor(10),
        ]

    @builds_indexes
    def test_an_explicit_value_wins_over_either_regime(self, tmp_path):
        index = _seeded_index(tmp_path, nprobes=7)

        assert self._knobs(index, _basis_embeddings(3)[0]) == [
            mock.call.nprobes(7),
            mock.call.refine_factor(10),
        ]

    @builds_indexes
    def test_refine_factor_can_be_turned_off(self, tmp_path):
        # The probe regime is independent of it and still applies
        index = _seeded_index(tmp_path, refine_factor=None)

        assert self._knobs(index, _basis_embeddings(3)[0]) == [
            mock.call.maximum_nprobes(0),
            mock.call.minimum_nprobes(1),
        ]

    @builds_indexes
    def test_each_is_applied_when_set(self, tmp_path):
        # Built from constructor kwargs rather than by assigning to the
        # config, because the kwarg is the only route a caller has and
        # assignment would not notice it being dropped
        index = _seeded_index(tmp_path, nprobes=25, ef=128, refine_factor=10)

        assert self._knobs(index, _basis_embeddings(3)[0]) == [
            mock.call.nprobes(25),
            mock.call.ef(128),
            mock.call.refine_factor(10),
        ]

    @builds_indexes
    def test_a_set_knob_reaches_the_plan(self, tmp_path):
        # Reaching the plan is not the same as changing the answer -- see
        # the two tests below, which pin where it actually bites
        index = _seeded_index(tmp_path, nprobes=3)

        assert "minimum_nprobes=3" in _plan(index, _basis_embeddings(3)[0])

    # Enough rows that a partitioned family separates on probe count
    PARTITION_ROWS = 300

    def _ids_at_nprobes(self, index, nprobes):
        return (
            index.table.search(_random_embeddings(1, seed=9)[0])
            .metric("l2")
            .limit(10)
            .nprobes(nprobes)
            .to_pandas()["id"]
            .tolist()
        )

    def _partitioned(self, tmp_path, **config_kwargs):
        index = _unbound_index(tmp_path, **config_kwargs)
        index.add_to_index(
            _random_embeddings(self.PARTITION_ROWS),
            _row_ids(self.PARTITION_ROWS),
            reload=False,
        )
        return index

    @builds_indexes
    def test_nprobes_does_nothing_on_the_default_family(self, tmp_path):
        # `ivf_hnsw_sq` builds one IVF partition, so there is nothing for a
        # probe count to choose between. The plan still prints the value and
        # the engine ignores it, which is why the config documents `ef` as
        # the pruning knob there rather than this one
        index = self._partitioned(tmp_path)

        assert self._ids_at_nprobes(index, 1) == self._ids_at_nprobes(
            index, 50
        )

    @builds_indexes
    def test_nprobes_bites_once_the_family_has_partitions(self, tmp_path):
        # Guards the premise above: without it, a release that gave the HNSW
        # families real partitions would leave that test passing while its
        # reasoning had gone stale
        index = self._partitioned(
            tmp_path, index_params={"num_partitions": 16}
        )

        assert self._ids_at_nprobes(index, 1) != self._ids_at_nprobes(
            index, 50
        )


class TestConfigValidation:
    """Values rejected when the config is built rather than at query time."""

    @pytest.mark.parametrize(
        "kwargs,message",
        [
            pytest.param(
                {"metric": "hamming"}, "Unsupported metric", id="metric"
            ),
            pytest.param(
                {"index_type": "brute_force"},
                "Unsupported index type",
                id="index_type",
            ),
            pytest.param(
                {"index_params": "not-a-dict"},
                "index_params must be a dict",
                id="index_params_type",
            ),
            pytest.param(
                {"index_params": {"distance_type": "l2"}},
                "distance type is set from",
                id="index_params_distance_type",
            ),
            pytest.param(
                {"nprobes": 0}, "nprobes must be a positive", id="nprobes_zero"
            ),
            pytest.param(
                {"ef": -5}, "ef must be a positive", id="ef_negative"
            ),
            pytest.param(
                {"refine_factor": "ten"},
                "refine_factor must be a positive",
                id="refine_factor_type",
            ),
        ],
    )
    @builds_indexes
    def test_bad_values_raise_where_they_are_set(self, kwargs, message):
        # These otherwise surface from LanceDB's Rust layer at query time, in
        # whichever session loads the brain run rather than the one that set
        # them. `distance_type` is the worst of them: it collides with the
        # value taken from `metric` and leaves the table unindexed for good
        with pytest.raises(ValueError, match=message):
            LanceDBSimilarityConfig(**kwargs)


class TestIndexedDistances:
    """Distances an indexed query reports against an unindexed one."""

    def _both_paths(self, index, query):
        metric = _SUPPORTED_METRICS[index.config.metric]
        indexed = index._search(query, metric, 3).to_pandas()
        scanned = index._search(
            query, metric, 3, bypass_index=True
        ).to_pandas()
        return list(indexed._distance), list(scanned._distance)

    def test_the_default_refine_makes_them_agree(self, tmp_path):
        # `find_duplicates(thresh=...)` is a distance threshold a user tunes,
        # so the indexed and unindexed paths have to report on one scale
        index = _seeded_index(tmp_path, metric="cosine")

        indexed, scanned = self._both_paths(index, _basis_embeddings(3)[0])

        np.testing.assert_allclose(indexed, scanned, atol=1e-5)

    @builds_indexes
    def test_without_the_refine_they_disagree(self, tmp_path):
        # Guards the premise of the test above. How far the indexed values
        # sit from the scanned ones is the family's business -- SQ reports
        # twice, RQ a little under, PQ something else again -- so this pins
        # only that they differ, which is what the re-rank is there for
        index = _seeded_index(tmp_path, metric="cosine", refine_factor=None)

        indexed, scanned = self._both_paths(index, _basis_embeddings(3)[0])

        assert indexed != pytest.approx(scanned, abs=1e-5)


class TestExactK:
    """Returning every row the caller asked for."""

    # Partitions enough that probing one reaches a small share of the table
    SCATTERED = {
        "index_type": "ivf_flat",
        "index_params": {"num_partitions": 64},
    }
    ROWS = 500

    def _index(self, tmp_path, **kwargs):
        index = _unbound_index(tmp_path, nprobes=1, **self.SCATTERED, **kwargs)
        index.add_to_index(
            _random_embeddings(self.ROWS), _row_ids(self.ROWS), reload=False
        )
        return index

    @builds_indexes
    def test_an_indexed_query_alone_comes_back_short(self, tmp_path):
        # Guards the premise: without a table the index answers partially,
        # the fallback below would be untested rather than merely unused
        index = self._index(tmp_path, refine_factor=None)

        short = index._search(
            _random_embeddings(1)[0], "l2", self.ROWS
        ).to_pandas()

        assert len(short) < self.ROWS

    def test_a_short_result_falls_back_to_a_scan(self, tmp_path):
        # An indexed query sees only the partitions it probes, so a `k` near
        # the table size returns fewer rows than asked with no error, and
        # `sort_by_similarity` documents `k=None` as sorting every sample
        index = self._index(tmp_path, refine_factor=None)

        results = index._search_exactly_k(
            _random_embeddings(1)[0], "l2", self.ROWS
        )

        assert len(results) == self.ROWS

    def test_a_table_smaller_than_k_is_not_retried(self, populated_index):
        # Three rows cannot answer k=10, and a scan would not change that
        with mock.patch.object(
            LanceDBSimilarityIndex, "_search", wraps=populated_index._search
        ) as search:
            results = populated_index._search_exactly_k(
                _basis_embeddings(3)[0], "l2", 10
            )

        assert len(results) == 3
        assert search.call_count == 1


class TestPredicateBatching:
    """Splitting an ID list across predicates.

    Nothing about the resulting rows observes the split — a batch size large
    enough to hold every ID produces the same table — so these assert on the
    predicates themselves. Without that, disabling batching entirely goes
    unnoticed, and the row-count fixtures scale off ``_ID_BATCH_SIZE`` so they
    cannot notice either.
    """

    @pytest.fixture(name="small_batch", autouse=True)
    def fixture_small_batch(self, monkeypatch):
        monkeypatch.setattr(lancedb_backend, "_ID_BATCH_SIZE", 2)

    def test_lookup_splits_the_predicate(self, populated_index):
        with mock.patch.object(
            lancedb_backend,
            "_id_predicate",
            wraps=lancedb_backend._id_predicate,
        ) as id_predicate:
            found = populated_index._get_existing_ids(["a", "b", "c"])

        assert sorted(found) == ["a", "b", "c"]
        assert [
            list(call.args[0]) for call in id_predicate.call_args_list
        ] == [["a", "b"], ["c"]]

    def test_the_slice_lookup_splits_its_id_list(self, index):
        # The first add of an index carries the whole dataset, and a `$in`
        # naming every sample of a large one approaches the 16 MB ceiling
        # on the aggregation command that would carry it
        view = mock.MagicMock()
        view.select.return_value.values.return_value = ([], [])
        dataset = mock.MagicMock(group_field="group")
        dataset.select_group_slices.return_value = view
        index._samples = mock.MagicMock(_root_dataset=dataset)

        assert index._resolve_slice_names(["a", "b", "c"]) == [
            None,
            None,
            None,
        ]
        assert [list(call.args[0]) for call in view.select.call_args_list] == [
            ["a", "b"],
            ["c"],
        ]

    def test_delete_splits_the_predicate(self, populated_index):
        with mock.patch.object(
            type(populated_index.table),
            "delete",
            autospec=True,
            side_effect=type(populated_index.table).delete,
        ) as delete:
            populated_index.remove_from_index(
                sample_ids=["a", "b", "c"], reload=False
            )

        assert _ids(populated_index) == []
        assert [call.args[1] for call in delete.call_args_list] == [
            "id IN ('a', 'b')",
            "id IN ('c')",
        ]


class TestGetExistingIds:
    """Looking up which IDs the index already holds."""

    def test_empty_index_holds_nothing(self, index):
        assert index._get_existing_ids(["a"]) == []

    def test_returns_only_present_ids(self, populated_index):
        found = populated_index._get_existing_ids(["a", "c", "missing"])

        assert sorted(found) == ["a", "c"]

    def test_finds_rows_added_after_the_index_was_built(self, populated_index):
        # Rows written since the scalar index was built sit in its unindexed
        # tail. Reporting them absent would make a removal silently skip them
        # and stop `allow_existing=False` from raising
        populated_index.add_to_index(
            _random_embeddings(1, seed=1), np.array(["d"]), reload=False
        )

        assert populated_index._get_existing_ids(["d"]) == ["d"]

    def test_a_duplicated_row_is_reported_once(self, populated_index):
        # A raw `add` bypasses the merge, so the table can hold an ID twice.
        # Callers take `len()` of this list for the count they report, so a
        # duplicate row would inflate "Found N IDs" for a single ID
        populated_index.table.add(
            _to_arrow_table(["a"], ["a"], _random_embeddings(1, seed=2))
        )

        assert populated_index.total_index_size == 4
        assert populated_index._get_existing_ids(["a"]) == ["a"]

    def test_spans_more_ids_than_one_batch(self, batched_index):
        index, ids = batched_index

        found = index._get_existing_ids(ids + ["missing"])

        assert len(found) == BATCHED_ROWS


class TestRemoveFromIndex:
    """Removing embeddings from the index."""

    @pytest.mark.parametrize(
        "sample_ids,kwargs,expected",
        [
            pytest.param(["b"], {}, ["a", "c"], id="present_id"),
            pytest.param(
                ["a", "missing"], {}, ["b", "c"], id="missing_tolerated"
            ),
            pytest.param(
                ["a"],
                {"allow_missing": False},
                ["b", "c"],
                id="strict_and_all_present",
            ),
            pytest.param(
                ["a", "missing"],
                {"warn_missing": True},
                ["b", "c"],
                id="warns_and_removes_the_rest",
            ),
        ],
    )
    def test_removes_the_rows(
        self, populated_index, sample_ids, kwargs, expected
    ):
        populated_index.remove_from_index(
            sample_ids=sample_ids, reload=False, **kwargs
        )

        assert _ids(populated_index) == expected

    def test_leaves_data_files_untouched(self, populated_index):
        before = _data_files(populated_index)
        assert before

        populated_index.remove_from_index(sample_ids=["b"], reload=False)

        # Lance records the removal as a deletion sidecar rather than
        # rewriting the table, which is the point of the delete path
        assert _data_files(populated_index) == before
        assert _deletion_files(populated_index)

    def test_does_not_recreate_the_table(self, populated_index):
        with mock.patch.object(
            type(populated_index._db), "create_table", autospec=True
        ) as create_table:
            populated_index.remove_from_index(sample_ids=["b"], reload=False)

        create_table.assert_not_called()
        assert _ids(populated_index) == ["a", "c"]

    def test_removes_by_label_id(self, index):
        index.add_to_index(
            _random_embeddings(2),
            np.array(["s1", "s2"]),
            label_ids=np.array(["l1", "l2"]),
            reload=False,
        )

        index.remove_from_index(label_ids=["l1"], reload=False)

        assert _ids(index) == ["l2"]

    def test_allow_missing_false_raises(self, populated_index):
        with pytest.raises(ValueError, match="not present in the index"):
            populated_index.remove_from_index(
                sample_ids=["missing"], allow_missing=False, reload=False
            )

    def test_warn_missing_logs(self, populated_index, caplog):
        with caplog.at_level(logging.WARNING):
            populated_index.remove_from_index(
                sample_ids=["a", "missing"], warn_missing=True, reload=False
            )

        assert [r.getMessage() for r in caplog.records] == [
            "Ignoring 1 IDs that are not present in the index"
        ]

    def test_removes_a_row_added_after_the_index_was_built(
        self, populated_index
    ):
        populated_index.add_to_index(
            _random_embeddings(1, seed=1), np.array(["d"]), reload=False
        )

        populated_index.remove_from_index(
            sample_ids=["d"], allow_missing=False, reload=False
        )

        assert _ids(populated_index) == ["a", "b", "c"]

    def test_spans_more_ids_than_one_batch(self, batched_index):
        index, ids = batched_index
        index.add_to_index(
            _random_embeddings(1, seed=1), np.array(["keep"]), reload=False
        )

        index.remove_from_index(sample_ids=ids, reload=False)

        assert _ids(index) == ["keep"]

    def test_scalar_sample_id_is_not_split_into_characters(self, index):
        # `list("abc")` yields ['a', 'b', 'c'], which would address the wrong
        # rows instead of the one asked for
        index.add_to_index(
            _random_embeddings(3),
            np.array(["abc", "a", "b"]),
            reload=False,
        )

        index.remove_from_index(sample_ids="abc", reload=False)

        assert _ids(index) == ["a", "b"]

    def test_empty_index_tolerates_a_removal(self, index):
        index.remove_from_index(sample_ids=["a"], reload=False)

        assert index.total_index_size == 0


class TestAnIndexWithNoTable:
    """Reads against an index whose adds never created a table.

    ``fbu.get_embeddings()`` returns an empty array for a collection that
    yields no embeddings, and ``similarity.py`` guards only on ``None``, so
    ``compute_similarity`` saves a brain key whose table was never written.
    Every read below reached a ``NoneType`` attribute before.
    """

    @pytest.fixture(name="tableless")
    def fixture_tableless(self, index):
        index.add_to_index(np.empty((0, 0)), [], reload=False)
        assert index.table is None
        return index

    def test_get_embeddings_is_empty(self, tableless):
        embeddings, sample_ids, label_ids = tableless.get_embeddings()

        assert embeddings.size == 0
        assert sample_ids.size == 0
        assert label_ids is None

    def test_get_embeddings_reports_the_ids_as_missing(self, tableless):
        # The rows are absent, so every ID asked for is missing -- which is
        # what `allow_missing` exists to report
        with pytest.raises(ValueError, match="do not exist in the index"):
            tableless.get_embeddings(sample_ids=["a"], allow_missing=False)

    @pytest.mark.parametrize("return_dists", [False, True])
    @pytest.mark.parametrize("patches", [False, True])
    @pytest.mark.parametrize("single", [True, False])
    def test_a_query_returns_nothing(
        self, tableless, return_dists, patches, single
    ):
        # The empty result decides on all three of these, and its shape has
        # to match what a populated query returns for each combination
        tableless._config.patches_field = "ground_truth" if patches else None
        query = (
            np.zeros(DIMS, dtype=np.float32)
            if single
            else np.zeros((2, DIMS), dtype=np.float32)
        )
        empty = [] if single else [[], []]

        with _no_view():
            got = tableless._kneighbors(
                query=query, k=3, return_dists=return_dists
            )

        expected_labels = empty if patches else None
        if return_dists:
            assert got == (empty, expected_labels, empty)
        else:
            assert got == (empty, expected_labels)

    def test_the_empty_slots_are_separate_lists(self, tableless):
        # `_set_list_values_by_id` in fiftyone-core takes all three, and one
        # list under three names would have a mutation of any show up in all
        tableless._config.patches_field = "ground_truth"

        with _no_view():
            sample_ids, label_ids, dists = tableless._kneighbors(
                query=np.zeros(DIMS, dtype=np.float32),
                k=3,
                return_dists=True,
            )

        assert sample_ids is not label_ids
        assert sample_ids is not dists
        assert label_ids is not dists

    def test_the_empty_embeddings_are_two_dimensional(self, tableless):
        # A reducer handed a (0,) array reports "Expected 2D array" from
        # somewhere unrelated to the index that produced it
        embeddings, _, _ = tableless.get_embeddings()

        assert embeddings.ndim == 2

    def test_a_query_by_id_says_the_id_is_not_there(self, tableless):
        with _no_view():
            with pytest.raises(ValueError, match="were not found in the"):
                tableless._kneighbors(query="a", k=3)

    def test_a_row_arriving_later_is_read(self, tableless):
        # The empty add is a no-op, not a terminal state
        tableless.add_to_index(
            _basis_embeddings(1), np.array(["a"]), reload=False
        )

        embeddings, sample_ids, _ = tableless.get_embeddings()

        assert embeddings.shape == (1, DIMS)
        assert list(sample_ids) == ["a"]


class TestADamagedTable:
    """A table that is listed but will not open."""

    @pytest.mark.parametrize(
        "damage",
        [
            pytest.param("delete_manifest", id="manifest_deleted"),
            pytest.param("remove_versions", id="versions_removed"),
        ],
    )
    def test_it_is_raised_rather_than_read_as_absent(
        self, populated_index, damage
    ):
        # Both of these make `open_table` say "was not found" for a table
        # whose data files are still there and whose name is still listed.
        # Reported as absent, the next add replaces it and the rows go.
        versions = os.path.join(populated_index.table.uri, "_versions")
        if damage == "delete_manifest":
            for name in os.listdir(versions):
                os.remove(os.path.join(versions, name))
        else:
            shutil.rmtree(versions)

        db = populated_index._db
        assert "test" in _table_names(db)

        # 0.37.1 raises `RuntimeError: ... exists but could not be
        # loaded`; the releases either side raise `ValueError: ... was not
        # found`. Which one matters far less than that it is raised
        with pytest.raises((ValueError, RuntimeError)):
            _open_table(db, "test")

    def test_a_name_that_is_simply_absent_is_not(self, index):
        assert _open_table(index._db, "never-written") is None


class TestBoundedReads:
    """Reads that name IDs do not read the whole table."""

    def _without_whole_table_reads(self, index):
        """Fails the test if anything reads the table back in full."""
        return mock.patch.object(
            type(index.table),
            "to_pandas",
            autospec=True,
            side_effect=AssertionError(
                "read the whole table to answer a question about some rows"
            ),
        )

    def test_a_query_by_id_reads_only_those_rows(self, basis_index):
        with self._without_whole_table_reads(basis_index):
            query = basis_index._parse_neighbors_query("b")

        assert query.shape == (DIMS,)

    def test_get_embeddings_by_id_reads_only_those_rows(self, populated_index):
        with self._without_whole_table_reads(populated_index):
            embeddings, sample_ids, _ = populated_index.get_embeddings(
                sample_ids=["a", "c"]
            )

        assert sorted(sample_ids) == ["a", "c"]
        assert embeddings.shape == (2, DIMS)

    def test_get_embeddings_for_everything_still_reads_everything(
        self, populated_index
    ):
        # The whole index is the answer here, so the whole table is the
        # right read -- the point is that naming IDs avoids it, not that
        # `to_pandas` is banned
        embeddings, sample_ids, _ = populated_index.get_embeddings()

        assert sorted(sample_ids) == ["a", "b", "c"]
        assert embeddings.shape == (3, DIMS)

    def test_a_scalar_id_is_not_split_into_characters(self, populated_index):
        embeddings, sample_ids, _ = populated_index.get_embeddings(
            sample_ids="a"
        )

        assert list(sample_ids) == ["a"]


class TestCleanup:
    """Dropping what a run leaves in the store."""

    def test_the_table_is_dropped(self, populated_index):
        populated_index.cleanup()

        assert _table_names(populated_index._db) == []
        assert populated_index._table is None

    def test_a_scratch_filter_table_is_dropped_too(self, populated_index):
        # Nothing writes this second table any more, but runs created by
        # older releases have one sitting in the store and dropping the run
        # is the occasion to take it with them
        name = populated_index.config.table_name
        populated_index._db.create_table(
            name + "_filter",
            _to_arrow_table(["a"], ["a"], _basis_embeddings(1)),
        )

        populated_index.cleanup()

        assert _table_names(populated_index._db) == []

    def test_an_index_that_named_no_table_tolerates_cleanup(
        self, populated_index
    ):
        # `_initialize` mints a name, so this is the state a run reaches by
        # failing to save one. The name is concatenated before it is looked
        # up, so an unnamed run would raise rather than find nothing to drop
        name = populated_index.config.table_name
        populated_index.config.table_name = None

        populated_index.cleanup()

        # The handle is released either way; nothing was named, so nothing
        # in the store was dropped
        assert populated_index._table is None
        assert _table_names(populated_index._db) == [name]


class TestKneighbors:
    """Querying the index."""

    def test_returns_nearest_ids_and_distances(self, basis_index):
        query = _basis_embeddings(3)[1]

        with _no_view():
            ids, label_ids, dists = basis_index._kneighbors(
                query=query, k=2, return_dists=True
            )

        assert ids[0] == "b"
        assert label_ids is None
        assert dists[0] == pytest.approx(0.0)
        assert dists == sorted(dists)

    @pytest.mark.parametrize(
        "kwargs,message",
        [
            pytest.param(
                dict(query=None), "full index neighbors", id="no_query"
            ),
            pytest.param(
                dict(query=_basis_embeddings(3)[0], reverse=True),
                "least similarity",
                id="reverse",
            ),
            pytest.param(
                dict(query=_basis_embeddings(3)[0], aggregation="max"),
                "max aggregation",
                id="unsupported_aggregation",
            ),
        ],
    )
    def test_unsupported_queries_raise(self, basis_index, kwargs, message):
        with pytest.raises(ValueError, match=message):
            basis_index._kneighbors(**kwargs)

    def test_mean_aggregation_collapses_a_stack_to_one_query(
        self, basis_index
    ):
        stack = _basis_embeddings(3)[:2]

        with _no_view():
            aggregated, _, _ = basis_index._kneighbors(
                query=stack, k=1, aggregation="mean", return_dists=True
            )
            direct, _, _ = basis_index._kneighbors(
                query=stack.mean(axis=0), k=1, return_dists=True
            )

        # One result set for the stack rather than one per row, and the
        # same one that querying its mean directly gives
        assert aggregated == direct
        assert len(aggregated) < len(stack)

    def test_a_stack_without_aggregation_queries_each_row(self, basis_index):
        queries = _basis_embeddings(3)[:2]

        with _no_view():
            ids, _, _ = basis_index._kneighbors(
                query=queries, k=1, return_dists=True
            )

        assert [group[0] for group in ids] == ["a", "b"]


class TestQueryFilters:
    """Choosing the predicates a query runs under.

    These read the predicates rather than the rows, because the rows do not
    observe the choice: an `IN` list, its complement and a batched split all
    select the same rows, and only their cost differs.
    """

    def test_the_whole_index_is_unrestricted(self, populated_index):
        with _no_view():
            assert populated_index._query_filters() == [(None, None)]

    @pytest.mark.parametrize(
        "table_ids,view_ids,expected",
        [
            pytest.param(
                ["a", "b", "c", "d"],
                ["a", "b"],
                [("id IN ('a', 'b')", 2)],
                id="a_view_names_its_ids",
            ),
            # A repeat would put its row in two batches, and the merge would
            # hand the caller that row twice
            pytest.param(
                ["a", "b", "c", "d"],
                ["a", "b", "a"],
                [("id IN ('a', 'b')", 2)],
                id="an_id_is_named_once",
            ),
            pytest.param(
                ["a", "b", "c"],
                "a",
                [("id IN ('a')", 1)],
                id="a_scalar_id_is_not_split_into_characters",
            ),
            # Distinct from `[(None, None)]`, which is the whole table: no
            # filter at all and no rows to filter are opposite answers
            pytest.param(
                ["a", "b", "c"],
                [],
                [],
                id="a_view_holding_nothing_filters_to_nothing",
            ),
            # `id NOT IN ()` is a parse error rather than the tautology it
            # looks like, and a view that happens to hold the whole index is
            # ordinary -- any view that reorders without filtering is one
            pytest.param(
                ["a", "b", "c"],
                ["a", "b", "c"],
                [(None, 3)],
                id="a_view_holding_every_row_restricts_nothing",
            ),
            pytest.param(
                ["a", "b", "c", "d"],
                ["a", "b", "c"],
                [("id NOT IN ('d')", 3)],
                id="most_of_the_table_names_the_rest",
            ),
            # The row count only predicts which list is shorter; a view
            # naming IDs the index does not hold is where it misses
            pytest.param(
                ["a", "b", "c", "d"],
                ["a", "x", "y"],
                [("id IN ('a', 'x', 'y')", 3)],
                id="a_complement_no_shorter_than_the_view_is_not_taken",
            ),
        ],
    )
    def test_what_a_view_turns_into(
        self, index, table_ids, view_ids, expected
    ):
        _filled(index, table_ids)

        with _view_of(view_ids):
            assert index._query_filters() == expected

    def test_a_patches_index_names_its_label_ids(self, index):
        # The rows are labels, so it is the label IDs a view restricts to
        index.config.patches_field = "ground_truth"
        index.add_to_index(
            _random_embeddings(4),
            np.array(["s1", "s1", "s2", "s2"]),
            label_ids=np.array(["a", "b", "c", "d"]),
            reload=False,
        )

        with _view_of(["s1"], label_ids=["a", "b"]):
            assert index._query_filters() == [("id IN ('a', 'b')", 2)]

    def test_a_long_id_list_is_split_across_predicates(
        self, index, monkeypatch
    ):
        monkeypatch.setattr(lancedb_backend, "_ID_BATCH_SIZE", 2)
        _filled(index, ["a", "b", "c", "d", "e", "f"])

        with _view_of(["a", "b", "c"]):
            assert index._query_filters() == [
                ("id IN ('a', 'b')", 2),
                ("id IN ('c')", 1),
            ]

    def test_a_view_holding_half_the_table_names_itself(self, index):
        # Half cannot have the shorter complement, so the ID scan that would
        # find out is not worth running
        _filled(index, ["a", "b", "c", "d"])

        with mock.patch.object(
            LanceDBSimilarityIndex, "_scan_ids", wraps=index._scan_ids
        ) as scan_ids, _view_of(["a", "b"]):
            assert index._query_filters() == [("id IN ('a', 'b')", 2)]

        scan_ids.assert_not_called()

    def test_the_complement_is_one_predicate_however_long(
        self, index, monkeypatch
    ):
        # `NOT IN` batches partition nothing -- each excludes only the IDs
        # it names -- so they cannot be split the way an `IN` list is
        monkeypatch.setattr(lancedb_backend, "_ID_BATCH_SIZE", 1)
        index.add_to_index(_random_embeddings(6), _row_ids(6), reload=False)

        with _view_of(list(_row_ids(6))[:4]):
            filters = index._query_filters()

        assert len(filters) == 1
        assert filters[0][0] == "id NOT IN ('id-00004', 'id-00005')"


class TestSliceFilter:
    """Restricting a search to one group slice."""

    def test_an_index_without_the_column_names_no_slice(self, basis_index):
        # Every index written before the column existed, which has to keep
        # answering rather than ask for a rebuild
        with _scoped_to(basis_index, "left"):
            assert basis_index._current_slice_name() is None

    def test_a_flattened_view_names_no_slice(self, sliced_index):
        # `group_slice` is None exactly when a stage has flattened the
        # slices, which is a search across all of them
        with _scoped_to(sliced_index, None):
            assert sliced_index._current_slice_name() is None

        with _scoped_to(sliced_index, None), _no_view():
            assert sliced_index._query_filters() == [(None, None)]

    def test_a_slice_is_a_predicate_rather_than_an_id_list(self, sliced_index):
        with _scoped_to(sliced_index, "left"), _no_view():
            assert sliced_index._query_filters() == [
                ("slice_name = 'left'", None)
            ]

    @pytest.mark.parametrize(
        "view_ids,expected",
        [
            pytest.param(["a"], [("id IN ('a')", 1)], id="named"),
            pytest.param(["a", "b"], [("id NOT IN ('c')", 2)], id="excluded"),
        ],
    )
    def test_a_view_names_no_slice_of_its_own(
        self, sliced_index, view_ids, expected
    ):
        # A view holds one slice, so its IDs already say which. Naming the
        # slice as well would say it twice out of two snapshots: the IDs
        # are cached and the slice is read live, so a dataset whose active
        # slice moves between them would ask for rows no ID names
        with _scoped_to(sliced_index, "left"), _view_of(view_ids):
            assert sliced_index._query_filters() == expected

    def test_a_slice_that_moves_under_a_view_does_not_empty_it(
        self, sliced_index
    ):
        # The rows the view names, whatever the dataset's active slice has
        # since become
        with _scoped_to(sliced_index, "right"), _view_of(["a", "b"]):
            ids, _, _ = sliced_index._kneighbors(
                query=_basis_embeddings(3)[0], k=3, return_dists=True
            )

        assert sorted(ids) == ["a", "b"]

    def test_a_quoted_slice_name_round_trips(self, index):
        with mock.patch.object(
            LanceDBSimilarityIndex,
            "_resolve_slice_names",
            return_value=["o'clock", "o'clock", "right"],
        ):
            index.add_to_index(
                _basis_embeddings(3), np.array(["a", "b", "c"]), reload=False
            )

        with _scoped_to(index, "o'clock"), _no_view():
            ids, _, _ = index._kneighbors(
                query=_basis_embeddings(3)[2], k=3, return_dists=True
            )

        assert sorted(ids) == ["a", "b"]

    def test_a_query_stays_inside_the_slice(self, sliced_index):
        # `c` is the nearest row to this query and sits in the other slice
        with _scoped_to(sliced_index, "left"), _no_view():
            ids, _, dists = sliced_index._kneighbors(
                query=_basis_embeddings(3)[2], k=3, return_dists=True
            )

        assert sorted(ids) == ["a", "b"]
        assert len(dists) == len(ids)

    def test_an_unwritten_slice_is_not_matched(self, sliced_index):
        with _scoped_to(sliced_index, "nowhere"), _no_view():
            ids, _, dists = sliced_index._kneighbors(
                query=_basis_embeddings(3)[0], k=3, return_dists=True
            )

        assert ids == []
        assert dists == []


class TestKneighborsOverAView:
    """The filtered query path.

    A view is a prefilter on the indexed table: the restriction decides
    which rows the vector search runs over, so the query keeps the table's
    index and writes nothing.
    """

    def _query_over_view(self, index, visible_ids, query, k):
        with _view_of(visible_ids):
            return index._kneighbors(query=query, k=k, return_dists=True)

    def test_restricts_results_to_the_view(self, basis_index):
        # Query nearest "a", but hide it: the answer must come from the rest
        ids, _, _ = self._query_over_view(
            basis_index, ["b", "c"], _basis_embeddings(3)[0], 3
        )

        assert "a" not in ids
        assert set(ids) == {"b", "c"}

    def test_an_empty_view_returns_nothing(self, basis_index):
        ids, label_ids, dists = self._query_over_view(
            basis_index, [], _basis_embeddings(3)[0], 3
        )

        assert ids == []
        assert label_ids is None
        assert dists == []

    def test_an_empty_view_keeps_the_shape_of_a_stack(self, basis_index):
        # Callers unpack a result per query vector, and a filter selecting
        # no rows does not excuse handing back one list for two queries
        ids, _, dists = self._query_over_view(
            basis_index, [], _basis_embeddings(3)[:2], 3
        )

        assert ids == [[], []]
        assert dists == [[], []]

    def test_a_filtered_query_writes_nothing(self, basis_index):
        # The rewrite this replaced landed a Lance version per query and
        # left a `<table>_filter` table behind: five queries measured five
        # versions and 24.0 MB on disk for a 4.8 MB view
        db = basis_index._db
        name = basis_index.config.table_name
        before = len(db.open_table(name).list_versions())

        for _ in range(5):
            self._query_over_view(
                basis_index, ["b"], _basis_embeddings(3)[0], 1
            )

        assert len(db.open_table(name).list_versions()) == before
        assert _table_names(db) == [name]

    def test_two_views_of_one_run_do_not_collide(self, tmp_path):
        # The scratch table this replaced was named per run rather than per
        # query, so a second reader's filter overwrote the first's and each
        # could be served the other's rows. Interleaving is what shows the
        # two readers are now independent
        embeddings = _basis_embeddings(4)
        writer = _unbound_index(tmp_path)
        writer.add_to_index(
            embeddings, np.array(["a", "b", "c", "d"]), reload=False
        )
        reader = _unbound_index(tmp_path)

        db = writer._db
        name = writer.config.table_name
        before = len(db.open_table(name).list_versions())
        query = embeddings[0]

        def answer(index, visible):
            ids, _, _ = self._query_over_view(index, visible, query, 4)
            return sorted(ids)

        assert answer(writer, ["b", "c"]) == ["b", "c"]
        assert answer(reader, ["a", "d"]) == ["a", "d"]
        assert answer(writer, ["b", "c"]) == ["b", "c"]

        assert _table_names(db) == [name]
        assert len(db.open_table(name).list_versions()) == before

    def test_a_view_keeps_the_tables_vector_index(self, tmp_path):
        # The per-query copy carried no index, so a view query was brute
        # force where a whole-index query was not: 2.1 ms against 51.3 ms
        # over a 90% view at 50k rows
        index = _unbound_index(tmp_path)
        index.add_to_index(
            _random_embeddings(PQ_TRAINING_ROWS),
            _row_ids(PQ_TRAINING_ROWS),
            reload=False,
        )

        with _view_of(list(_row_ids(PQ_TRAINING_ROWS))[:10]):
            plan = _plan(
                index,
                _random_embeddings(1)[0],
                where=index._query_filters()[0][0],
            )

        assert "ANNSubIndex: name=%s" % _VECTOR_INDEX_NAME in plan

    @pytest.mark.parametrize("visible", ["half", "most"])
    def test_matches_the_exhaustive_answer(
        self, tmp_path, monkeypatch, visible
    ):
        # The merge is where a wrong-results bug would hide, so the answer
        # is computed here rather than asked of the index a second way.
        # `min_index_rows` above the row count keeps the search exhaustive,
        # so the comparison is against a definite answer
        monkeypatch.setattr(lancedb_backend, "_ID_BATCH_SIZE", 7)
        rows = 60
        embeddings = _random_embeddings(rows, seed=3)
        ids = list(_row_ids(rows))

        index = _unbound_index(tmp_path, min_index_rows=rows + 1)
        index.add_to_index(embeddings, np.array(ids), reload=False)

        # Half spans several `IN` batches; most of the table inverts to a
        # single `NOT IN`, and the two have to agree
        among = ids[::2] if visible == "half" else ids[:50]
        query = _random_embeddings(1, seed=9)[0]

        with _view_of(among):
            found, _, dists = index._kneighbors(
                query=query, k=10, return_dists=True
            )

        assert found == _nearest_ids(embeddings, ids, query, 10, among=among)
        assert dists == sorted(dists)

    def test_matches_the_exhaustive_answer_across_real_batches(self, tmp_path):
        # At the shipped batch size rather than a monkeypatched one, so a
        # split that only ever happens on a large table is exercised at the
        # size it happens. A view has to be under half the table to stay on
        # the `IN` path, so the table is four batches wide to make room for
        # a two-batch view
        rows = 4 * _ID_BATCH_SIZE + 1
        embeddings = _random_embeddings(rows, seed=5)
        ids = list(_row_ids(rows))

        index = _unbound_index(tmp_path, min_index_rows=rows + 1)
        index.add_to_index(embeddings, np.array(ids), reload=False)

        among = ids[: 2 * _ID_BATCH_SIZE]
        query = _random_embeddings(1, seed=11)[0]

        with _view_of(among):
            assert len(index._query_filters()) == 2

            found, _, _ = index._kneighbors(
                query=query, k=10, return_dists=True
            )

        assert found == _nearest_ids(embeddings, ids, query, 10, among=among)

    def test_a_view_holding_every_row_still_answers(self, basis_index):
        with _view_of(["a", "b", "c"]):
            ids, _, _ = basis_index._kneighbors(
                query=_basis_embeddings(3)[1], k=3, return_dists=True
            )

        assert ids[0] == "b"
        assert sorted(ids) == ["a", "b", "c"]

    def test_an_emptied_index_answers_a_filtered_query(self, populated_index):
        populated_index.remove_from_index(
            sample_ids=["a", "b", "c"], reload=False
        )

        with _view_of(["a"]):
            ids, _, dists = populated_index._kneighbors(
                query=_random_embeddings(1)[0], k=3, return_dists=True
            )

        assert ids == []
        assert dists == []

    def test_a_query_leaves_the_vectors_on_the_server(self, basis_index):
        # A result row carries its whole embedding otherwise, and `k=None`
        # asks for every row -- which is the whole-table read this replaced.
        # Spelled out rather than compared against `_QUERY_COLUMNS`, which
        # would agree with whatever that constant came to say
        results = basis_index._search(_basis_embeddings(3)[0], "l2", 3)

        assert results.to_arrow().column_names == [
            "id",
            "sample_id",
            "_distance",
        ]

    def _wide_index(self, tmp_path):
        """An index holding more rows than a search returns in order."""
        index = _unbound_index(tmp_path, min_index_rows=UNSORTED_ROWS + 1)
        index.add_to_index(
            _random_embeddings(UNSORTED_ROWS, seed=4),
            _row_ids(UNSORTED_ROWS),
            reload=False,
        )
        return index

    def test_a_large_result_arrives_unsorted(self, tmp_path):
        # Guards the premise: were a search to sort its own output, the
        # test below would be passing for a reason that is not the merge
        index = self._wide_index(tmp_path)

        dists = index._search(
            _random_embeddings(1, seed=6)[0], "l2", UNSORTED_ROWS
        ).to_pandas()["_distance"]

        assert list(dists) != sorted(dists)

    def test_a_large_result_is_sorted_before_it_is_returned(self, tmp_path):
        # `sort_by_similarity` documents `k=None` as sorting every sample,
        # so a result the caller reads as ranked has to be one
        index = self._wide_index(tmp_path)

        with _no_view():
            ids, _, dists = index._kneighbors(
                query=_random_embeddings(1, seed=6)[0],
                k=UNSORTED_ROWS,
                return_dists=True,
            )

        assert len(ids) == UNSORTED_ROWS
        assert dists == sorted(dists)

    def test_an_index_emptied_of_its_rows_answers_with_nothing(
        self, populated_index
    ):
        # `k=None` becomes `k=0` there, which Lance rejects as a missing
        # limit rather than answering with no rows
        populated_index.remove_from_index(
            sample_ids=["a", "b", "c"], reload=False
        )

        with _no_view(), mock.patch.object(
            LanceDBSimilarityIndex,
            "index_size",
            new_callable=mock.PropertyMock,
            return_value=0,
        ):
            ids, _, dists = populated_index._kneighbors(
                query=_random_embeddings(1)[0], k=None, return_dists=True
            )

        assert ids == []
        assert dists == []

    #: Rows the scattered fixture below holds
    SCATTERED_ROWS = 500

    def _scattered_index(self, tmp_path):
        """An index whose partitions a single probe barely reaches.

        Returns ``(index, embeddings, ids)``. An indexed query over it
        comes back short, which is what makes the rescan inside
        :meth:`_search_exactly_k` fire rather than sit unused.
        """
        embeddings = _random_embeddings(self.SCATTERED_ROWS, seed=8)
        ids = list(_row_ids(self.SCATTERED_ROWS))

        index = _unbound_index(
            tmp_path,
            nprobes=1,
            index_type="ivf_flat",
            index_params={"num_partitions": 64},
            refine_factor=None,
        )
        index.add_to_index(embeddings, np.array(ids), reload=False)

        return index, embeddings, ids

    @builds_indexes
    def test_matches_the_exhaustive_answer_under_a_vector_index(
        self, tmp_path, monkeypatch
    ):
        # The other merge tests keep the table under `min_index_rows`, so
        # every search in them is exhaustive and the per-batch rescan never
        # fires.
        #
        # An indexed query is approximate, so the equality below holds
        # because the rescan makes it exact, not in spite of the index: the
        # scattered partitions send every batch short of `k`. Recall where
        # the rescan does not fire is a measurement rather than an
        # invariant -- 0.992 to 1.000 across filter selectivity on real
        # CLIP-512 embeddings -- so it is not asserted anywhere
        monkeypatch.setattr(lancedb_backend, "_ID_BATCH_SIZE", 40)
        index, embeddings, ids = self._scattered_index(tmp_path)

        among = ids[:200]
        query = _random_embeddings(1, seed=12)[0]

        with mock.patch.object(
            LanceDBSimilarityIndex, "_search", wraps=index._search
        ) as search, _view_of(among):
            filters = index._query_filters()
            assert len(filters) == 5

            found, _, _ = index._kneighbors(
                query=query, k=25, return_dists=True
            )

        # Guards the premise: an indexed query that answered in full would
        # make this a recall assertion rather than an exactness one
        assert search.call_count > len(filters)
        assert found == _nearest_ids(embeddings, ids, query, 25, among=among)

    @builds_indexes
    def test_a_rescanned_query_keeps_its_filter(self, tmp_path):
        # The rescan is a second search, and one that dropped the
        # restriction would answer from the whole table: measured at 39% of
        # the returned rows coming from outside the view
        index, _, ids = self._scattered_index(tmp_path)
        among = ids[:300]

        with mock.patch.object(
            LanceDBSimilarityIndex, "_search", wraps=index._search
        ) as search, _view_of(among):
            found, _, _ = index._kneighbors(
                query=_random_embeddings(1, seed=12)[0],
                k=len(among),
                return_dists=True,
            )

        # Guards the premise: a query that answered in full would leave the
        # rescan untested rather than merely unused
        assert search.call_count == 2
        assert search.call_args_list[1].kwargs["bypass_index"] is True
        assert (
            search.call_args_list[1].kwargs["where"]
            == search.call_args_list[0].kwargs["where"]
        )
        assert set(found) <= set(among)

    @builds_indexes
    def test_a_view_narrower_than_k_still_answers_in_full(self, tmp_path):
        # The shortfall `_search_exactly_k` exists to catch, on the path
        # that matters: `sort_by_similarity` documents `k=None` as sorting
        # every sample, so a short result drops samples from a view
        index, _, ids = self._scattered_index(tmp_path)
        among = ids[:200]

        with _view_of(among):
            found, _, _ = index._kneighbors(
                query=_random_embeddings(1, seed=12)[0],
                k=len(among) + 100,
                return_dists=True,
            )

        assert sorted(found) == sorted(among)

    def test_a_query_that_answers_in_full_is_not_rescanned(self, tmp_path):
        # The rescan costs a whole-table scan -- 417 ms at 1M rows and 512
        # dimensions -- so it has to stay the exception. Built with the
        # configured index rather than the scattered one above, whose whole
        # purpose is to come back short
        rows = self.SCATTERED_ROWS
        index = _unbound_index(tmp_path)
        index.add_to_index(
            _random_embeddings(rows, seed=8), _row_ids(rows), reload=False
        )

        with mock.patch.object(
            LanceDBSimilarityIndex, "_search", wraps=index._search
        ) as search, _view_of(list(_row_ids(rows))[:200]):
            found, _, _ = index._kneighbors(
                query=_random_embeddings(1, seed=12)[0], k=5, return_dists=True
            )

        assert len(found) == 5
        assert search.call_count == 1

    def test_a_stack_of_queries_is_filtered_the_same_way(self, basis_index):
        ids, _, _ = self._query_over_view(
            basis_index, ["b", "c"], _basis_embeddings(3)[:2], 1
        )

        assert ids == [["b"], ["b"]]

    def test_a_short_view_is_not_rescanned(self, basis_index):
        # A view narrower than `k` is short for the honest reason, and
        # confirming it against the whole table would double every query
        # the App makes
        with mock.patch.object(
            LanceDBSimilarityIndex, "_search", wraps=basis_index._search
        ) as search:
            ids, _, _ = self._query_over_view(
                basis_index, ["b"], _basis_embeddings(3)[0], 10
            )

        assert ids == ["b"]
        assert search.call_count == 1


class TestSliceWritePath:
    """Deciding whether a write carries slice names."""

    def _grouped(self):
        return mock.patch.object(
            LanceDBSimilarityIndex, "_group_field", return_value="group"
        )

    def test_an_index_with_no_collection_writes_no_slice_column(
        self, basis_index
    ):
        assert basis_index._resolve_slice_names(["a", "b", "c"]) is None
        assert _SLICE_NAME_COLUMN not in basis_index.table.schema.names

    def test_a_flat_dataset_writes_no_slice_column(self, basis_index):
        # A flat dataset has no group field, which is the test -- and a
        # different one from having no collection at all
        basis_index._samples = mock.MagicMock(
            _root_dataset=mock.MagicMock(group_field=None)
        )

        assert basis_index._resolve_slice_names(["a", "b", "c"]) is None
        assert _SLICE_NAME_COLUMN not in basis_index.table.schema.names

    def test_a_table_written_without_the_column_is_not_given_one(
        self, basis_index
    ):
        # Lance rejects a merge whose source names a column the target
        # lacks, so an index written before the column existed can only be
        # written the way it was created
        with self._grouped():
            assert basis_index._resolve_slice_names(["a"]) is None

        with self._grouped():
            basis_index.add_to_index(
                _basis_embeddings(1), np.array(["d"]), reload=False
            )

        assert _SLICE_NAME_COLUMN not in basis_index.table.schema.names
        assert _ids(basis_index) == ["a", "b", "c", "d"]

    def test_a_table_carries_the_column_when_the_rows_do(self, sliced_index):
        assert _SLICE_NAME_COLUMN in sliced_index.table.schema.names
        assert sliced_index.table.to_arrow()[
            _SLICE_NAME_COLUMN
        ].to_pylist() == ["left", "left", "right"]

    def _grouped_samples(self, index, by_id):
        """Binds the index to a grouped collection returning ``by_id``.

        The lookup answers in its own order, as an aggregation does, which
        is what makes keying the result by ID rather than by position
        observable.
        """
        view = mock.MagicMock()
        view.select.return_value.values.return_value = (
            list(by_id.keys()),
            list(by_id.values()),
        )
        dataset = mock.MagicMock(group_field="group")
        dataset.select_group_slices.return_value = view
        index._samples = mock.MagicMock(_root_dataset=dataset)

    def test_slice_names_follow_the_ids_not_the_lookup_order(self, index):
        # An aggregation returns its rows in an order of its own, so the
        # names have to be matched back to the IDs that asked for them.
        # Positional labeling agrees by accident whenever the two coincide
        self._grouped_samples(
            index, {"s1": "left", "s2": "right", "s3": "left"}
        )

        assert index._resolve_slice_names(["s3", "s1", "s2"]) == [
            "left",
            "left",
            "right",
        ]

    def test_a_sample_named_once_per_label_is_labeled_each_time(self, index):
        # A patches index writes one row per label, so the same sample is
        # named once per label it carries and every row needs its own value
        self._grouped_samples(index, {"s1": "left", "s2": "right"})

        assert index._resolve_slice_names(["s1", "s1", "s2"]) == [
            "left",
            "left",
            "right",
        ]

    def test_a_later_add_keeps_filling_the_column(self, sliced_index):
        with mock.patch.object(
            LanceDBSimilarityIndex,
            "_resolve_slice_names",
            return_value=["right"],
        ):
            sliced_index.add_to_index(
                _basis_embeddings(1), np.array(["d"]), reload=False
            )

        rows = sliced_index.table.to_arrow().to_pylist()
        assert {row["id"]: row[_SLICE_NAME_COLUMN] for row in rows} == {
            "a": "left",
            "b": "left",
            "c": "right",
            "d": "right",
        }

    def test_a_row_outside_the_flattened_view_is_left_unlabeled(self, index):
        # Typed rather than inferred, or a batch of nothing but None would
        # be an untyped null column that will not merge into a string one
        with mock.patch.object(
            LanceDBSimilarityIndex,
            "_resolve_slice_names",
            return_value=[None, None, None],
        ):
            index.add_to_index(
                _basis_embeddings(3), np.array(["a", "b", "c"]), reload=False
            )

        assert index.table.schema.field(_SLICE_NAME_COLUMN).type == pa.string()

        # SQL leaves a null out of an equality, which is what keeps an
        # unlabeled row from answering for every slice
        with _scoped_to(index, "left"), _no_view():
            ids, _, _ = index._kneighbors(
                query=_basis_embeddings(3)[0], k=3, return_dists=True
            )

        assert ids == []


class TestReload:
    """The refresh that follows a write."""

    def test_picks_up_another_writer(self, populated_index, tmp_path):
        # A table handle is pinned to the version it was opened at, so a row
        # committed through a second connection is invisible until reload
        other = lancedb.connect(str(tmp_path)).open_table("test")
        other.add(_to_arrow_table(["d"], ["d"], _random_embeddings(1, seed=1)))
        assert populated_index.total_index_size == 3

        populated_index.reload()

        assert populated_index.total_index_size == 4

    def test_reload_survives_an_index_with_no_table(self, index):
        index.reload()

        index.reload_mock.assert_called_once()

    def test_add_reloads_by_default(self, index):
        index.add_to_index(_random_embeddings(1), np.array(["a"]))

        index.reload_mock.assert_called_once()

    def test_remove_reloads_by_default(self, populated_index):
        populated_index.remove_from_index(sample_ids=["a"])

        populated_index.reload_mock.assert_called_once()

    def test_empty_add_reloads_by_default(self, index):
        index.add_to_index(
            np.empty((0, 0), dtype=np.float64), np.array([], dtype="<U24")
        )

        index.reload_mock.assert_called_once()

    def test_reload_false_is_honored(self, index):
        index.add_to_index(
            _random_embeddings(1), np.array(["a"]), reload=False
        )

        index.reload_mock.assert_not_called()


@pytest.fixture(name="empty_backend_config")
def fixture_empty_backend_config():
    """Isolates tests from whatever the ambient brain config carries."""
    with mock.patch.object(fob.brain_config, "similarity_backends", {}):
        yield


class TestStorageOptions:
    """The storage options that carry a credential to the object store."""

    def test_defaults_to_none(self):
        config = LanceDBSimilarityConfig()
        assert config.storage_options is None

    @pytest.mark.parametrize(
        "supply",
        [
            pytest.param(
                lambda options: LanceDBSimilarityConfig(
                    storage_options=options
                ),
                id="constructor",
            ),
            pytest.param(
                lambda options: _assign(
                    LanceDBSimilarityConfig(), "storage_options", options
                ),
                id="setter",
            ),
        ],
    )
    def test_lands_on_the_private_attribute(self, supply):
        # The base config setattrs any unrecognized keyword, so reading the
        # value back proves nothing. What the field adds is that it is stored
        # privately, which is what keeps it out of `serialize()`
        options = {"aws_access_key_id": "key", "aws_region": "us-east-1"}
        config = supply(options)

        assert config._storage_options == options
        assert "storage_options" not in vars(config)

    @pytest.mark.usefixtures("empty_backend_config")
    def test_load_credentials_supplies_them(self):
        config = LanceDBSimilarityConfig()
        config.load_credentials(storage_options={"aws_session_token": "token"})
        assert config.storage_options == {"aws_session_token": "token"}

    @pytest.mark.usefixtures("empty_backend_config")
    def test_refreshing_only_the_uri_keeps_the_options(self):
        # ``None`` means "not supplied", so a caller that only refreshes the
        # URI must not blank out the credential it is still using
        config = LanceDBSimilarityConfig(storage_options={"key": "value"})
        config.load_credentials(uri="/tmp/lancedb")
        assert config.storage_options == {"key": "value"}

    def test_assigned_options_win_over_the_backend(self):
        # The backend supplies options only where the config carries none.
        # Routing an assigned value through `_load_parameters` would let the
        # deployment's credential replace the one the caller handed in.
        assigned = {"aws_session_token": "assigned"}
        with mock.patch.object(
            fob.brain_config,
            "similarity_backends",
            {"lancedb": {"storage_options": {"google_service_account": "sa"}}},
        ):
            config = LanceDBSimilarityConfig(storage_options=assigned)
            config.load_credentials()

        assert config.storage_options == assigned

    def test_load_credentials_falls_back_to_the_brain_config(self):
        # The backend entry in ``~/.fiftyone/brain_config.json`` supplies the
        # options when the call site passes none. Note that file is plaintext
        # and `fiftyone brain config` prints it, so it is not a secret store
        options = {"google_service_account": "/var/run/sa.json"}
        with mock.patch.object(
            fob.brain_config,
            "similarity_backends",
            {"lancedb": {"storage_options": options}},
        ):
            config = LanceDBSimilarityConfig()
            config.load_credentials()

        assert config.storage_options == options

    def test_the_backend_config_dict_is_not_shared(self):
        # `_load_parameters` hands over the global dict itself. Aliasing it
        # would let one index's credential refresh rewrite every other
        # index's credential in the process
        options = {"aws_session_token": "first"}
        with mock.patch.object(
            fob.brain_config,
            "similarity_backends",
            {"lancedb": {"storage_options": options}},
        ):
            config = LanceDBSimilarityConfig()
            config.load_credentials()
            other = LanceDBSimilarityConfig()
            other.load_credentials()

            config.storage_options["aws_session_token"] = "refreshed"

            assert other.storage_options == {"aws_session_token": "first"}
            assert options == {"aws_session_token": "first"}


class TestSerialization:
    """What reaches the database when a brain run is saved."""

    def test_credentials_are_not_serialized(self):
        # Stored privately so a vended credential never lands in the
        # dataset's brain document
        config = LanceDBSimilarityConfig(
            storage_options={"aws_secret_access_key": "SENSITIVE"}
        )
        serialized = config.serialize()

        assert "storage_options" not in serialized
        assert "SENSITIVE" not in str(serialized)

    def test_the_uri_is_serialized(self):
        # A location rather than a credential, and no more revealing than
        # the sample filepaths in the same database. Recorded so two runs
        # may keep their tables in different stores, and so a run keeps the
        # one it was built in whatever a later reader is configured with
        config = LanceDBSimilarityConfig(uri="s3://bucket/vectors")

        assert config.serialize()["uri"] == "s3://bucket/vectors"

    def test_a_run_naming_no_store_serializes_none(self):
        # The fallback is resolved when the store is opened rather than
        # assigned here, so an unnamed run cannot acquire whichever store
        # happened to be configured the first time something read it
        config = LanceDBSimilarityConfig(table_name="a-table")

        assert config.serialize()["uri"] is None


class TestResolveUri:
    """Which store a run opens."""

    @pytest.fixture(name="backend")
    def fixture_backend(self):
        backends = fob.brain_config.similarity_backends
        original = backends.get("lancedb", {}).copy()
        entry = backends.setdefault("lancedb", {})
        entry.pop("uri", None)

        yield entry

        backends["lancedb"] = original

    def test_the_run_s_own_uri_wins(self, backend):
        backend["uri"] = "s3://configured/vectors"
        config = LanceDBSimilarityConfig(uri="s3://recorded/vectors")

        config.load_credentials()

        assert config.resolve_uri() == "s3://recorded/vectors"

    def test_an_unnamed_run_takes_the_configured_store(self, backend):
        backend["uri"] = "s3://configured/vectors"
        config = LanceDBSimilarityConfig()

        config.load_credentials()

        assert config.resolve_uri() == "s3://configured/vectors"
        # Resolved, not assigned: a later save must not record it
        assert config.uri is None

    @pytest.mark.usefixtures("backend")
    def test_neither_falls_back_to_a_local_directory(self):
        config = LanceDBSimilarityConfig()

        config.load_credentials()

        assert config.resolve_uri() == lancedb_backend.DEFAULT_URI

    def test_an_explicit_argument_wins_over_both(self, backend):
        backend["uri"] = "s3://configured/vectors"
        config = LanceDBSimilarityConfig(uri="s3://recorded/vectors")

        config.load_credentials(uri="s3://explicit/vectors")

        assert config.resolve_uri() == "s3://explicit/vectors"

    def test_the_rest_of_the_config_still_serializes(self):
        config = LanceDBSimilarityConfig(
            table_name="a-table",
            metric="euclidean",
            storage_options={"aws_access_key_id": "key"},
        )
        serialized = config.serialize()

        assert serialized["table_name"] == "a-table"
        assert serialized["metric"] == "euclidean"
        assert "storage_options" not in serialized


def _assign(obj, name, value):
    setattr(obj, name, value)
    return obj
