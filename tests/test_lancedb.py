"""
Unit tests for the LanceDB similarity backend.

LanceDB is embedded, so these exercise a real table under ``tmp_path`` and
need no service. They are skipped where ``lancedb`` is not installed, which
includes CI, since it is an optional dependency. The tests that build an index
over a dataset live in ``tests/intensive/test_similarity.py``.

| Copyright 2017-2026, Voxel51, Inc.
| `voxel51.com <https://voxel51.com/>`_
|
"""
import logging
import os
from unittest import mock

import numpy as np
import pytest

lancedb = pytest.importorskip("lancedb")
pa = pytest.importorskip("pyarrow")

from fiftyone.brain.similarity import SimilarityIndex  # noqa: E402
from fiftyone.brain.internal.core.lancedb import (  # noqa: E402 — after skip
    LanceDBSimilarityConfig,
    LanceDBSimilarityIndex,
    _ID_BATCH_SIZE,
    _id_predicate,
    _to_arrow_table,
)

DIMS = 8

# Enough IDs to span more than one predicate batch
BATCHED_ROWS = 2 * _ID_BATCH_SIZE + 1


def _random_embeddings(num_rows, seed=0):
    rng = np.random.default_rng(seed)
    return rng.random((num_rows, DIMS), dtype=np.float32)


def _constant_embeddings(num_rows, fill):
    return np.full((num_rows, DIMS), fill, dtype=np.float32)


def _basis_embeddings(num_rows):
    """Unit basis vectors, so nearest-neighbor order is unambiguous."""
    return np.eye(num_rows, DIMS, dtype=np.float32)


def _ids(index):
    return sorted(index.table.to_arrow()["id"].to_pylist())


def _sample_ids(index):
    return sorted(index.table.to_arrow()["sample_id"].to_pylist())


def _row_for(index, _id):
    rows = index.table.to_arrow().to_pylist()
    return next(row for row in rows if row["id"] == _id)


def _indexed_columns(index):
    return [config.columns for config in index.table.list_indices()]


def _data_files(index):
    path = os.path.join(index.table.uri, "data")
    return sorted(os.listdir(path)) if os.path.isdir(path) else []


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
    index = LanceDBSimilarityIndex.__new__(LanceDBSimilarityIndex)
    index._config = LanceDBSimilarityConfig(
        table_name="test", uri=str(tmp_path), metric="euclidean"
    )
    index._initialize()

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


@pytest.fixture(name="batched_index")
def fixture_batched_index(index):
    """An index holding more rows than fit in one predicate batch."""
    ids = np.array(["id-%05d" % i for i in range(BATCHED_ROWS)])
    index.add_to_index(_random_embeddings(BATCHED_ROWS), ids, reload=False)
    return index, list(ids)


class TestIdPredicate:
    """Building the SQL predicate that matches a set of IDs."""

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

    def test_float64_embeddings_are_cast(self):
        embeddings = np.ones((2, DIMS), dtype=np.float64)

        table = _to_arrow_table(["a", "b"], ["s1", "s2"], embeddings)

        assert table["vector"].type.value_type == pa.float32()
        assert table["vector"].to_pylist()[0] == [1.0] * DIMS


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

    def test_allow_existing_false_raises(self, populated_index):
        with pytest.raises(ValueError, match="already exist"):
            populated_index.add_to_index(
                _random_embeddings(1),
                np.array(["a"]),
                allow_existing=False,
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
        other = LanceDBSimilarityIndex.__new__(LanceDBSimilarityIndex)
        other._config = LanceDBSimilarityConfig(
            table_name="test", uri=str(tmp_path), metric="euclidean"
        )
        other._initialize()
        assert index.table is None and other.table is None

        index.add_to_index(
            _random_embeddings(1), np.array(["a"]), reload=False
        )
        other.add_to_index(
            _random_embeddings(1, seed=1), np.array(["b"]), reload=False
        )

        assert _ids(other) == ["a", "b"]

    def test_second_add_does_not_recreate_the_table(self, populated_index):
        # Recreating the table is what made add cost scale with table size
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

    def test_add_does_not_rewrite_existing_data_files(self, populated_index):
        before = _data_files(populated_index)
        assert before

        populated_index.add_to_index(
            _random_embeddings(2, seed=1), np.array(["d", "e"]), reload=False
        )

        assert set(before).issubset(_data_files(populated_index))


class TestIdIndex:
    """The scalar index on the ``id`` column."""

    def test_created_on_first_add(self, populated_index):
        assert ["id"] in _indexed_columns(populated_index)

    def test_added_to_a_table_that_lacks_it(self, index):
        index.add_to_index(
            _random_embeddings(2), np.array(["a", "b"]), reload=False
        )
        index.table.drop_index(index.table.list_indices()[0].name)
        assert not index.table.list_indices()

        index.add_to_index(
            _random_embeddings(1, seed=1), np.array(["c"]), reload=False
        )

        assert ["id"] in _indexed_columns(index)


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


class TestKneighbors:
    """Querying the index."""

    @pytest.fixture(name="basis_index")
    def fixture_basis_index(self, index):
        index.add_to_index(
            _basis_embeddings(3), np.array(["a", "b", "c"]), reload=False
        )
        return index

    def test_returns_nearest_ids_and_distances(self, basis_index):
        query = _basis_embeddings(3)[1]

        with mock.patch.object(
            LanceDBSimilarityIndex,
            "has_view",
            new_callable=mock.PropertyMock,
            return_value=False,
        ):
            ids, label_ids, dists = basis_index._kneighbors(
                query=query, k=2, return_dists=True
            )

        assert ids[0] == "b"
        assert label_ids is None
        assert dists[0] == pytest.approx(0.0)
        assert dists == sorted(dists)


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
