"""
Within-slice search on a grouped dataset, over a real collection.

A grouped dataset's index carries the slice each row belongs to, so that a
search restricted to one slice is a single predicate at any selectivity
rather than an enumerated list of every ID in it. Reaching that needs a
dataset -- the slice is read off the group field, and which slice a view is
scoped to is a property of the view -- which is what separates this module
from ``test_lancedb.py``, where the index is built without one.

| Copyright 2017-2026, Voxel51, Inc.
| `voxel51.com <https://voxel51.com/>`_
|
"""
from unittest import mock

import numpy as np
import pytest

lancedb = pytest.importorskip("lancedb")

# Every import below is E402 by construction: it has to follow the
# `importorskip` above, which decides whether this module runs at all
import fiftyone as fo  # noqa: E402
import fiftyone.brain as fob  # noqa: E402
from fiftyone import ViewField as F  # noqa: E402
from fiftyone.brain.internal.core.lancedb import (  # noqa: E402
    LanceDBSimilarityIndex,
    _SLICE_NAME_COLUMN,
)

DIMS = 8

#: Group slices the fixtures build, mapped to the extension each holds.
#: The media types differ because a grouped dataset's ordinarily do, and a
#: lookup that spans the slices has to say so -- `select_group_slices()`
#: refuses a dataset holding more than one media type unless it is asked to
#: allow them
SLICES = {"left": "png", "right": "pcd"}

#: Groups per fixture dataset, so each slice holds this many samples
GROUPS = 6


def _embeddings(num_rows):
    """Unit basis vectors, so nearest-neighbor order is unambiguous."""
    return np.eye(num_rows, DIMS, dtype=np.float32)


def _grouped_dataset():
    """A grouped dataset of :data:`GROUPS` groups over :data:`SLICES`."""
    dataset = fo.Dataset()
    dataset.add_group_field("group", default=next(iter(SLICES)))

    samples = []
    for idx in range(GROUPS):
        group = fo.Group()
        for name, ext in SLICES.items():
            samples.append(
                fo.Sample(
                    filepath="/tmp/%s-%d.%s" % (name, idx, ext),
                    group=group.element(name),
                    idx=idx,
                )
            )

    dataset.add_samples(samples, progress=False)
    return dataset


def _flattened(dataset):
    """Every slice of a grouped dataset, which is what an index spans."""
    return dataset.select_group_slices(_allow_mixed=True)


def _index(samples, uri):
    """Builds a LanceDB index over the given collection.

    Embeddings are supplied rather than computed: what is under test is
    which rows a query reaches, and a model would only decide which of them
    is nearest.
    """
    ids = samples.values("id")
    return fob.compute_similarity(
        samples,
        embeddings=_embeddings(len(ids)),
        brain_key="sim",
        backend="lancedb",
        uri=str(uri),
        metric="euclidean",
        min_index_rows=0,
        progress=False,
    )


def _slices_of(dataset, sample_ids):
    """The slice each of the given samples belongs to."""
    return _flattened(dataset).select(sample_ids).values("group.name")


def _add_late_group(dataset):
    """Adds one group after the index was built, and returns its IDs."""
    group = fo.Group()
    return dataset.add_samples(
        [
            fo.Sample(
                filepath="/tmp/%s-late.%s" % (name, ext),
                group=group.element(name),
                idx=GROUPS,
            )
            for name, ext in SLICES.items()
        ],
        progress=False,
    )


@pytest.fixture(name="grouped")
def fixture_grouped(tmp_path):
    """Yields ``(index, dataset)`` for a grouped dataset's index."""
    dataset = _grouped_dataset()
    index = _index(_flattened(dataset), tmp_path)
    try:
        yield index, dataset
    finally:
        index.cleanup()
        dataset.delete()


@pytest.fixture(name="flat")
def fixture_flat(tmp_path):
    """Yields ``(index, dataset)`` for an ungrouped dataset's index."""
    dataset = fo.Dataset()
    dataset.add_samples(
        [fo.Sample(filepath="/tmp/%d.png" % i) for i in range(3)],
        progress=False,
    )
    index = _index(dataset, tmp_path)
    try:
        yield index, dataset
    finally:
        index.cleanup()
        dataset.delete()


@pytest.fixture(name="legacy")
def fixture_legacy(tmp_path):
    """Yields ``(index, dataset)`` for an index written before the column.

    Built by withholding the group field from the write path, which is what
    an older release did by never having looked for one. The table it
    leaves behind is the three-column one those releases wrote.
    """
    dataset = _grouped_dataset()
    with mock.patch.object(
        LanceDBSimilarityIndex, "_group_field", return_value=None
    ):
        index = _index(_flattened(dataset), tmp_path)

    try:
        yield index, dataset
    finally:
        index.cleanup()
        dataset.delete()


class TestSliceColumn:
    """What a grouped dataset's write path records."""

    def test_the_column_is_written(self, grouped):
        index, _ = grouped

        rows = index.table.to_arrow().to_pylist()

        assert _SLICE_NAME_COLUMN in index.table.schema.names
        assert len(rows) == GROUPS * len(SLICES)
        assert sorted(row[_SLICE_NAME_COLUMN] for row in rows) == sorted(
            [name for name in SLICES for _ in range(GROUPS)]
        )

    def test_every_row_names_its_own_sample_s_slice(self, grouped):
        index, dataset = grouped

        rows = index.table.to_arrow().to_pylist()
        expected = dict(zip(*_flattened(dataset).values(["id", "group.name"])))

        assert {row["id"]: row[_SLICE_NAME_COLUMN] for row in rows} == expected

    def test_a_flat_dataset_has_no_slice_column(self, flat):
        index, dataset = flat

        assert _SLICE_NAME_COLUMN not in index.table.schema.names
        assert index._resolve_slice_names(dataset.values("id")) is None

    def test_a_later_add_carries_the_slice_too(self, grouped):
        index, dataset = grouped
        added = _add_late_group(dataset)

        index.add_to_index(_embeddings(len(added)), np.array(added))

        rows = {
            row["id"]: row[_SLICE_NAME_COLUMN]
            for row in index.table.to_arrow().to_pylist()
        }

        assert [rows[_id] for _id in added] == list(SLICES)


class TestWithinSliceSearch:
    """Restricting a search to one slice of a grouped dataset."""

    def test_the_slice_is_a_predicate_rather_than_an_id_list(self, grouped):
        index, dataset = grouped

        # The dataset shows one slice at a time, so this is the ordinary
        # query the App makes against a grouped dataset
        index.use_view(dataset)

        assert index.has_view is False
        assert index._query_filters() == [("slice_name = 'left'", None)]

    def test_results_stay_inside_the_slice(self, grouped):
        index, dataset = grouped
        index.use_view(dataset)

        ids, _, _ = index._kneighbors(
            query=_embeddings(1)[0], k=GROUPS * len(SLICES), return_dists=True
        )

        assert len(ids) == GROUPS
        assert set(_slices_of(dataset, ids)) == {"left"}

    def test_the_other_slice_answers_for_itself(self, grouped):
        index, dataset = grouped
        dataset.group_slice = "right"
        index.use_view(dataset)

        assert index._query_filters() == [("slice_name = 'right'", None)]

        ids, _ = index._kneighbors(query=_embeddings(1)[0], k=GROUPS)

        assert set(_slices_of(dataset, ids)) == {"right"}

    def test_a_flattened_view_searches_every_slice(self, grouped):
        index, dataset = grouped
        index.use_view(_flattened(dataset))

        assert index._query_filters() == [(None, None)]

        ids, _ = index._kneighbors(
            query=_embeddings(1)[0], k=GROUPS * len(SLICES)
        )

        assert set(_slices_of(dataset, ids)) == set(SLICES)

    def test_a_filtered_view_names_its_ids_and_not_its_slice(self, grouped):
        # The view's IDs are already confined to its slice, so naming the
        # slice as well would restate it out of a second snapshot
        index, dataset = grouped
        view = dataset.match(F("idx") < 2)
        index.use_view(view)

        predicate, max_rows = index._query_filters()[0]

        assert predicate.startswith("id ")
        assert _SLICE_NAME_COLUMN not in predicate
        assert max_rows == 2

        ids, _ = index._kneighbors(query=_embeddings(1)[0], k=GROUPS)

        assert sorted(ids) == sorted(view.values("id"))

    def test_an_active_slice_moving_does_not_empty_a_view(self, grouped):
        # `group_slice` tracks its dataset live while the view's IDs are
        # cached, so reading both would disagree and match no rows at all
        index, dataset = grouped
        view = dataset.match(F("idx") < 2)
        index.use_view(view)

        expected = sorted(view.values("id"))
        assert index.index_size == 2

        dataset.group_slice = "right"
        ids, _ = index._kneighbors(query=_embeddings(1)[0], k=GROUPS)

        assert sorted(ids) == expected

    def test_a_filtered_query_writes_nothing(self, grouped):
        index, dataset = grouped
        name = index.config.table_name
        before = len(index._db.open_table(name).list_versions())

        index.use_view(dataset.match(F("idx") < 3))
        for _ in range(5):
            index._kneighbors(query=_embeddings(1)[0], k=2)

        assert len(index._db.open_table(name).list_versions()) == before
        assert name + "_filter" not in index._db.table_names()

    def test_sort_by_similarity_stays_inside_the_slice(self, grouped):
        _, dataset = grouped
        query = dataset.first().id

        found = dataset.sort_by_similarity(query, k=3, brain_key="sim")

        assert len(found) == 3
        assert set(_slices_of(dataset, found.values("id"))) == {"left"}


class TestAnIndexWrittenWithoutTheColumn:
    """What a run built before the slice column existed still does."""

    def test_it_has_no_slice_column(self, legacy):
        index, _ = legacy

        assert _SLICE_NAME_COLUMN not in index.table.schema.names

    def test_a_filtered_query_falls_back_to_the_ids(self, legacy):
        index, dataset = legacy
        view = dataset.match(F("idx") < 2)
        index.use_view(view)

        predicate, _ = index._query_filters()[0]

        assert predicate.startswith("id ")
        assert _SLICE_NAME_COLUMN not in predicate

        ids, _ = index._kneighbors(query=_embeddings(1)[0], k=GROUPS)

        assert sorted(ids) == sorted(view.values("id"))

    def test_a_later_add_is_not_given_the_column(self, legacy):
        # Lance rejects a merge whose source names a column the target
        # lacks, so an add that started carrying slices would fail rather
        # than upgrade the table
        index, dataset = legacy
        added = _add_late_group(dataset)

        index.add_to_index(_embeddings(len(added)), np.array(added))

        assert _SLICE_NAME_COLUMN not in index.table.schema.names
        assert index.total_index_size == (GROUPS + 1) * len(SLICES)
