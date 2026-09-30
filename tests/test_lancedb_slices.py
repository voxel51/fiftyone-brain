"""
Grouped datasets over a real collection.

A grouped dataset shows one slice at a time while its index spans every slice
it was built from. An unfiltered query on it is unrestricted: a caller that
wants fewer rows restricts the index to a view, and callers such as
``compute_uniqueness`` query a loaded index expecting every row. Nor is the
slice the index holds reliably the caller's: a ``SortBySimilarity`` from
before voxel51/fiftyone#8582 skips ``use_view`` when two views differ only in
their slice. These run those paths end to end, through the stage and the IDs
a real dataset reports, so they need a database.

| Copyright 2017-2026, Voxel51, Inc.
| `voxel51.com <https://voxel51.com/>`_
|
"""
import numpy as np
import pytest

pytest.importorskip("lancedb")

# Every import below is E402 by construction: it has to follow the
# `importorskip` above, which decides whether this module runs at all
import fiftyone as fo  # noqa: E402
import fiftyone.brain as fob  # noqa: E402
from fiftyone import ViewField as F  # noqa: E402
from fiftyone.brain.internal.core.lancedb import _table_names  # noqa: E402

DIMS = 8

#: Group slices the fixtures build, mapped to the extension each holds. Mixed
#: media, as a grouped dataset's slices ordinarily are, which is why
#: `_flattened` has to pass `_allow_mixed=True`
SLICES = {"left": "png", "right": "pcd"}

#: Groups per fixture dataset, so each slice holds this many samples
GROUPS = 6

#: Rows the grouped fixture's index holds
ROWS = GROUPS * len(SLICES)


def _embeddings(num_rows):
    """Seeded random vectors, so that both slices' rows crowd the nearest
    neighbors of any query and a search confined to one slice shows.
    """
    rng = np.random.default_rng(0)
    return rng.random((num_rows, DIMS), dtype=np.float32)


def _query():
    """The first indexed row's vector: `_embeddings` reseeds on every call,
    so a one-row call returns it.
    """
    return _embeddings(1)[0]


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
    """Every slice of a grouped dataset, which is what its index spans."""
    return dataset.select_group_slices(_allow_mixed=True)


def _index(samples, uri):
    """Builds a LanceDB index over the given collection.

    Embeddings are supplied rather than computed: what is under test is
    which rows a query reaches, and a model would only decide which of them
    is nearest. The table stays under ``min_index_rows``, so every search is
    exhaustive and the expected answers are exact.
    """
    return fob.compute_similarity(
        samples,
        embeddings=_embeddings(len(samples)),
        brain_key="sim",
        backend="lancedb",
        uri=str(uri),
        metric="euclidean",
        min_index_rows=len(samples) + 1,
        progress=False,
    )


def _slices_of(dataset, sample_ids):
    """The slice each of the given samples belongs to."""
    return _flattened(dataset).select(sample_ids).values("group.name")


def _ids_in(dataset, slice_name):
    """The IDs of one slice's samples."""
    return dataset.select_group_slices(slice_name).values("id")


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


class TestUnfilteredGroupedQuery:
    """A query on a grouped dataset itself, which shows one slice."""

    def test_it_searches_every_slice(self, grouped):
        index, dataset = grouped
        index.use_view(dataset)

        assert index.has_view is False
        assert index._query_filters() == [(None, None)]

        embeddings = _embeddings(ROWS)
        ids = _flattened(dataset).values("id")
        query = embeddings[0]
        order = np.argsort(
            ((embeddings - query) ** 2).sum(axis=1), kind="stable"
        )

        found, _ = index._kneighbors(query=query, k=ROWS)

        assert found == [ids[i] for i in order]

    def _assert_shown_slice(self, dataset, found):
        """Asserts ``found`` is not empty and holds only the right slice."""
        found_ids = found.values("id")
        assert found_ids
        assert set(_slices_of(dataset, found_ids)) == {"right"}

    def test_the_dataset_switched_between_queries(self, grouped):
        # A stage from before voxel51/fiftyone#8582 compares the views
        # without their slices, so the second query runs against the index as
        # the first left it
        _, dataset = grouped
        left, right = _ids_in(dataset, "left"), _ids_in(dataset, "right")
        dataset.sort_by_similarity(left[0], k=GROUPS, brain_key="sim")

        dataset.group_slice = "right"
        found = dataset.sort_by_similarity(right[0], k=GROUPS, brain_key="sim")

        self._assert_shown_slice(dataset, found)

    def test_a_view_pinned_to_another_slice(self, grouped):
        # What the App's group filter builds: the dataset on its default
        # slice, the view pinned to another
        _, dataset = grouped
        view = dataset.view()
        view.group_slice = "right"

        found = view.sort_by_similarity(
            _ids_in(dataset, "right")[0], k=GROUPS, brain_key="sim"
        )

        self._assert_shown_slice(dataset, found)

    def test_a_results_object_reused_across_a_switch(self, grouped):
        # A results object keeps the view it was loaded against, and reading
        # its size caches that view's IDs
        _, dataset = grouped
        results = dataset.load_brain_results("sim")
        assert results.index_size == GROUPS

        dataset.group_slice = "right"
        found = results.sort_by_similarity(
            _ids_in(dataset, "right")[0], k=GROUPS
        )

        self._assert_shown_slice(dataset, found)


class TestUniqueness:
    """`compute_uniqueness` through the index, which queries it unbound."""

    def test_it_matches_the_exact_answer_over_every_slice(self, grouped):
        _, dataset = grouped
        flat = _flattened(dataset)

        fob.compute_uniqueness(
            flat, uniqueness_field="u", similarity_index="sim", progress=False
        )

        # `compute_uniqueness` weights the three nearest others by
        # .6/.3/.1 over squared L2, which is Lance's distance, and
        # normalizes by the largest
        embeddings = _embeddings(ROWS)
        dists = ((embeddings[:, None] - embeddings[None]) ** 2).sum(axis=-1)
        nearest = np.sort(dists, axis=1)[:, 1:4]
        expected = np.mean(nearest * [0.6, 0.3, 0.1], axis=1)
        expected /= expected.max()

        np.testing.assert_allclose(flat.values("u"), expected, atol=1e-5)


class TestViews:
    """Views of a grouped dataset, which restrict a query by their IDs."""

    def test_a_flattened_view_searches_every_slice(self, grouped):
        index, dataset = grouped
        index.use_view(_flattened(dataset))

        assert index._query_filters() == [(None, None)]

        ids, _ = index._kneighbors(query=_query(), k=ROWS)

        assert set(_slices_of(dataset, ids)) == set(SLICES)

    def test_a_slice_selected_by_a_stage_is_restricted_by_its_ids(
        self, grouped
    ):
        index, dataset = grouped
        index.use_view(dataset.select_group_slices("right"))

        ids, _ = index._kneighbors(query=_query(), k=ROWS)

        assert sorted(ids) == sorted(_ids_in(dataset, "right"))

    def test_a_filtered_view_is_restricted_by_its_own_ids(self, grouped):
        index, dataset = grouped
        view = dataset.match(F("idx") < 2)
        index.use_view(view)

        ids, _ = index._kneighbors(query=_query(), k=GROUPS)

        assert sorted(ids) == sorted(view.values("id"))

    def test_a_flat_dataset_is_unrestricted(self, flat):
        index, dataset = flat
        index.use_view(dataset)

        assert index._query_filters() == [(None, None)]


class TestWrites:
    """A query leaves the store as it found it."""

    def test_a_filtered_query_writes_nothing(self, grouped):
        index, dataset = grouped
        name = index.config.table_name
        before = len(index._db.open_table(name).list_versions())

        index.use_view(dataset.match(F("idx") < 3))
        for _ in range(5):
            index._kneighbors(query=_query(), k=2)

        assert len(index._db.open_table(name).list_versions()) == before
        assert _table_names(index._db) == [name]
