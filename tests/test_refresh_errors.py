"""
Refresh error-handling and lazy-training-size tests.

These pin two contracts around the stored reducer:

-   ``get_new_ids(..., include_training_size=False)`` must never load the
    reducer, so interactive callers can count new samples without paying
    the unpickle (which imports ``umap`` and can take ~10s cold)
-   failures while *applying* the reducer (hydration or transform) must
    raise an actionable recompute error rather than leak internals such as
    numba's ``AssertionError`` — the failure mode when a reducer pickled
    under one Python/umap/numba version is used under another, where the
    unpickle itself succeeds

All tests use small synthetic datasets and stub reducers; no models, no
zoo, and ``umap`` itself is never imported.

| Copyright 2017-2026, Voxel51, Inc.
| `voxel51.com <https://voxel51.com/>`_
|
"""

from unittest import mock

import numpy as np
import pytest

import fiftyone as fo
import fiftyone.brain.visualization as fbv


def _make_umap_run(n=8, n_new=3, dim=4):
    """A sample-level UMAP-method run over the first ``n`` samples of an
    ``n + n_new``-sample dataset, built directly (no umap import)."""
    dataset = fo.Dataset()
    dataset.add_samples(
        [fo.Sample(filepath="/tmp/img%02d.png" % i) for i in range(n + n_new)]
    )

    config = fbv.UMAPVisualizationConfig(embeddings_field="emb")
    points = np.zeros((n, 2))
    results = fbv.VisualizationResults(dataset.limit(n), config, "viz", points)
    return dataset, results


class _BoomReducer:
    """A hydrated-looking reducer whose transform fails the way a
    cross-environment numba failure does."""

    def __init__(self, n_train=8, dim=4):
        self._raw_data = np.zeros((n_train, dim))
        self.embedding_ = np.zeros((n_train, 2))

    def transform(self, X):
        raise AssertionError("key already in dictionary: 56")


def test_get_new_ids_opt_out_never_loads_reducer():
    dataset, results = _make_umap_run()
    try:
        results._reducer_blob = "sentinel-blob"
        with mock.patch.object(
            fbv, "_unpickle_reducer", side_effect=AssertionError("loaded!")
        ) as unpickle:
            new_ids, train_size = results.get_new_ids(
                dataset, include_training_size=False
            )

        assert unpickle.call_count == 0
        assert results._reducer is None
        assert train_size is None
        assert len(new_ids) == 3
    finally:
        dataset.delete()


def test_get_new_ids_default_behavior_unchanged():
    dataset, results = _make_umap_run()
    try:
        # unreadable blob: the default path must degrade to None, not raise
        results._reducer_blob = "not-a-real-blob"
        new_ids, train_size = results.get_new_ids(dataset)

        assert train_size is None
        assert len(new_ids) == 3
    finally:
        dataset.delete()


def test_hydration_failure_raises_actionable_error():
    dataset, results = _make_umap_run()
    try:
        reducer = _BoomReducer()
        reducer._raw_data = None  # not hydrated -> hydration path runs
        results._reducer = reducer

        with mock.patch.object(
            fbv,
            "_hydrate_umap_reducer",
            side_effect=AssertionError("key already in dictionary: 56"),
        ), mock.patch.object(
            results, "_load_training_embeddings", return_value=np.zeros((8, 4))
        ):
            with pytest.raises(ValueError, match="different environment"):
                results._prepare_reducer(dataset)
    finally:
        dataset.delete()


def test_transform_failure_raises_actionable_error():
    dataset, results = _make_umap_run()
    try:
        results._reducer = _BoomReducer()

        new_view = dataset.skip(8)
        embeddings = np.random.rand(3, 4)

        with pytest.raises(ValueError, match="different environment"):
            results.add_samples(new_view, embeddings=embeddings)
    finally:
        dataset.delete()
