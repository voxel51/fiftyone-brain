"""
VisualizationResults.use_view() tests, focused on the root-dataset fast
path: loading results against a root dataset must not run the id
aggregation that view-scoped calls require, and must be behaviorally
identical to the slow path for pristine datasets.

All tests use synthetic manually-provided points (no models, no zoo
downloads), so this file is fast and hermetic.

| Copyright 2017-2026, Voxel51, Inc.
| `voxel51.com <https://voxel51.com/>`_
|
"""

from unittest import mock

import numpy as np

import fiftyone as fo
import fiftyone.brain as fob
import fiftyone.brain.internal.core.utils as fbu


def _make_samples_run(n=20):
    dataset = fo.Dataset()
    dataset.add_samples(
        [
            fo.Sample(filepath="/tmp/img%d.png" % i, cluster="c%d" % (i % 3))
            for i in range(n)
        ]
    )
    points = np.stack(
        [np.arange(n, dtype=float), -np.arange(n, dtype=float)], axis=1
    )
    fob.compute_visualization(dataset, points=points, brain_key="viz")
    return dataset, n


def _make_patches_run(num_samples=4, labels_per_sample=2):
    dataset = fo.Dataset()
    samples = []
    for i in range(num_samples):
        detections = [
            fo.Detection(label="d%d" % j, bounding_box=[0.1, 0.1, 0.2, 0.2])
            for j in range(labels_per_sample)
        ]
        samples.append(
            fo.Sample(
                filepath="/tmp/img%d.png" % i,
                ground_truth=fo.Detections(detections=detections),
            )
        )

    dataset.add_samples(samples)

    n = num_samples * labels_per_sample
    points = np.zeros((n, 2))
    fob.compute_visualization(
        dataset,
        points=points,
        patches_field="ground_truth",
        brain_key="viz",
    )
    return dataset, n


def test_load_skips_root_dataset_aggregation():
    # The optimization itself: loading results against a root dataset
    # must not aggregate ids (filter_ids runs a full values("id") pull)
    dataset, n = _make_samples_run()
    try:
        with mock.patch.object(
            fbu, "filter_ids", wraps=fbu.filter_ids
        ) as filter_ids:
            results = dataset.load_brain_results("viz", cache=False)

        assert filter_ids.call_count == 0
        assert results.index_size == n
        assert results._curr_keep_inds is None
        assert len(results._curr_sample_ids) == n
    finally:
        dataset.delete()


def test_root_dataset_matches_slow_path():
    # For a pristine dataset the fast path must be indistinguishable
    # from the view-scoped slow path
    dataset, _ = _make_samples_run()
    try:
        results = dataset.load_brain_results("viz", cache=False)

        results.use_view(dataset.view())  # DatasetView: slow path
        slow = (
            results._curr_points.copy(),
            list(results._curr_sample_ids),
            results._curr_keep_inds,
            results._curr_good_inds,
        )

        results.use_view(dataset)  # Dataset: fast path
        assert np.array_equal(results._curr_points, slow[0])
        assert list(results._curr_sample_ids) == slow[1]
        assert results._curr_keep_inds is None and slow[2] is None
        assert results._curr_good_inds is None and slow[3] is None
    finally:
        dataset.delete()


def test_views_still_filter():
    # The fast path must not hijack view-scoped calls
    dataset, n = _make_samples_run()
    try:
        results = dataset.load_brain_results("viz", cache=False)

        view = dataset.take(5, seed=51)
        results.use_view(view)
        assert results.index_size == 5
        assert len(results._curr_keep_inds) == 5
        assert set(results._curr_sample_ids) == set(view.values("id"))

        results.use_view(dataset)
        assert results.index_size == n
    finally:
        dataset.delete()


def test_clear_view_restores_full_index():
    # clear_view() re-enters use_view with the root dataset, so it rides
    # the fast path too
    dataset, n = _make_samples_run()
    try:
        results = dataset.load_brain_results("viz", cache=False)

        results.use_view(dataset.take(5, seed=51))
        assert results.index_size == 5

        results.clear_view()
        assert results.index_size == n
        assert results._curr_keep_inds is None
    finally:
        dataset.delete()


def test_patches_run_fast_path():
    dataset, n = _make_patches_run()
    try:
        with mock.patch.object(
            fbu, "filter_ids", wraps=fbu.filter_ids
        ) as filter_ids:
            results = dataset.load_brain_results("viz", cache=False)

        assert filter_ids.call_count == 0
        assert results.index_size == n
        assert len(results._curr_label_ids) == n

        # Label-scoped views still take the slow path and filter
        view = dataset.filter_labels(
            "ground_truth", fo.ViewField("label") == "d0"
        )
        results.use_view(view)
        assert results.index_size == n // 2
    finally:
        dataset.delete()


def test_root_dataset_is_a_snapshot():
    # Pinned deliberately: the fast path treats the run as a snapshot —
    # samples deleted since compute are NOT pruned on root-dataset calls
    # (they never held points for samples ADDED since compute, either).
    # View-scoped calls still prune, and refreshing the run is the
    # supported way to sync a visualization with dataset changes
    dataset, n = _make_samples_run()
    try:
        dataset.delete_samples([dataset.first().id])

        results = dataset.load_brain_results("viz", cache=False)
        assert results.index_size == n  # snapshot retained

        results.use_view(dataset.view())
        assert results.index_size == n - 1  # slow path prunes

        results.use_view(dataset)
        assert results.index_size == n
    finally:
        dataset.delete()
