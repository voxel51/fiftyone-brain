"""
fiftyone.brain.internal.core.utils.filter_ids() tests.

These pin the exact output contract — values, ordering, dtypes, and the
None conventions — so the membership computation can be reimplemented
(e.g. vectorized) with confidence that behavior is unchanged. All tests
use small synthetic datasets and manual points; no models, no zoo.

| Copyright 2017-2026, Voxel51, Inc.
| `voxel51.com <https://voxel51.com/>`_
|
"""

import numpy as np
import pytest

import fiftyone as fo
import fiftyone.brain.internal.core.utils as fbu


def _make_dataset(n=12):
    dataset = fo.Dataset()
    dataset.add_samples(
        [fo.Sample(filepath="/tmp/img%02d.png" % i) for i in range(n)]
    )
    return dataset


def _make_patches_dataset(num_samples=4, labels_per_sample=2):
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
    return dataset


def test_identical_collection_early_exits():
    dataset = _make_dataset()
    try:
        ids = dataset.values("id")
        index_ids = np.array(ids)

        sample_ids, label_ids, keep_inds, good_inds = fbu.filter_ids(
            dataset, index_ids, None
        )

        assert list(sample_ids) == ids
        assert label_ids is None
        # The early-exit convention: None means "the full index, as is"
        assert keep_inds is None
        assert good_inds is None
    finally:
        dataset.delete()


def test_no_index_returns_collection_ids():
    dataset = _make_dataset()
    try:
        sample_ids, label_ids, keep_inds, good_inds = fbu.filter_ids(
            dataset, None, None
        )

        assert list(sample_ids) == dataset.values("id")
        assert label_ids is None and keep_inds is None and good_inds is None
    finally:
        dataset.delete()


def test_subset_view_keep_inds_are_index_positions():
    dataset = _make_dataset()
    try:
        index_ids = np.array(dataset.values("id"))

        view = dataset.skip(3).limit(5)
        sample_ids, _, keep_inds, good_inds = fbu.filter_ids(
            view, index_ids, None
        )

        assert list(sample_ids) == view.values("id")
        assert keep_inds.dtype == np.int64
        assert list(keep_inds) == [3, 4, 5, 6, 7]
        assert good_inds is None
    finally:
        dataset.delete()


def test_keep_inds_follow_collection_order():
    # keep_inds is a reordering map: index positions listed in the
    # order the collection iterates, not index order
    dataset = _make_dataset()
    try:
        index_ids = np.array(dataset.values("id"))

        view = dataset.sort_by("filepath", reverse=True)
        _, _, keep_inds, _ = fbu.filter_ids(view, index_ids, None)

        n = len(index_ids)
        assert list(keep_inds) == list(range(n - 1, -1, -1))
    finally:
        dataset.delete()


def test_collection_ids_missing_from_index_are_pruned():
    # Samples added after compute: present in the collection, absent
    # from the index -> pruned from the returned ids via good_inds
    dataset = _make_dataset()
    try:
        all_ids = dataset.values("id")
        index_ids = np.array(all_ids[:8])

        sample_ids, _, keep_inds, good_inds = fbu.filter_ids(
            dataset, index_ids, None
        )

        assert list(keep_inds) == list(range(8))
        assert good_inds.dtype == bool
        assert list(good_inds) == [True] * 8 + [False] * 4
        assert list(sample_ids) == all_ids[:8]
    finally:
        dataset.delete()


def test_index_ids_missing_from_collection_are_dropped():
    # Samples deleted after compute (or a view): index entries with no
    # collection counterpart simply do not appear in keep_inds; there
    # are no bad collection entries, so good_inds stays None
    dataset = _make_dataset()
    try:
        index_ids = np.array(dataset.values("id"))

        view = dataset.skip(4)
        sample_ids, _, keep_inds, good_inds = fbu.filter_ids(
            view, index_ids, None
        )

        assert list(keep_inds) == list(range(4, len(index_ids)))
        assert good_inds is None
        assert list(sample_ids) == view.values("id")
    finally:
        dataset.delete()


def test_allow_missing_false_raises_both_directions():
    dataset = _make_dataset()
    try:
        all_ids = dataset.values("id")

        # Index entries missing from the collection
        with pytest.raises(ValueError):
            fbu.filter_ids(
                dataset.skip(4),
                np.array(all_ids),
                None,
                allow_missing=False,
            )

        # Collection entries missing from the index
        with pytest.raises(ValueError):
            fbu.filter_ids(
                dataset, np.array(all_ids[:8]), None, allow_missing=False
            )
    finally:
        dataset.delete()


def test_patches_field_filters_by_label_ids():
    dataset = _make_patches_dataset()
    try:
        # Wire order: labels flattened in sample order
        full_sample_ids, full_label_ids = fbu._get_patch_ids(
            dataset, "ground_truth"
        )
        index_label_ids = np.array(full_label_ids)

        view = dataset.filter_labels(
            "ground_truth", fo.ViewField("label") == "d0"
        )
        sample_ids, label_ids, keep_inds, good_inds = fbu.filter_ids(
            view,
            np.array(full_sample_ids),
            index_label_ids,
            patches_field="ground_truth",
        )

        # One "d0" label per sample = every even index position
        assert list(keep_inds) == [0, 2, 4, 6]
        assert good_inds is None
        assert list(label_ids) == list(index_label_ids[keep_inds])
        assert len(sample_ids) == len(label_ids)
    finally:
        dataset.delete()


def test_patches_view_maps_rows_to_sample_ids():
    # A to_patches() view hits the _is_patches branch: collection ids
    # are the (repeating) owning sample ids per patch row
    dataset = _make_patches_dataset()
    try:
        index_sample_ids = np.array(dataset.values("id"))

        patches = dataset.to_patches("ground_truth")
        sample_ids, _, keep_inds, good_inds = fbu.filter_ids(
            patches, index_sample_ids, None
        )

        assert list(sample_ids) == patches.values("sample_id")
        # Two patches per sample -> each index position appears twice,
        # in collection (patch-row) order
        assert list(keep_inds) == [0, 0, 1, 1, 2, 2, 3, 3]
        assert good_inds is None
    finally:
        dataset.delete()
