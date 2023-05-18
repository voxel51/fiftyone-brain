"""
Uniqueness tests.

| Copyright 2017-2023, Voxel51, Inc.
| `voxel51.com <https://voxel51.com/>`_
|
"""
import unittest

import fiftyone as fo
import fiftyone.brain as fob
import fiftyone.zoo as foz


def test_uniqueness():
    _run_uniqueness()


def test_uniqueness_torch():
    model = foz.load_zoo_model("inception-v3-imagenet-torch")
    _run_uniqueness(model=model, batch_size=16)


def test_uniqueness_tf():
    model = foz.load_zoo_model("resnet-v2-50-imagenet-tf1")
    _run_uniqueness(model=model, batch_size=16)


def test_uniqueness_missing():
    dataset = fo.Dataset()
    dataset.add_samples(
        [
            fo.Sample(filepath="non-existent1.png"),
            fo.Sample(filepath="non-existent2.png"),
            fo.Sample(filepath="non-existent3.png"),
            fo.Sample(filepath="non-existent4.png"),
        ]
    )

    fob.compute_uniqueness(dataset, batch_size=1)

    view = dataset.exists("uniqueness")

    assert dataset.has_field("uniqueness")
    assert len(view) == 0


def test_roi_uniqueness():
    _run_uniqueness(roi_field="ground_truth")


def test_roi_uniqueness_torch():
    model = foz.load_zoo_model("inception-v3-imagenet-torch")
    _run_uniqueness(roi_field="ground_truth", model=model, batch_size=16)


def test_roi_uniqueness_tf():
    model = foz.load_zoo_model("resnet-v2-50-imagenet-tf1")
    _run_uniqueness(roi_field="ground_truth", model=model, batch_size=16)


def test_roi_uniqueness_missing():
    dataset = fo.Dataset()
    dataset.add_samples(
        [
            fo.Sample(filepath="non-existent1.png"),
            fo.Sample(filepath="non-existent2.png"),
            fo.Sample(filepath="non-existent3.png"),
            fo.Sample(filepath="non-existent4.png"),
        ]
    )

    for sample in dataset:
        sample["ground_truth"] = fo.Detections(
            detections=[fo.Detection(bounding_box=[0.1, 0.1, 0.8, 0.8])]
        )
        sample.save()

    fob.compute_uniqueness(dataset, roi_field="ground_truth", batch_size=1)

    view = dataset.exists("uniqueness")

    assert dataset.has_field("uniqueness")
    assert len(view) == 0


def _run_uniqueness(roi_field=None, model=None, batch_size=None):
    dataset = foz.load_zoo_dataset(
        "quickstart", dataset_name=fo.get_default_dataset_name()
    )
    dataset.delete_sample_field("uniqueness")

    view = dataset.take(50)
    num_samples = len(view)

    fob.compute_uniqueness(
        view, roi_field=roi_field, model=model, batch_size=batch_size
    )

    num_uniqueness = dataset.count("uniqueness")
    assert num_uniqueness == num_samples

    bounds = dataset.bounds("uniqueness")

    assert bounds[0] >= 0
    assert bounds[1] <= 1


if __name__ == "__main__":
    fo.config.show_progress_bars = True
    unittest.main(verbosity=2)
