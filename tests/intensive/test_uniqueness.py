"""
Uniqueness tests.

| Copyright 2017-2022, Voxel51, Inc.
| `voxel51.com <https://voxel51.com/>`_
|
"""
import os
import unittest

import eta.core.storage as etas
import eta.core.utils as etau

import fiftyone as fo
import fiftyone.brain as fob
import fiftyone.zoo as foz


def test_uniqueness():
    _run_uniqueness()


def test_uniqueness_torch_model():
    model = foz.load_zoo_model("inception-v3-imagenet-torch")
    _run_uniqueness(model=model, batch_size=16)


def test_uniqueness_tf_model():
    model = foz.load_zoo_model("resnet-v2-50-imagenet-tf1")
    _run_uniqueness(model=model, batch_size=16)


def test_roi_uniqueness():
    _run_uniqueness(roi_field="ground_truth")


def test_roi_uniqueness_torch_model():
    model = foz.load_zoo_model("inception-v3-imagenet-torch")
    _run_uniqueness(roi_field="ground_truth", model=model, batch_size=16)


def test_roi_uniqueness_tf_model():
    model = foz.load_zoo_model("resnet-v2-50-imagenet-tf1")
    _run_uniqueness(roi_field="ground_truth", model=model, batch_size=16)


def _run_uniqueness(roi_field=None, model=None, batch_size=None):
    dataset = foz.load_zoo_dataset(
        "quickstart", dataset_name=fo.get_default_dataset_name()
    )
    dataset.delete_sample_field("uniqueness")

    view = dataset.take(50)
    num_samples = len(view)

    # Custom Torch model
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
