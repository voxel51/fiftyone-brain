"""
Redaction tests.

| Copyright 2017-2025, Voxel51, Inc.
| `voxel51.com <https://voxel51.com/>`_
|
"""
import unittest
import os

import fiftyone as fo
import fiftyone.brain as fob
import fiftyone.zoo as foz
from fiftyone import ViewField as F


def test_create_redaction_fields():
    dataset = foz.load_zoo_dataset("quickstart").clone()
    test_view = dataset.take(10, seed=42)

    brain_key = "test_create_redaction_fields"
    results = fob.create_redaction(
        test_view,
        label_field="ground_truth",
        label_classes="person,car",
        redaction_type="bounding_box",
        redaction_method="stack_blur",
        redaction_field=brain_key,
    )
    assert brain_key in dataset.list_brain_runs()
    redacted_image_path = test_view.first()[brain_key + "_filepath"]
    assert redacted_image_path is not None
    assert os.path.exists(redacted_image_path)

    dataset.delete_brain_run(brain_key)


def test_recreate_redaction_fields():
    dataset = foz.load_zoo_dataset("quickstart").clone()
    test_view = dataset.take(5, seed=42)

    _ = fob.create_redaction(
        test_view,
        label_field="ground_truth",
        label_classes="person,car",
        redaction_type="bounding_box",
        redaction_method="stack_blur",
    )
    expected_brain_key = "redacted_ground_truth_bounding_box_stack_blur"

    for sample in test_view.iter_samples():
        redacted_image_path = sample[expected_brain_key + "_filepath"]
        if os.path.exists(redacted_image_path):
            os.remove(redacted_image_path)

    _ = fob.create_redaction(
        test_view,
        label_field="ground_truth",
        label_classes="person,car",
        redaction_type="bounding_box",
        redaction_method="stack_blur",
        force_recreate=False,
        # still recreates since the files are missing
    )
    for sample in test_view.iter_samples():
        redacted_image_path = sample[expected_brain_key + "_filepath"]
        assert redacted_image_path is not None
        assert os.path.exists(redacted_image_path)
    modification_time = os.path.getmtime(
        test_view.first()[expected_brain_key + "_filepath"]
    )

    _ = fob.create_redaction(
        test_view,
        label_field="ground_truth",
        label_classes="person,car",
        redaction_type="bounding_box",
        redaction_method="stack_blur",
        force_recreate=True,
    )
    new_modification_time = os.path.getmtime(
        test_view.first()[expected_brain_key + "_filepath"]
    )
    assert new_modification_time > modification_time

    dataset.delete_brain_run(expected_brain_key)


def test_create_redaction_samples():
    dataset = foz.load_zoo_dataset("quickstart").clone()
    test_view = dataset.take(10, seed=42)

    brain_key = "test_create_redaction_samples"
    results = fob.create_redaction(
        test_view,
        label_field="ground_truth",
        label_classes="person,car",
        redaction_type="bounding_box",
        redaction_method="mask",
        redaction_field=brain_key,
    )
    redacted_dataset = results.generate_redacted_dataset(
        name="test_10_redacted_dataset", overwrite=True
    )
    for redacted_sample in redacted_dataset.iter_samples():
        assert brain_key in redacted_sample.tags
        assert redacted_sample["filepath"] is not None
        assert os.path.exists(redacted_sample["filepath"])

    assert (
        len(
            redacted_dataset.match(
                F("ground_truth.detections.label").contains(["person", "car"])
            )
        )
        == 0
    )

    dataset.delete_brain_run(brain_key)


if __name__ == "__main__":
    fo.config.show_progress_bars = True
    unittest.main(verbosity=2)
