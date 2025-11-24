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


def test_create_redaction_fields():
    dataset = foz.load_zoo_dataset("quickstart").clone()
    test_view = dataset.take(10, seed=42)

    fob.create_redaction(
        test_view,
        label_field="ground_truth",
        label_classes="person,car",
        redaction_type="bounding_box",
        redaction_method="stack_blur",
    )
    expected_brain_key = (
        "redacted_ground_truth_person_car_bounding_box_stack_blur"
    )
    assert expected_brain_key in dataset.list_brain_runs()
    redacted_image_path = test_view.first()[expected_brain_key + "_filepath"]
    assert redacted_image_path is not None
    assert os.path.exists(redacted_image_path)

    dataset.delete_brain_runs()


def test_recreate_redaction_fields():
    dataset = foz.load_zoo_dataset("quickstart").clone()
    test_view = dataset.take(5, seed=42)

    fob.create_redaction(
        test_view,
        label_field="ground_truth",
        label_classes="person,car",
        redaction_type="bounding_box",
        redaction_method="stack_blur",
    )
    expected_brain_key = (
        "redacted_ground_truth_person_car_bounding_box_stack_blur"
    )

    for sample in test_view.iter_samples():
        redacted_image_path = sample[expected_brain_key + "_filepath"]
        if os.path.exists(redacted_image_path):
            os.remove(redacted_image_path)

    fob.create_redaction(
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

    fob.create_redaction(
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

    dataset.delete_brain_runs()


def test_create_redaction_samples():
    dataset = foz.load_zoo_dataset("quickstart").clone()
    test_view = dataset.take(10, seed=42)

    fob.create_redaction(
        test_view,
        label_field="ground_truth",
        label_classes="person,car",
        redaction_type="bounding_box",
        redaction_method="mask",
        create_as_new_sample=True,
    )
    expected_brain_key = "redacted_ground_truth_person_car_bounding_box_mask"
    for sample in test_view.iter_samples():
        if expected_brain_key in sample.tags:
            continue
        redacted_sample_id = sample["redacted_sample_ids"][expected_brain_key]
        assert redacted_sample_id is not None
        assert redacted_sample_id in dataset.values("id")
        redacted_sample = dataset[redacted_sample_id]
        assert redacted_sample["filepath"] is not None
        assert os.path.exists(redacted_sample["filepath"])
        assert redacted_sample["original_sample_id"] is not None
        assert redacted_sample["original_sample_id"] == sample.id

    dataset.delete_brain_runs()


if __name__ == "__main__":
    fo.config.show_progress_bars = True
    unittest.main(verbosity=2)
