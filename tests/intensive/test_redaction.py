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
import fiftyone.core.storage as fos
from fiftyone import ViewField as F


def test_create_redaction_fields():
    dataset = foz.load_zoo_dataset("quickstart").clone()
    test_view = dataset.take(10, seed=42)

    brain_key = "test_create_redaction_fields"
    results = fob.create_redaction(
        test_view,
        label_field="ground_truth",
        label_classes=["person", "car"],
        redaction_type="bounding_box",
        redaction_method="stack_blur",
        redaction_field=brain_key,
    )
    assert brain_key in dataset.list_brain_runs()
    redacted_image_path = test_view.first()[brain_key + "_filepath"]
    assert redacted_image_path is not None
    assert fos.exists(redacted_image_path)
    nontrivial_samples = test_view.match(
        F("ground_truth.detections.label").contains(["person", "car"])
    )
    if len(nontrivial_samples) > 0:
        assert (
            nontrivial_samples.first()[brain_key + "_filepath"]
            != nontrivial_samples.first()["filepath"]
        )

    dataset.delete_brain_run(brain_key)


def test_create_redaction_samples_video():
    dataset = foz.load_zoo_dataset("quickstart-video").clone()
    test_view = dataset.take(1, seed=42)

    brain_key = "test_create_redaction_samples_video"
    results = fob.create_redaction(
        test_view,
        label_field="frames.detections",
        label_classes=["person", "car"],
        redaction_type="bounding_box",
        redaction_method="mask",
        redaction_field=brain_key,
    )
    redacted_dataset = results.generate_redacted_dataset(
        name="test_1_redacted_dataset", overwrite=True
    )
    for redacted_sample in redacted_dataset.iter_samples():
        assert redacted_sample["filepath"] is not None
        assert fos.exists(redacted_sample["filepath"])

    assert (
        len(
            redacted_dataset.match(
                F("frames.detections.detections.label").contains(
                    ["person", "car"]
                )
            )
        )
        == 0
    )

    dataset.delete_brain_run(brain_key)


def test_create_redaction_samples():
    dataset = foz.load_zoo_dataset("quickstart").clone()
    test_view = dataset.take(10, seed=42)

    brain_key = "test_create_redaction_samples"
    results = fob.create_redaction(
        test_view,
        label_field="ground_truth",
        label_classes=["person", "car"],
        redaction_type="bounding_box",
        redaction_method="mask",
        redaction_field=brain_key,
    )
    redacted_dataset = results.generate_redacted_dataset(
        name="test_10_redacted_dataset", overwrite=True
    )
    for redacted_sample in redacted_dataset.iter_samples():
        assert redacted_sample["filepath"] is not None
        assert fos.exists(redacted_sample["filepath"])

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
