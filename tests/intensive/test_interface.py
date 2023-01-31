"""
Brain interface tests.

| Copyright 2017-2023, Voxel51, Inc.
| `voxel51.com <https://voxel51.com/>`_
|
"""
import unittest

import fiftyone as fo
import fiftyone.brain as fob
import fiftyone.zoo as foz


def test_uniqueness():
    dataset = foz.load_zoo_dataset("quickstart").clone()

    fob.compute_uniqueness(dataset)
    print(dataset.list_brain_runs())
    print(dataset.get_brain_info("uniqueness"))
    print(dataset.bounds("uniqueness"))

    dataset.delete_brain_runs()
    print(dataset)


def test_detection_mistakenness():
    dataset = foz.load_zoo_dataset("quickstart").clone()

    fob.compute_mistakenness(
        dataset, "predictions", label_field="ground_truth", copy_missing=True
    )
    print(dataset.list_brain_runs())
    print(dataset.get_brain_info("mistakenness"))

    # should be non-trivial
    print(dataset.bounds("mistakenness"))
    print(dataset.bounds("possible_missing"))
    print(dataset.bounds("possible_spurious"))
    print(dataset.bounds("ground_truth.detections.mistakenness"))
    print(dataset.bounds("ground_truth.detections.mistakenness_loc"))
    print(dataset.count_values("ground_truth.detections.possible_spurious"))
    print(dataset.count_values("predictions.detections.possible_missing"))
    print(dataset.count_values("ground_truth.detections.possible_missing"))

    dataset.delete_brain_runs()
    print(dataset)

    # should be None
    print(dataset.bounds("ground_truth.detections.mistakenness"))
    print(dataset.bounds("ground_truth.detections.mistakenness_loc"))
    print(dataset.count_values("ground_truth.detections.possible_spurious"))
    print(dataset.count_values("predictions.detections.possible_missing"))
    print(dataset.count_values("ground_truth.detections.possible_missing"))


def test_classification_mistakenness_confidence():
    dataset = foz.load_zoo_dataset("quickstart").clone()
    test_view = dataset.take(10)

    # labels proxy
    model = foz.load_zoo_model("alexnet-imagenet-torch")
    test_view.apply_model(model, "alexnet")

    # predictions proxy
    model = foz.load_zoo_model("resnet50-imagenet-torch")
    test_view.apply_model(model, "resnet50")

    fob.compute_mistakenness(test_view, "resnet50", label_field="alexnet")
    print(dataset.list_brain_runs())
    print(dataset.load_brain_view("mistakenness"))
    print(dataset.bounds("mistakenness"))

    dataset.delete_brain_runs()
    print(dataset)


def test_classification_mistakenness_logits():
    dataset = foz.load_zoo_dataset("quickstart").clone()
    test_view = dataset.take(10)

    # labels proxy
    model = foz.load_zoo_model("alexnet-imagenet-torch")
    test_view.apply_model(model, "alexnet")

    # predictions proxy
    model = foz.load_zoo_model("resnet50-imagenet-torch")
    test_view.apply_model(model, "resnet50", store_logits=True)

    fob.compute_mistakenness(
        test_view, "resnet50", label_field="alexnet", use_logits=True
    )
    print(dataset.list_brain_runs())
    print(dataset.load_brain_view("mistakenness"))
    print(dataset.bounds("mistakenness"))

    dataset.delete_brain_runs()
    print(dataset)


def test_hardness():
    dataset = foz.load_zoo_dataset("quickstart").clone()
    test_view = dataset.take(10)
    model = foz.load_zoo_model("alexnet-imagenet-torch")
    test_view.apply_model(model, "alexnet", store_logits=True)

    fob.compute_hardness(test_view, "alexnet")
    print(dataset.list_brain_runs())
    print(dataset.get_brain_info("hardness"))
    print(dataset.load_brain_view("hardness"))
    print(dataset.bounds("hardness"))

    dataset.delete_brain_runs()
    print(dataset)


if __name__ == "__main__":
    fo.config.show_progress_bars = True
    unittest.main(verbosity=2)
