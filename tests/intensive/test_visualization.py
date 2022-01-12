"""
Visualization tests.

| Copyright 2017-2022, Voxel51, Inc.
| `voxel51.com <https://voxel51.com/>`_
|
"""
import unittest

import cv2
import numpy as np

import fiftyone as fo
import fiftyone.brain as fob
import fiftyone.zoo as foz


_DATASET_NAME = "visualization-test"


def test_mnist():
    dataset = foz.load_zoo_dataset("mnist", split="test")

    # pylint: disable=no-member
    embeddings = np.array(
        [
            cv2.imread(f, cv2.IMREAD_UNCHANGED).ravel()
            for f in dataset.values("filepath")
        ]
    )

    results = fob.compute_visualization(
        dataset, embeddings=embeddings, num_dims=2, verbose=True, seed=51
    )

    plot = results.visualize(labels="ground_truth.label")
    plot.show()

    input("Press enter to continue...")


def test_images():
    dataset = _load_dataset()

    results = fob.compute_visualization(
        dataset, embeddings="embeddings", num_dims=2, verbose=True, seed=51
    )

    plot = results.visualize(labels="uniqueness")
    plot.show()

    input("Press enter to continue...")


def test_objects():
    dataset = _load_dataset()

    results = fob.compute_visualization(
        dataset,
        patches_field="ground_truth",
        embeddings="gt_embeddings",
        num_dims=2,
        verbose=True,
        seed=51,
    )

    plot = results.visualize(labels="ground_truth.detections.label")
    plot.show()

    input("Press enter to continue...")


def test_objects_subset():
    dataset = _load_dataset()

    results = fob.compute_visualization(
        dataset,
        patches_field="ground_truth",
        embeddings="gt_embeddings",
        num_dims=2,
        verbose=True,
        seed=51,
    )

    counts = dataset.count_values("ground_truth.detections.label")
    classes = sorted(counts, key=counts.get, reverse=True)[:5]

    plot = results.visualize(
        labels="ground_truth.detections.label", classes=classes
    )
    plot.show()

    input("Press enter to continue...")


def _load_dataset():
    if fo.dataset_exists(_DATASET_NAME):
        return fo.load_dataset(_DATASET_NAME)

    dataset = foz.load_zoo_dataset("quickstart", dataset_name=_DATASET_NAME)
    dataset.persistent = True

    model = foz.load_zoo_model("inception-v3-imagenet-torch")

    # Embed images
    dataset.compute_embeddings(
        model, embeddings_field="embeddings", batch_size=16
    )

    # Embed ground truth patches
    dataset.compute_patch_embeddings(
        model,
        "ground_truth",
        embeddings_field="gt_embeddings",
        batch_size=16,
        force_square=True,
    )

    return dataset


if __name__ == "__main__":
    fo.config.show_progress_bars = True
    unittest.main(verbosity=2)
