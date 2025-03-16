"""
Visualization tests.

All of these tests are designed to be run manually via::

    pytest tests/intensive/test_visualization.py -s -k test_<name>

| Copyright 2017-2025, Voxel51, Inc.
| `voxel51.com <https://voxel51.com/>`_
|
"""
import unittest

import cv2
import numpy as np
import random as rand

import fiftyone as fo
import fiftyone.brain as fob
import fiftyone.zoo as foz
from fiftyone import ViewField as F


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
        dataset,
        embeddings=embeddings,
        num_dims=2,
        verbose=True,
        seed=51,
    )

    plot = results.visualize(labels="ground_truth.label")
    plot.show()

    input("Press enter to continue...")


def test_images():
    dataset = _load_images_dataset()

    results = dataset.load_brain_results("img_viz")

    assert results.total_index_size == len(dataset)
    assert set(dataset.values("id")) == set(results.sample_ids)

    plot = results.visualize(labels="uniqueness")
    plot.show()

    input("Press enter to continue...")


def test_images_subset():
    dataset = _load_images_dataset()

    results = dataset.load_brain_results("img_viz")

    view = dataset.take(10)
    results.use_view(view)

    assert results.index_size == len(view)
    assert set(view.values("id")) == set(results.current_sample_ids)

    plot = results.visualize(labels="uniqueness")
    plot.show()

    input("Press enter to continue...")


def test_images_missing():
    dataset = _load_images_dataset().limit(4).clone()
    dataset.add_samples(
        [
            fo.Sample(filepath="non-existent1.png"),
            fo.Sample(filepath="non-existent2.png"),
            fo.Sample(filepath="non-existent3.png"),
            fo.Sample(filepath="non-existent4.png"),
        ]
    )

    sample_ids = dataset[:4].values("id")

    results = fob.compute_visualization(dataset, batch_size=1)

    assert results.total_index_size == 4
    assert set(sample_ids) == set(results.sample_ids)

    model = foz.load_zoo_model("inception-v3-imagenet-torch")
    results = fob.compute_visualization(
        dataset,
        model=model,
        embeddings="embeddings_missing",
        batch_size=1,
    )

    assert len(dataset.exists("embeddings_missing")) == 4
    assert results.total_index_size == 4
    assert set(sample_ids) == set(results.sample_ids)


def test_patches():
    dataset = _load_patches_dataset()

    results = dataset.load_brain_results("gt_viz")

    label_ids = dataset.values("ground_truth.detections.id", unwind=True)

    assert results.total_index_size == len(label_ids)
    assert set(label_ids) == set(results.label_ids)

    plot = results.visualize(labels="ground_truth.detections.label")
    plot.show()

    input("Press enter to continue...")


def test_patches_subset():
    dataset = _load_patches_dataset()

    results = dataset.load_brain_results("gt_viz")

    plot = results.visualize(
        labels="ground_truth.detections.label",
        classes=["person"],
    )
    plot.show()

    input("Press enter to continue...")

    view = dataset.filter_labels("ground_truth", F("label") == "person")
    results.use_view(view)

    label_ids = view.values("ground_truth.detections.id", unwind=True)

    assert results.index_size == len(label_ids)
    assert set(label_ids) == set(results.current_label_ids)

    plot = results.visualize(labels="ground_truth.detections.label")
    plot.show()

    input("Press enter to continue...")


def test_patches_missing():
    dataset = _load_patches_dataset().limit(4).clone()
    dataset.add_samples(
        [
            fo.Sample(filepath="non-existent1.png"),
            fo.Sample(filepath="non-existent2.png"),
            fo.Sample(filepath="non-existent3.png"),
            fo.Sample(filepath="non-existent4.png"),
        ]
    )

    for sample in dataset[4:]:
        sample["ground_truth"] = fo.Detections(
            detections=[fo.Detection(bounding_box=[0.1, 0.1, 0.8, 0.8])]
        )
        sample.save()

    results = fob.compute_visualization(
        dataset, patches_field="ground_truth", batch_size=1
    )

    num_patches = dataset[:4].count("ground_truth.detections")
    label_ids = dataset[:4].values("ground_truth.detections.id", unwind=True)

    assert results.total_index_size == num_patches
    assert set(label_ids) == set(results.label_ids)

    model = foz.load_zoo_model("inception-v3-imagenet-torch")
    results = fob.compute_visualization(
        dataset,
        model=model,
        patches_field="ground_truth",
        embeddings="embeddings_missing",
        batch_size=1,
    )

    view = dataset.filter_labels(
        "ground_truth", F("embeddings_missing") != None
    )

    assert view.count("ground_truth.detections") == num_patches
    assert results.total_index_size == num_patches
    assert set(label_ids) == set(results.label_ids)


def test_points():
    dataset = foz.load_zoo_dataset("quickstart")

    n = len(dataset)
    p = dataset.count("ground_truth.detections")
    d = 512

    points1 = np.random.rand(n, d)
    results1 = fob.compute_visualization(
        dataset,
        points=points1,
        brain_key="test1",
    )
    assert results1.points.shape == (n, d)

    points2 = {_id: np.random.rand(d) for _id in dataset.values("id")}
    results2 = fob.compute_visualization(
        dataset,
        points=points2,
        brain_key="test2",
    )
    assert results2.points.shape == (n, d)

    points3 = np.random.rand(p, d)
    results3 = fob.compute_visualization(
        dataset,
        patches_field="ground_truth",
        points=points3,
        brain_key="test3",
    )
    assert results3.points.shape == (p, d)

    points4 = {
        _id: np.random.rand(d)
        for _id in dataset.values("ground_truth.detections.id", unwind=True)
    }
    results4 = fob.compute_visualization(
        dataset,
        patches_field="ground_truth",
        points=points4,
        brain_key="test4",
    )
    assert results4.points.shape == (p, d)

    dataset.delete()


def test_similarity_index():
    dataset = foz.load_zoo_dataset(
        "quickstart", dataset_name=fo.get_default_dataset_name()
    )

    # Full similarity index

    similarity_index = fob.compute_similarity(
        dataset, brain_key="sklearn_index", backend="sklearn"
    )

    results = fob.compute_visualization(
        dataset,
        brain_key="img_viz",
        similarity_index=similarity_index,
    )

    assert len(results.points) == len(dataset)

    # Partial similarity index

    view = dataset.take(100, seed=51)
    similarity_index2 = fob.compute_similarity(
        view, brain_key="sklearn_index2", backend="sklearn"
    )

    results2 = fob.compute_visualization(
        dataset,
        brain_key="img_viz2",
        similarity_index="sklearn_index2",
    )

    assert len(results2.points) == len(view)


def test_points_field():
    dataset = _load_images_dataset()

    num_points = len(dataset)
    points = np.random.randn(num_points, 2)

    brain_key = "test_points"
    points_field = brain_key

    fob.compute_visualization(
        dataset,
        brain_key=brain_key,
        points=points,
        create_index=True,
    )

    dataset.clear_cache()
    results = dataset.load_brain_results(brain_key)

    assert results.config.points_field == points_field
    assert dataset.has_sample_field(points_field)
    assert points_field in dataset.list_indexes()

    sample_points = dataset.first()[points_field]

    assert isinstance(sample_points, list)
    assert len(sample_points) == 2
    assert isinstance(sample_points[0], float)

    points = results.points

    assert len(points) == num_points
    assert len(points[0]) == 2

    all_points = dataset.values(points_field)

    assert np.allclose(points, all_points)

    dataset.delete_brain_run(brain_key)

    assert not dataset.has_sample_field(points_field)
    assert points_field not in dataset.list_indexes()


def test_points_field_patches():
    dataset = _load_patches_dataset()

    num_points = dataset.count("ground_truth.detections")
    points = np.random.randn(num_points, 2)

    brain_key = "test_points"
    points_field = brain_key
    points_path = f"ground_truth.detections.{points_field}"

    fob.compute_visualization(
        dataset,
        brain_key=brain_key,
        points=points,
        patches_field="ground_truth",
        create_index=True,
    )

    dataset.clear_cache()
    results = dataset.load_brain_results(brain_key)

    assert results.config.points_field == points_field
    assert dataset.has_sample_field(points_path)
    # Patch visualizations can't currently make use of database indexes
    assert points_path not in dataset.list_indexes()

    label_points = dataset.first().ground_truth.detections[0][points_field]

    assert isinstance(label_points, list)
    assert len(label_points) == 2
    assert isinstance(label_points[0], float)

    points = results.points

    assert len(points) == num_points
    assert len(points[0]) == 2

    all_points = dataset.values(f"ground_truth.detections[].{points_field}")

    assert np.allclose(points, all_points)

    dataset.delete_brain_run(brain_key)

    assert not dataset.has_sample_field(points_path)


def test_index_points():
    dataset = _load_images_dataset()

    num_points = len(dataset)
    points = np.random.randn(num_points, 2)

    brain_key = "test_points"
    points_field = brain_key

    fob.compute_visualization(dataset, brain_key=brain_key, points=points)

    dataset.clear_cache()
    results = dataset.load_brain_results(brain_key)

    assert results.config.points_field is None
    assert not dataset.has_sample_field(points_field)
    assert points_field not in dataset.list_indexes()

    results.index_points()

    dataset.clear_cache()
    results = dataset.load_brain_results(brain_key)

    assert results.config.points_field == points_field
    assert dataset.has_sample_field(points_field)
    assert points_field in dataset.list_indexes()

    points = results.points
    all_points = dataset.values(points_field)

    assert np.allclose(points, all_points)

    results.remove_index()

    dataset.clear_cache()
    results = dataset.load_brain_results(brain_key)

    assert results.config.points_field is None
    assert not dataset.has_sample_field(points_field)
    assert points_field not in dataset.list_indexes()


def test_index_points_patches():
    dataset = _load_patches_dataset()

    num_points = dataset.count("ground_truth.detections")
    points = np.random.randn(num_points, 2)

    brain_key = "test_points"
    points_field = brain_key
    points_path = f"ground_truth.detections.{points_field}"

    fob.compute_visualization(
        dataset,
        brain_key=brain_key,
        points=points,
        patches_field="ground_truth",
    )

    dataset.clear_cache()
    results = dataset.load_brain_results(brain_key)

    assert results.config.points_field is None
    assert not dataset.has_sample_field(points_path)

    results.index_points()

    dataset.clear_cache()
    results = dataset.load_brain_results(brain_key)

    assert results.config.points_field == points_field
    assert dataset.has_sample_field(points_path)

    points = results.points
    all_points = dataset.values(f"ground_truth.detections[].{points_field}")

    assert np.allclose(points, all_points)

    results.remove_index()

    dataset.clear_cache()
    results = dataset.load_brain_results(brain_key)

    assert results.config.points_field is None
    assert not dataset.has_sample_field(points_path)


def _load_images_dataset():
    name = "test-visualization-images"

    if fo.dataset_exists(name):
        return fo.load_dataset(name)

    return _make_images_dataset(name)


def _load_patches_dataset():
    name = "test-visualization-patches"

    if fo.dataset_exists(name):
        return fo.load_dataset(name)

    return _make_patches_dataset(name)


def _make_images_dataset(name):
    dataset = foz.load_zoo_dataset(
        "quickstart", max_samples=20, dataset_name=name
    )
    model = foz.load_zoo_model("inception-v3-imagenet-torch")

    # Embed images
    dataset.compute_embeddings(
        model, embeddings_field="embeddings", batch_size=8
    )

    # Image visualization
    fob.compute_visualization(
        dataset,
        embeddings="embeddings",
        num_dims=2,
        verbose=True,
        seed=51,
        brain_key="img_viz",
    )

    return dataset


def _make_patches_dataset(name):
    dataset = foz.load_zoo_dataset(
        "quickstart", max_samples=20, dataset_name=name
    )
    model = foz.load_zoo_model("inception-v3-imagenet-torch")

    # Embed ground truth patches
    dataset.compute_patch_embeddings(
        model,
        "ground_truth",
        embeddings_field="embeddings",
        batch_size=8,
        force_square=True,
    )

    # Patch visualization
    fob.compute_visualization(
        dataset,
        patches_field="ground_truth",
        embeddings="embeddings",
        num_dims=2,
        verbose=True,
        seed=51,
        brain_key="gt_viz",
    )

    return dataset


if __name__ == "__main__":
    fo.config.show_progress_bars = True
    unittest.main(verbosity=2)
