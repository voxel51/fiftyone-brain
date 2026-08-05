"""
Visualization tests.

All of these tests are designed to be run manually via::

    pytest tests/intensive/test_visualization.py -s -k test_<name>

| Copyright 2017-2026, Voxel51, Inc.
| `voxel51.com <https://voxel51.com/>`_
|
"""
import unittest

import cv2
import numpy as np

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


def _make_synthetic_dataset(name, n=80, dim=64, seed=7):
    if fo.dataset_exists(name):
        fo.delete_dataset(name)

    dataset = fo.Dataset(name)
    dataset.add_samples(
        [fo.Sample(filepath="/tmp/%s_%d.jpg" % (name, i)) for i in range(n)]
    )

    rng = np.random.default_rng(seed)
    for sample in dataset:
        sample["emb"] = rng.normal(size=dim).astype("float32")
        sample.save()

    return dataset, rng


def _make_synthetic_patches_dataset(
    name, n_samples=20, patches_per_sample=3, dim=64, seed=7
):
    if fo.dataset_exists(name):
        fo.delete_dataset(name)

    dataset = fo.Dataset(name)
    rng = np.random.default_rng(seed)

    samples = []
    for i in range(n_samples):
        s = fo.Sample(filepath="/tmp/%s_%d.jpg" % (name, i))
        detections = [
            fo.Detection(
                label="obj",
                bounding_box=[0.05 + 0.1 * j, 0.1, 0.2, 0.2],
            )
            for j in range(patches_per_sample)
        ]
        s["ground_truth"] = fo.Detections(detections=detections)
        samples.append(s)

    dataset.add_samples(samples)

    label_ids = dataset.values("ground_truth.detections.id", unwind=True)
    embeddings = {
        lid: rng.normal(size=dim).astype("float32") for lid in label_ids
    }
    dataset.set_label_values(
        "ground_truth.detections.emb", embeddings, dynamic=True
    )

    return dataset, rng


def _add_new_samples(dataset, n, dim, rng, prefix="new", populate_field=True):
    new_samples = [
        fo.Sample(filepath="/tmp/%s_%s_%d.jpg" % (dataset.name, prefix, i))
        for i in range(n)
    ]
    dataset.add_samples(new_samples)
    new_view = dataset.match({"filepath": {"$regex": "_%s_" % prefix}})
    new_emb = rng.normal(size=(n, dim)).astype("float32")
    new_ids = new_view.values("id")
    if populate_field:
        for sample, emb in zip(new_view, new_emb):
            sample["emb"] = emb
            sample.save()
    return new_view, new_emb, new_ids


def test_add_samples_umap_field():
    dataset, rng = _make_synthetic_dataset("test_add_samples_umap_field")

    results = fob.compute_visualization(
        dataset,
        embeddings="emb",
        method="umap",
        brain_key="vk",
        num_dims=2,
        seed=42,
        verbose=False,
    )

    initial = results.total_index_size
    new_view, _, new_ids = _add_new_samples(dataset, 20, 64, rng)

    results.add_samples(new_view, embeddings="emb")

    assert results.total_index_size == initial + 20
    assert results.points.shape == (initial + 20, 2)
    assert set(new_ids).issubset(set(results.sample_ids.tolist()))

    reloaded = dataset.load_brain_results("vk")
    assert reloaded.total_index_size == initial + 20

    dataset.delete()


def test_add_samples_umap_array():
    dataset, rng = _make_synthetic_dataset("test_add_samples_umap_array")

    results = fob.compute_visualization(
        dataset,
        embeddings="emb",
        method="umap",
        brain_key="vk",
        num_dims=2,
        seed=42,
        verbose=False,
    )

    initial = results.total_index_size
    new_view, new_emb, new_ids = _add_new_samples(
        dataset, 15, 64, rng, populate_field=False
    )

    results.add_samples(new_view, embeddings=new_emb)

    assert results.total_index_size == initial + 15
    assert set(new_ids).issubset(set(results.sample_ids.tolist()))
    for sample in new_view:
        assert sample["emb"] is not None

    dataset.delete()


def test_add_samples_pca():
    dataset, rng = _make_synthetic_dataset("test_add_samples_pca")

    results = fob.compute_visualization(
        dataset,
        embeddings="emb",
        method="pca",
        brain_key="vk",
        num_dims=2,
        seed=42,
    )

    initial = results.total_index_size
    new_view, _, new_ids = _add_new_samples(dataset, 20, 64, rng)
    results.add_samples(new_view)

    assert results.total_index_size == initial + 20

    reloaded = dataset.load_brain_results("vk")
    assert reloaded.total_index_size == initial + 20

    dataset.delete()


def test_add_samples_tsne_raises():
    dataset, _ = _make_synthetic_dataset("test_add_samples_tsne", n=80, dim=64)

    results = fob.compute_visualization(
        dataset,
        embeddings="emb",
        method="tsne",
        brain_key="vk",
        num_dims=2,
        seed=42,
        verbose=False,
    )

    try:
        results.add_samples(dataset)
        assert False, "expected ValueError"
    except ValueError as e:
        assert "does not support" in str(e)

    dataset.delete()


def test_add_samples_manual_raises():
    dataset, rng = _make_synthetic_dataset("test_add_samples_manual")

    points = rng.normal(size=(len(dataset), 2))
    results = fob.compute_visualization(dataset, points=points, brain_key="vk")

    try:
        results.add_samples(dataset)
        assert False, "expected ValueError"
    except ValueError as e:
        assert "does not support" in str(e)

    dataset.delete()


def test_add_samples_umap_no_embeddings_field_raises():
    dataset, rng = _make_synthetic_dataset("test_add_samples_no_ef")

    embeddings = np.stack(dataset.values("emb"))
    results = fob.compute_visualization(
        dataset,
        embeddings=embeddings,
        method="umap",
        brain_key="vk",
        num_dims=2,
        seed=42,
        verbose=False,
    )

    _add_new_samples(dataset, 5, 64, rng, populate_field=False)

    # UMAP runs computed without an embeddings_field cannot support
    # incremental updates
    try:
        results.add_samples(dataset, embeddings=np.zeros((5, 64)))
        assert False, "expected ValueError"
    except ValueError as e:
        assert "embeddings field" in str(e)

    dataset.delete()


def test_add_samples_legacy_result_raises():
    dataset, rng = _make_synthetic_dataset("test_add_samples_legacy")

    results = fob.compute_visualization(
        dataset,
        embeddings="emb",
        method="pca",
        brain_key="vk",
        num_dims=2,
        seed=42,
    )

    results._reducer = None
    results._reducer_blob = None

    new_view, _, _ = _add_new_samples(dataset, 5, 64, rng)

    try:
        results.add_samples(new_view)
        assert False, "expected ValueError"
    except ValueError as e:
        assert "before incremental updates" in str(e)

    dataset.delete()


def test_add_samples_dim_mismatch_raises():
    dataset, rng = _make_synthetic_dataset("test_add_samples_dim")

    results = fob.compute_visualization(
        dataset,
        embeddings="emb",
        method="pca",
        brain_key="vk",
        num_dims=2,
        seed=42,
    )

    new_view, _, _ = _add_new_samples(
        dataset, 5, 64, rng, populate_field=False
    )
    bad = np.zeros((5, 32), dtype="float32")

    try:
        results.add_samples(new_view, embeddings=bad)
        assert False, "expected ValueError"
    except ValueError as e:
        assert "dimension" in str(e)

    dataset.delete()


def test_add_samples_duplicates_not_overwritten():
    dataset, _ = _make_synthetic_dataset("test_add_samples_dup")

    results = fob.compute_visualization(
        dataset,
        embeddings="emb",
        method="pca",
        brain_key="vk",
        num_dims=2,
        seed=42,
    )

    initial = results.total_index_size
    results.add_samples(
        dataset,
        skip_existing=False,
        warn_existing=True,
    )

    assert results.total_index_size == initial

    dataset.delete()


def test_add_samples_skip_existing():
    dataset, rng = _make_synthetic_dataset("test_add_samples_skip")

    results = fob.compute_visualization(
        dataset,
        embeddings="emb",
        method="pca",
        brain_key="vk",
        num_dims=2,
        seed=42,
    )

    initial = results.total_index_size
    _add_new_samples(dataset, 15, 64, rng)

    results.add_samples(dataset)

    assert results.total_index_size == initial + 15

    dataset.delete()


def test_add_samples_no_new_is_noop():
    dataset, _ = _make_synthetic_dataset("test_add_samples_noop")

    results = fob.compute_visualization(
        dataset,
        embeddings="emb",
        method="pca",
        brain_key="vk",
        num_dims=2,
        seed=42,
    )

    initial = results.total_index_size
    results.add_samples(dataset)

    assert results.total_index_size == initial

    dataset.delete()


def test_add_samples_other_field_writes_through():
    dataset, rng = _make_synthetic_dataset("test_add_samples_otherfield")

    results = fob.compute_visualization(
        dataset,
        embeddings="emb",
        method="pca",
        brain_key="vk",
        num_dims=2,
        seed=42,
    )

    initial = results.total_index_size
    new_view, _, _ = _add_new_samples(
        dataset, 5, 64, rng, populate_field=False
    )
    for sample in new_view:
        sample["emb_v2"] = rng.normal(size=64).astype("float32")
        sample.save()

    results.add_samples(new_view, embeddings="emb_v2")

    assert results.total_index_size == initial + 5
    for sample in new_view:
        assert sample["emb"] is not None

    dataset.delete()


def test_add_samples_dict_writes_through():
    dataset, rng = _make_synthetic_dataset("test_add_samples_dictwt")

    results = fob.compute_visualization(
        dataset,
        embeddings="emb",
        method="pca",
        brain_key="vk",
        num_dims=2,
        seed=42,
    )

    initial = results.total_index_size
    new_view, _, new_ids = _add_new_samples(
        dataset, 5, 64, rng, populate_field=False
    )
    emb_dict = {_id: rng.normal(size=64).astype("float32") for _id in new_ids}

    results.add_samples(new_view, embeddings=emb_dict)

    assert results.total_index_size == initial + 5
    for sample in new_view:
        assert sample["emb"] is not None

    dataset.delete()


def test_add_samples_umap_rehydrate_training_set_only():
    dataset, rng = _make_synthetic_dataset("test_add_samples_rehydr")

    results = fob.compute_visualization(
        dataset,
        embeddings="emb",
        method="umap",
        brain_key="vk",
        num_dims=2,
        seed=42,
        verbose=False,
    )

    n_train = results._reducer.embedding_.shape[0]

    _add_new_samples(dataset, 10, 64, rng, prefix="r1")
    results = dataset.load_brain_results("vk")
    results.add_samples(dataset)

    _add_new_samples(dataset, 10, 64, rng, prefix="r2")
    results = dataset.load_brain_results("vk")
    results.add_samples(dataset)

    assert results._reducer.embedding_.shape[0] == n_train
    assert results.total_index_size == n_train + 20

    dataset.delete()


def test_add_samples_persistence():
    dataset, rng = _make_synthetic_dataset("test_add_samples_persist")

    results = fob.compute_visualization(
        dataset,
        embeddings="emb",
        method="pca",
        brain_key="vk",
        num_dims=2,
        seed=42,
    )

    _, _, new_ids = _add_new_samples(dataset, 10, 64, rng)
    results.add_samples(dataset)

    del results

    reloaded = dataset.load_brain_results("vk")
    assert reloaded.total_index_size == 90
    assert set(new_ids).issubset(set(reloaded.sample_ids.tolist()))

    dataset.delete()


def test_add_samples_requires_source_when_unset():
    dataset, rng = _make_synthetic_dataset("test_add_samples_unset")

    embeddings = np.stack(dataset.values("emb"))
    results = fob.compute_visualization(
        dataset,
        embeddings=embeddings,
        method="pca",
        brain_key="vk",
        num_dims=2,
        seed=42,
    )

    new_view, _, _ = _add_new_samples(
        dataset, 5, 64, rng, populate_field=False
    )

    try:
        results.add_samples(new_view)
        assert False, "expected ValueError"
    except ValueError as e:
        assert "No embeddings" in str(e)

    dataset.delete()


def test_add_samples_field_unpopulated_falls_back_to_config_model():
    """When the canonical embeddings_field is on the schema but unpopulated
    for the new samples, add_samples should fall back to config.model."""
    from unittest import mock

    dataset, rng = _make_synthetic_dataset(
        "test_add_samples_unpopulated_fallback", n=40, dim=64
    )

    results = fob.compute_visualization(
        dataset,
        embeddings="emb",
        model="fake-zoo-model",
        method="pca",
        brain_key="vk",
        num_dims=2,
        seed=42,
    )
    assert results.config.model == "fake-zoo-model"

    new_samples = [
        fo.Sample(filepath="/tmp/%s_new_%d.jpg" % (dataset.name, i))
        for i in range(8)
    ]
    dataset.add_samples(new_samples)
    new_view = dataset.match({"filepath": {"$regex": "_new_"}})

    class _StubModel:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def _fake_compute_embeddings(
        self_view, model, embeddings_field=None, **kwargs
    ):
        n = len(self_view)
        out = rng.normal(size=(n, 64)).astype("float32")
        if embeddings_field is not None:
            ids = self_view.values("id")
            self_view._dataset.set_values(
                embeddings_field,
                dict(zip(ids, out)),
                key_field="id",
            )
        return out

    with mock.patch(
        "fiftyone.zoo.load_zoo_model", return_value=_StubModel()
    ), mock.patch(
        "fiftyone.core.collections.SampleCollection.compute_embeddings",
        autospec=True,
        side_effect=_fake_compute_embeddings,
    ):
        results.add_samples(new_view)

    assert results.total_index_size == 48
    for sample in new_view:
        assert sample["emb"] is not None

    dataset.delete()


def test_add_samples_reuses_saved_model():
    """End-to-end check with a real zoo model. Compute UMAP visualization with
    model='mobilenet-v2-imagenet-torch', add new unpopulated samples, and
    confirm add_samples computes their embeddings via the saved model and
    writes them through to the canonical embeddings field."""
    if fo.dataset_exists("test_add_samples_reuses_saved_model"):
        fo.delete_dataset("test_add_samples_reuses_saved_model")

    dataset = foz.load_zoo_dataset(
        "quickstart",
        max_samples=20,
        dataset_name="test_add_samples_reuses_saved_model",
    )
    model = foz.load_zoo_model("mobilenet-v2-imagenet-torch")
    dataset.compute_embeddings(model, embeddings_field="emb", batch_size=8)

    results = fob.compute_visualization(
        dataset,
        embeddings="emb",
        model="mobilenet-v2-imagenet-torch",
        method="umap",
        brain_key="vk",
        num_dims=2,
        seed=42,
        verbose=False,
    )
    assert results.config.model == "mobilenet-v2-imagenet-torch"
    initial = results.total_index_size

    new_samples = [
        fo.Sample(filepath=dataset.first().filepath) for _ in range(5)
    ]
    dataset.add_samples(new_samples)
    new_view = dataset.match({"id": {"$in": [s.id for s in new_samples]}})
    assert all(s["emb"] is None for s in new_view)

    results.add_samples(dataset)

    assert results.total_index_size == initial + 5
    for sample in new_view:
        assert sample["emb"] is not None

    dataset.delete()


def test_add_samples_patches_new_samples():
    """Happy path: visualization over patches, then add new samples each
    containing several detections. All new patches should be projected."""
    dataset, rng = _make_synthetic_patches_dataset(
        "test_add_samples_patches_new", n_samples=20, patches_per_sample=3
    )

    results = fob.compute_visualization(
        dataset,
        patches_field="ground_truth",
        embeddings="emb",
        method="pca",
        brain_key="vk",
        num_dims=2,
        seed=42,
    )

    initial = results.total_index_size
    assert initial == 20 * 3

    new_samples = []
    for i in range(5):
        s = fo.Sample(filepath="/tmp/%s_new_%d.jpg" % (dataset.name, i))
        detections = []
        for j in range(3):
            d = fo.Detection(
                label="obj",
                bounding_box=[0.1 * j, 0.1, 0.2, 0.2],
            )
            d["emb"] = rng.normal(size=64).astype("float32")
            detections.append(d)
        s["ground_truth"] = fo.Detections(detections=detections)
        new_samples.append(s)
    dataset.add_samples(new_samples)

    results.add_samples(dataset)

    assert results.total_index_size == initial + 5 * 3
    new_label_ids = []
    for s in dataset.match({"filepath": {"$regex": "_new_"}}):
        new_label_ids.extend([d.id for d in s["ground_truth"].detections])
    assert set(new_label_ids).issubset(set(results.label_ids.tolist()))

    dataset.delete()


def test_add_samples_patches_field_unpopulated_falls_back_to_config_model():
    """Regression: when new patches in the dataset don't have embeddings
    populated in `config.embeddings_field`, add_samples should fall back to
    `config.model` and compute their embeddings via that model."""
    from unittest import mock

    dataset, rng = _make_synthetic_patches_dataset(
        "test_add_samples_patches_unpop",
        n_samples=15,
        patches_per_sample=3,
    )

    results = fob.compute_visualization(
        dataset,
        patches_field="ground_truth",
        embeddings="emb",
        model="fake-zoo-model",
        method="pca",
        brain_key="vk",
        num_dims=2,
        seed=42,
    )
    assert results.config.model == "fake-zoo-model"
    initial = results.total_index_size
    assert initial == 15 * 3

    new_samples = []
    for i in range(5):
        s = fo.Sample(filepath="/tmp/%s_new_%d.jpg" % (dataset.name, i))
        detections = [
            fo.Detection(
                label="obj",
                bounding_box=[0.1 * j, 0.1, 0.2, 0.2],
            )
            for j in range(2)
        ]
        s["ground_truth"] = fo.Detections(detections=detections)
        new_samples.append(s)
    dataset.add_samples(new_samples)

    class _StubModel:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def _fake_compute_patch_embeddings(
        self_view,
        model,
        patches_field,
        embeddings_field=None,
        **kwargs,
    ):
        out = {}
        for sample in self_view:
            n = len(sample[patches_field].detections)
            if n > 0:
                out[sample.id] = rng.normal(size=(n, 64)).astype("float32")
        return out

    with mock.patch(
        "fiftyone.zoo.load_zoo_model", return_value=_StubModel()
    ), mock.patch(
        "fiftyone.core.collections.SampleCollection.compute_patch_embeddings",
        autospec=True,
        side_effect=_fake_compute_patch_embeddings,
    ):
        results.add_samples(dataset)

    assert results.total_index_size == initial + 5 * 2

    dataset.delete()


def test_add_samples_patches_new_patches_in_existing_sample():
    """Regression: adding new detections to an existing sample (same
    sample_id, new label_ids) should pick up the new patches when
    skip_existing=True."""
    dataset, rng = _make_synthetic_patches_dataset(
        "test_add_samples_patches_existing", n_samples=20, patches_per_sample=3
    )

    results = fob.compute_visualization(
        dataset,
        patches_field="ground_truth",
        embeddings="emb",
        method="pca",
        brain_key="vk",
        num_dims=2,
        seed=42,
    )

    initial = results.total_index_size

    first_sample = dataset.first()
    new_detections = list(first_sample["ground_truth"].detections)
    for j in range(2):
        d = fo.Detection(
            label="obj",
            bounding_box=[0.4 + 0.1 * j, 0.4, 0.2, 0.2],
        )
        d["emb"] = rng.normal(size=64).astype("float32")
        new_detections.append(d)
    first_sample["ground_truth"] = fo.Detections(detections=new_detections)
    first_sample.save()

    new_label_ids = [d.id for d in new_detections[-2:]]

    results.add_samples(dataset)

    assert results.total_index_size == initial + 2
    assert set(new_label_ids).issubset(set(results.label_ids.tolist()))

    dataset.delete()


if __name__ == "__main__":
    fo.config.show_progress_bars = True
    unittest.main(verbosity=2)
