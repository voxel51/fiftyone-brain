"""
Similarity tests.

| Copyright 2017-2023, Voxel51, Inc.
| `voxel51.com <https://voxel51.com/>`_
|
"""
import unittest

import fiftyone as fo
import fiftyone.brain as fob
import fiftyone.zoo as foz
from fiftyone import ViewField as F


def test_images():
    dataset = _load_images_dataset()

    results = dataset.load_brain_results("img_sim")

    assert results.total_index_size == len(dataset)
    assert set(dataset.values("id")) == set(results.sample_ids)


def test_images_subset():
    dataset = _load_images_dataset()

    results = dataset.load_brain_results("img_sim")

    view = dataset.take(10)
    results.use_view(view)

    assert results.index_size == len(view)
    assert set(view.values("id")) == set(results.current_sample_ids)


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

    results = fob.compute_similarity(dataset, batch_size=1)

    assert results.total_index_size == 4
    assert set(sample_ids) == set(results.sample_ids)

    model = foz.load_zoo_model("inception-v3-imagenet-torch")
    results = fob.compute_similarity(
        dataset,
        model=model,
        embeddings="embeddings_missing",
        batch_size=1,
    )

    assert len(dataset.exists("embeddings_missing")) == 4
    assert results.index_size == 4
    assert set(sample_ids) == set(results.sample_ids)


def test_patches():
    dataset = _load_patches_dataset()

    results = dataset.load_brain_results("gt_sim")

    label_ids = dataset.values("ground_truth.detections.id", unwind=True)

    assert results.total_index_size == len(label_ids)
    assert set(label_ids) == set(results.label_ids)


def test_patches_subset():
    dataset = _load_patches_dataset()

    results = dataset.load_brain_results("gt_sim")

    label_ids = dataset.values("ground_truth.detections.id", unwind=True)

    assert results.total_index_size == len(label_ids)
    assert set(label_ids) == set(results.label_ids)

    view = dataset.filter_labels("ground_truth", F("label") == "person")
    results.use_view(view)

    label_ids = view.values("ground_truth.detections.id", unwind=True)

    assert results.index_size == len(label_ids)
    assert set(label_ids) == set(results.current_label_ids)


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

    results = fob.compute_similarity(
        dataset, patches_field="ground_truth", batch_size=1
    )

    num_patches = dataset[:4].count("ground_truth.detections")
    label_ids = dataset[:4].values("ground_truth.detections.id", unwind=True)

    assert results.total_index_size == num_patches
    assert set(label_ids) == set(results.label_ids)

    model = foz.load_zoo_model("inception-v3-imagenet-torch")
    results = fob.compute_similarity(
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


def _load_images_dataset():
    name = "test-similarity-images"

    if fo.dataset_exists(name):
        return fo.load_dataset(name)

    return _make_images_dataset(name)


def _load_patches_dataset():
    name = "test-similarity-patches"

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

    # Image similarity
    fob.compute_similarity(
        dataset, embeddings="embeddings", brain_key="img_sim"
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

    # Patch similarity
    fob.compute_similarity(
        dataset,
        patches_field="ground_truth",
        embeddings="embeddings",
        brain_key="gt_sim",
    )

    return dataset


if __name__ == "__main__":
    fo.config.show_progress_bars = True
    unittest.main(verbosity=2)
