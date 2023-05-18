"""
Similarity tests.

Usage::

    pytest tests/intensive/test_similarity.py -s -k test_XXX

Qdrant setup::

    docker pull qdrant/qdrant
    docker run -p 6333:6333

    pip install qdrant-client

Pinecone setup::

    # Sign up at https://www.pinecone.io
    # Download API key and environment

    pip install pinecone-client

Brain config setup at `~/.fiftyone/brain_config.json`::

    {
        "similarity_backends": {
            "pinecone": {
                "environment": "us-east-1-aws",
                "api_key": "XXXXXXXX"
            },
            "qdrant": {
                "url": "http://localhost:6333"
            }
        }
    }

| Copyright 2017-2023, Voxel51, Inc.
| `voxel51.com <https://voxel51.com/>`_
|
"""
import random
import unittest

import numpy as np

import fiftyone as fo
import fiftyone.brain as fob  # pylint: disable=import-error,no-name-in-module
import fiftyone.zoo as foz
from fiftyone import ViewField as F


def test_brain_config():
    similarity_backends = fob.brain_config.similarity_backends

    assert "sklearn" in similarity_backends

    assert "qdrant" in similarity_backends
    assert "url" in similarity_backends["qdrant"]

    assert "pinecone" in similarity_backends
    assert "api_key" in similarity_backends["pinecone"]
    assert "environment" in similarity_backends["pinecone"]


def test_image_similarity_backends():
    dataset = foz.load_zoo_dataset("quickstart")

    index1 = fob.compute_similarity(
        dataset,
        model="clip-vit-base32-torch",
        metric="euclidean",
        embeddings=False,
        backend="sklearn",
        brain_key="clip_sklearn",
    )

    index2 = fob.compute_similarity(
        dataset,
        model="clip-vit-base32-torch",
        metric="euclidean",
        embeddings=False,
        backend="qdrant",
        brain_key="clip_qdrant",
    )

    index3 = fob.compute_similarity(
        dataset,
        model="clip-vit-base32-torch",
        metric="euclidean",
        embeddings=False,
        backend="pinecone",
        brain_key="clip_pinecone",
    )

    embeddings, sample_ids, _ = index1.compute_embeddings(dataset)

    index1.add_to_index(embeddings, sample_ids)
    index1.save()
    index1.reload()
    assert index1.total_index_size == 200
    assert index1.index_size == 200
    assert index1.missing_size is None

    index2.add_to_index(embeddings, sample_ids)
    assert index2.total_index_size == 200
    assert index2.index_size == 200
    assert index2.missing_size is None

    index3.add_to_index(embeddings, sample_ids)
    assert index3.total_index_size == 200
    assert index3.index_size == 200
    assert index3.missing_size is None

    prompt = "kites high in the air"

    view1 = dataset.sort_by_similarity(prompt, k=10, brain_key="clip_sklearn")
    assert len(view1) == 10

    view2 = dataset.sort_by_similarity(prompt, k=10, brain_key="clip_qdrant")
    assert len(view2) == 10

    view3 = dataset.sort_by_similarity(prompt, k=10, brain_key="clip_pinecone")
    assert len(view3) == 10

    del index1
    del index2
    del index3
    dataset.clear_cache()

    print(dataset.get_brain_info("clip_sklearn"))
    print(dataset.get_brain_info("clip_qdrant"))
    print(dataset.get_brain_info("clip_pinecone"))

    index1 = dataset.load_brain_results("clip_sklearn")
    index2 = dataset.load_brain_results("clip_qdrant")
    index3 = dataset.load_brain_results("clip_pinecone")

    assert index1.total_index_size == 200
    assert index2.total_index_size == 200
    assert index3.total_index_size == 200

    embeddings1, sample_ids1, _ = index1.get_embeddings()
    assert embeddings1.shape == (200, 512)
    assert sample_ids1.shape == (200,)

    embeddings2, sample_ids2, _ = index2.get_embeddings()
    assert embeddings2.shape == (200, 512)
    assert sample_ids2.shape == (200,)

    # Pinecone requires IDs
    # embeddings3, sample_ids3, _ = index3.get_embeddings()
    # assert embeddings3.shape == (200, 512)
    # assert sample_ids3.shape == (200,)

    ids = random.sample(list(index1.sample_ids), 100)

    embeddings1, sample_ids1, _ = index1.get_embeddings(sample_ids=ids)
    embeddings2, sample_ids2, _ = index2.get_embeddings(sample_ids=ids)
    embeddings3, sample_ids3, _ = index3.get_embeddings(sample_ids=ids)

    embeddings2_dict = dict(zip(sample_ids2, embeddings2))
    embeddings3_dict = dict(zip(sample_ids3, embeddings3))

    _embeddings2 = np.array([embeddings2_dict[i] for i in sample_ids1])
    _embeddings3 = np.array([embeddings3_dict[i] for i in sample_ids1])

    assert embeddings1.shape == (100, 512)
    assert sample_ids1.shape == (100,)

    assert embeddings2.shape == (100, 512)
    assert sample_ids2.shape == (100,)

    assert embeddings3.shape == (100, 512)
    assert sample_ids3.shape == (100,)

    assert set(sample_ids1) == set(sample_ids2)
    assert set(sample_ids1) == set(sample_ids3)

    assert np.allclose(embeddings1, _embeddings2)
    assert np.allclose(embeddings1, _embeddings3)

    index1.remove_from_index(sample_ids=ids)
    index2.remove_from_index(sample_ids=ids)
    index3.remove_from_index(sample_ids=ids)

    assert index1.total_index_size == 100
    assert index2.total_index_size == 100
    assert index3.total_index_size == 100

    index1.cleanup()
    index2.cleanup()
    index3.cleanup()

    dataset.delete_brain_run("clip_sklearn")
    dataset.delete_brain_run("clip_qdrant")
    dataset.delete_brain_run("clip_pinecone")
    dataset.delete()


def test_patch_similarity_backends():
    dataset = foz.load_zoo_dataset("quickstart")

    index1 = fob.compute_similarity(
        dataset,
        patches_field="ground_truth",
        model="clip-vit-base32-torch",
        metric="euclidean",
        embeddings=False,
        backend="sklearn",
        brain_key="gt_clip_sklearn",
    )

    index2 = fob.compute_similarity(
        dataset,
        patches_field="ground_truth",
        model="clip-vit-base32-torch",
        metric="euclidean",
        embeddings=False,
        backend="qdrant",
        brain_key="gt_clip_qdrant",
    )

    index3 = fob.compute_similarity(
        dataset,
        patches_field="ground_truth",
        model="clip-vit-base32-torch",
        metric="euclidean",
        embeddings=False,
        backend="pinecone",
        brain_key="gt_clip_pinecone",
    )

    embeddings, sample_ids, label_ids = index1.compute_embeddings(dataset)

    index1.add_to_index(embeddings, sample_ids, label_ids=label_ids)
    index1.save()
    index1.reload()
    assert index1.total_index_size == 1232
    assert index1.index_size == 1232
    assert index1.missing_size is None

    index2.add_to_index(embeddings, sample_ids, label_ids=label_ids)
    assert index2.total_index_size == 1232
    assert index2.index_size == 1232
    assert index2.missing_size is None

    index3.add_to_index(embeddings, sample_ids, label_ids=label_ids)
    assert index3.total_index_size == 1232
    assert index3.index_size == 1232
    assert index3.missing_size is None

    view = dataset.to_patches("ground_truth")

    prompt = "cute puppies"

    view1 = view.sort_by_similarity(prompt, k=10, brain_key="gt_clip_sklearn")
    assert len(view1) == 10

    view2 = view.sort_by_similarity(prompt, k=10, brain_key="gt_clip_qdrant")
    assert len(view2) == 10

    view3 = view.sort_by_similarity(prompt, k=10, brain_key="gt_clip_pinecone")
    assert len(view3) == 10

    del index1
    del index2
    del index3
    dataset.clear_cache()

    print(dataset.get_brain_info("gt_clip_sklearn"))
    print(dataset.get_brain_info("gt_clip_qdrant"))
    print(dataset.get_brain_info("gt_clip_pinecone"))

    index1 = dataset.load_brain_results("gt_clip_sklearn")
    index2 = dataset.load_brain_results("gt_clip_qdrant")
    index3 = dataset.load_brain_results("gt_clip_pinecone")

    assert index1.total_index_size == 1232
    assert index2.total_index_size == 1232
    assert index3.total_index_size == 1232

    embeddings1, sample_ids1, label_ids1 = index1.get_embeddings()
    assert embeddings1.shape == (1232, 512)
    assert sample_ids1.shape == (1232,)
    assert label_ids1.shape == (1232,)

    embeddings2, sample_ids2, label_ids2 = index2.get_embeddings()
    assert embeddings2.shape == (1232, 512)
    assert sample_ids2.shape == (1232,)
    assert label_ids2.shape == (1232,)

    # Pinecone requires IDs
    # embeddings3, sample_ids3, label_ids3 = index3.get_embeddings()
    # assert embeddings3.shape == (1232, 512)
    # assert sample_ids3.shape == (1232,)
    # assert label_ids3.shape == (1232,)

    ids = random.sample(list(index1.label_ids), 100)

    embeddings1, sample_ids1, label_ids1 = index1.get_embeddings(label_ids=ids)
    embeddings2, sample_ids2, label_ids2 = index2.get_embeddings(label_ids=ids)
    embeddings3, sample_ids3, label_ids3 = index3.get_embeddings(label_ids=ids)

    embeddings2_dict = dict(zip(label_ids2, embeddings2))
    embeddings3_dict = dict(zip(label_ids3, embeddings3))

    _embeddings2 = np.array([embeddings2_dict[i] for i in label_ids1])
    _embeddings3 = np.array([embeddings3_dict[i] for i in label_ids1])

    assert embeddings1.shape == (100, 512)
    assert sample_ids1.shape == (100,)
    assert label_ids1.shape == (100,)

    assert embeddings2.shape == (100, 512)
    assert sample_ids2.shape == (100,)
    assert label_ids2.shape == (100,)

    assert embeddings3.shape == (100, 512)
    assert sample_ids3.shape == (100,)
    assert label_ids3.shape == (100,)

    assert set(label_ids1) == set(label_ids2)
    assert set(label_ids1) == set(label_ids3)

    assert np.allclose(embeddings1, _embeddings2)
    assert np.allclose(embeddings1, _embeddings3)

    index1.remove_from_index(label_ids=ids)
    index2.remove_from_index(label_ids=ids)
    index3.remove_from_index(label_ids=ids)

    assert index1.total_index_size == 1132
    assert index2.total_index_size == 1132
    assert index3.total_index_size == 1132

    index1.cleanup()
    index2.cleanup()
    index3.cleanup()

    dataset.delete_brain_run("gt_clip_sklearn")
    dataset.delete_brain_run("gt_clip_qdrant")
    dataset.delete_brain_run("gt_clip_pinecone")


def test_images():
    dataset = _load_images_dataset()

    index = dataset.load_brain_results("img_sim")

    assert index.total_index_size == len(dataset)
    assert set(dataset.values("id")) == set(index.sample_ids)


def test_images_subset():
    dataset = _load_images_dataset()

    index = dataset.load_brain_results("img_sim")

    view = dataset.take(10)
    index.use_view(view)

    assert index.index_size == len(view)
    assert set(view.values("id")) == set(index.current_sample_ids)


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

    index = fob.compute_similarity(dataset, batch_size=1)

    assert index.total_index_size == 4
    assert set(sample_ids) == set(index.sample_ids)

    model = foz.load_zoo_model("inception-v3-imagenet-torch")
    index = fob.compute_similarity(
        dataset,
        model=model,
        embeddings="embeddings_missing",
        batch_size=1,
    )

    assert len(dataset.exists("embeddings_missing")) == 4
    assert index.index_size == 4
    assert set(sample_ids) == set(index.sample_ids)


def test_patches():
    dataset = _load_patches_dataset()

    index = dataset.load_brain_results("gt_sim")

    label_ids = dataset.values("ground_truth.detections.id", unwind=True)

    assert index.total_index_size == len(label_ids)
    assert set(label_ids) == set(index.label_ids)


def test_patches_subset():
    dataset = _load_patches_dataset()

    index = dataset.load_brain_results("gt_sim")

    label_ids = dataset.values("ground_truth.detections.id", unwind=True)

    assert index.total_index_size == len(label_ids)
    assert set(label_ids) == set(index.label_ids)

    view = dataset.filter_labels("ground_truth", F("label") == "person")
    index.use_view(view)

    label_ids = view.values("ground_truth.detections.id", unwind=True)

    assert index.index_size == len(label_ids)
    assert set(label_ids) == set(index.current_label_ids)


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

    index = fob.compute_similarity(
        dataset, patches_field="ground_truth", batch_size=1
    )

    num_patches = dataset[:4].count("ground_truth.detections")
    label_ids = dataset[:4].values("ground_truth.detections.id", unwind=True)

    assert index.total_index_size == num_patches
    assert set(label_ids) == set(index.label_ids)

    model = foz.load_zoo_model("inception-v3-imagenet-torch")
    index = fob.compute_similarity(
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
    assert index.total_index_size == num_patches
    assert set(label_ids) == set(index.label_ids)


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
