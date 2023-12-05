"""
Similarity tests.

Usage::

    # Optional: specific backends to test
    export SIMILARITY_BACKENDS=qdrant,pinecone,milvus,lancedb,redis

    pytest tests/intensive/test_similarity.py -s -k test_XXX

Qdrant setup::

    docker pull qdrant/qdrant
    docker run -p 6333:6333 qdrant/qdrant

    pip install qdrant-client

Pinecone setup::

    # Sign up at https://www.pinecone.io
    # Download API key and environment

    pip install pinecone-client

Milvus setup::

    # Instructions from: https://milvus.io/docs/install_standalone-docker.md
    wget https://github.com/milvus-io/milvus/releases/download/v2.2.11/milvus-standalone-docker-compose.yml -O docker-compose.yml
    docker compose up -d

    pip install pymilvus

LanceDB setup::

    pip install lancedb

Redis setup::

    brew tap redis-stack/redis-stack
    brew install redis-stack
    redis-stack-server

    pip install redis

Brain config setup at `~/.fiftyone/brain_config.json`::

    {
        "similarity_backends": {
            "pinecone": {
                "environment": "us-east-1-aws",
                "api_key": "XXXXXXXX"
            },
            "qdrant": {
                "url": "http://localhost:6333"
            },
            "milvus": {
                "uri": "http://localhost:19530"
            },
            "lancedb": {
                "uri": "/tmp/lancedb"
            },
            "redis": {
                "host": "localhost",
                "port": 6379
            }
        }
    }

| Copyright 2017-2023, Voxel51, Inc.
| `voxel51.com <https://voxel51.com/>`_
|
"""
import random
import os
import unittest

import numpy as np

import fiftyone as fo
import fiftyone.brain as fob  # pylint: disable=import-error,no-name-in-module
import fiftyone.zoo as foz
from fiftyone import ViewField as F


CUSTOM_BACKENDS = ["qdrant", "pinecone", "milvus", "lancedb", "redis"]


def get_custom_backends():
    if "SIMILARITY_BACKENDS" in os.environ:
        return os.environ["SIMILARITY_BACKENDS"].split(",")

    return CUSTOM_BACKENDS


def test_brain_config():
    similarity_backends = fob.brain_config.similarity_backends

    assert "sklearn" in similarity_backends

    for backend in get_custom_backends():
        if backend == "qdrant":
            assert "qdrant" in similarity_backends

            # this isn't mandatory
            # assert "url" in similarity_backends["qdrant"]

        if backend == "pinecone":
            assert "pinecone" in similarity_backends
            assert "api_key" in similarity_backends["pinecone"]
            assert "environment" in similarity_backends["pinecone"]

        if backend == "milvus":
            assert "milvus" in similarity_backends

            # this isn't mandatory
            # assert "uri" in similarity_backends["milvus"]

        if backend == "lancedb":
            assert "lancedb" in similarity_backends

            # this isn't mandatory
            # assert "uri" in similarity_backends["lancedb"]

        if backend == "redis":
            assert "redis" in similarity_backends

            # this isn't mandatory
            # assert "host" in similarity_backends["redis"]
            # assert "port" in similarity_backends["redis"]


def test_image_similarity_backends():
    dataset = foz.load_zoo_dataset(
        "quickstart", dataset_name="quickstart-test-similarity-image"
    )

    # sklearn backend
    ###########################################################################

    index1 = fob.compute_similarity(
        dataset,
        model="clip-vit-base32-torch",
        metric="euclidean",
        embeddings=False,
        backend="sklearn",
        brain_key="clip_sklearn",
    )

    embeddings, sample_ids, _ = index1.compute_embeddings(dataset)

    index1.add_to_index(embeddings, sample_ids)
    index1.save()
    index1.reload()
    assert index1.total_index_size == 200
    assert index1.index_size == 200
    assert index1.missing_size is None

    prompt = "kites high in the air"

    view1 = dataset.sort_by_similarity(prompt, k=10, brain_key="clip_sklearn")
    assert len(view1) == 10

    del index1
    dataset.clear_cache()

    print(dataset.get_brain_info("clip_sklearn"))

    index1 = dataset.load_brain_results("clip_sklearn")
    assert index1.total_index_size == 200

    embeddings1, sample_ids1, _ = index1.get_embeddings()
    assert embeddings1.shape == (200, 512)
    assert sample_ids1.shape == (200,)

    ids = random.sample(list(index1.sample_ids), 100)

    embeddings1, sample_ids1, _ = index1.get_embeddings(sample_ids=ids)
    assert embeddings1.shape == (100, 512)
    assert sample_ids1.shape == (100,)

    index1.remove_from_index(sample_ids=ids)
    assert index1.total_index_size == 100

    index1.cleanup()
    dataset.delete_brain_run("clip_sklearn")

    # custom backends
    ###########################################################################

    for backend in get_custom_backends():
        brain_key = "clip_" + backend

        index2 = fob.compute_similarity(
            dataset,
            model="clip-vit-base32-torch",
            metric="euclidean",
            embeddings=False,
            backend=backend,
            brain_key=brain_key,
        )

        index2.add_to_index(embeddings, sample_ids)
        assert index2.total_index_size == 200
        assert index2.index_size == 200
        assert index2.missing_size is None

        view2 = dataset.sort_by_similarity(prompt, k=10, brain_key=brain_key)
        assert len(view2) == 10

        del index2
        dataset.clear_cache()

        print(dataset.get_brain_info(brain_key))

        index2 = dataset.load_brain_results(brain_key)
        assert index2.total_index_size == 200

        # Pinecone and Milvus require IDs, so this method is not supported
        if backend not in ("pinecone", "milvus"):
            embeddings2, sample_ids2, _ = index2.get_embeddings()
            assert embeddings2.shape == (200, 512)
            assert sample_ids2.shape == (200,)

        embeddings2, sample_ids2, _ = index2.get_embeddings(sample_ids=ids)
        assert embeddings2.shape == (100, 512)
        assert sample_ids2.shape == (100,)
        assert set(sample_ids1) == set(sample_ids2)

        embeddings2_dict = dict(zip(sample_ids2, embeddings2))
        _embeddings2 = np.array([embeddings2_dict[i] for i in sample_ids1])
        assert np.allclose(embeddings1, _embeddings2)

        index2.remove_from_index(sample_ids=ids)

        # Collection size is known to be wrong in Milvus after deletions
        # As of July 5, 2023 this has not been fixed
        # https://github.com/milvus-io/milvus/issues/17193
        if backend != "milvus":
            assert index2.total_index_size == 100

        index2.cleanup()
        dataset.delete_brain_run(brain_key)

    dataset.delete()


def test_patch_similarity_backends():
    dataset = foz.load_zoo_dataset(
        "quickstart", dataset_name="quickstart-test-similarity-patch"
    )

    # sklearn backend
    ###########################################################################

    index1 = fob.compute_similarity(
        dataset,
        patches_field="ground_truth",
        model="clip-vit-base32-torch",
        metric="euclidean",
        embeddings=False,
        backend="sklearn",
        brain_key="gt_clip_sklearn",
    )

    embeddings, sample_ids, label_ids = index1.compute_embeddings(dataset)

    index1.add_to_index(embeddings, sample_ids, label_ids=label_ids)
    index1.save()
    index1.reload()
    assert index1.total_index_size == 1232
    assert index1.index_size == 1232
    assert index1.missing_size is None

    view = dataset.to_patches("ground_truth")

    prompt = "cute puppies"

    view1 = view.sort_by_similarity(prompt, k=10, brain_key="gt_clip_sklearn")
    assert len(view1) == 10

    del index1
    dataset.clear_cache()

    print(dataset.get_brain_info("gt_clip_sklearn"))

    index1 = dataset.load_brain_results("gt_clip_sklearn")
    assert index1.total_index_size == 1232

    embeddings1, sample_ids1, label_ids1 = index1.get_embeddings()
    assert embeddings1.shape == (1232, 512)
    assert sample_ids1.shape == (1232,)
    assert label_ids1.shape == (1232,)

    ids = random.sample(list(index1.label_ids), 100)

    embeddings1, sample_ids1, label_ids1 = index1.get_embeddings(label_ids=ids)
    assert embeddings1.shape == (100, 512)
    assert sample_ids1.shape == (100,)
    assert label_ids1.shape == (100,)

    index1.remove_from_index(label_ids=ids)
    assert index1.total_index_size == 1132

    index1.cleanup()

    dataset.delete_brain_run("gt_clip_sklearn")

    # custom backends
    ###########################################################################

    for backend in get_custom_backends():
        brain_key = "gt_clip_" + backend

        index2 = fob.compute_similarity(
            dataset,
            patches_field="ground_truth",
            model="clip-vit-base32-torch",
            metric="euclidean",
            embeddings=False,
            backend=backend,
            brain_key=brain_key,
        )

        index2.add_to_index(embeddings, sample_ids, label_ids=label_ids)
        assert index2.total_index_size == 1232
        assert index2.index_size == 1232
        assert index2.missing_size is None

        view2 = view.sort_by_similarity(prompt, k=10, brain_key=brain_key)
        assert len(view2) == 10

        del index2
        dataset.clear_cache()

        print(dataset.get_brain_info(brain_key))

        index2 = dataset.load_brain_results(brain_key)
        assert index2.total_index_size == 1232

        # Pinecone and Milvus require IDs, so this method is not supported
        if backend not in ("pinecone", "milvus"):
            embeddings2, sample_ids2, label_ids2 = index2.get_embeddings()
            assert embeddings2.shape == (1232, 512)
            assert sample_ids2.shape == (1232,)
            assert label_ids2.shape == (1232,)

        embeddings2, sample_ids2, label_ids2 = index2.get_embeddings(
            label_ids=ids
        )
        assert embeddings2.shape == (100, 512)
        assert sample_ids2.shape == (100,)
        assert label_ids2.shape == (100,)
        assert set(label_ids1) == set(label_ids2)

        embeddings2_dict = dict(zip(label_ids2, embeddings2))
        _embeddings2 = np.array([embeddings2_dict[i] for i in label_ids1])
        assert np.allclose(embeddings1, _embeddings2)

        index2.remove_from_index(label_ids=ids)

        # Collection size is known to be wrong in Milvus after deletions
        # As of July 5, 2023 this has not been fixed
        # https://github.com/milvus-io/milvus/issues/17193
        if backend != "milvus":
            assert index2.total_index_size == 1132

        index2.cleanup()
        dataset.delete_brain_run(brain_key)

    dataset.delete()


def test_qdrant_backend_config():
    """
    - *_similarity_backends tests run with custom backends as "externally" configured
    - To test varying connection details (eg with qdrant), re-configure externally and re-run tests
    - This test white-box tests that gRPC-related config settings are applied to QdrantClient
    """

    backend = "qdrant"
    if backend not in get_custom_backends():
        return

    dataset = foz.load_zoo_dataset("quickstart", max_samples=5)
    brain_key = "clip_" + backend
    index = fob.compute_similarity(
        dataset,
        model="clip-vit-base32-torch",
        metric="euclidean",
        embeddings=False,
        backend=backend,
        brain_key=brain_key,
    )

    qclient = index.client
    qremote = qclient._client
    qdrant_config = fob.brain_config.similarity_backends["qdrant"]

    if "prefer_grpc" in qdrant_config:
        prefer_grpc = qdrant_config["prefer_grpc"]
        assert qremote._prefer_grpc == prefer_grpc
        print(f"Applied qdrant config prefer_grpc={prefer_grpc}")
    else:
        print("Qdrant config prefer_grpc unset")

    if "grpc_port" in qdrant_config:
        grpc_port = qdrant_config["grpc_port"]
        assert qremote._grpc_port == grpc_port
        print(f"Applied qdrant config grpc_port={grpc_port}")
    else:
        print("Qdrant config grpc_port unset")

    dataset.delete()


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


def test_images_embeddings():
    dataset = foz.load_zoo_dataset("quickstart", max_samples=10)
    model = foz.load_zoo_model("clip-vit-base32-torch")
    n = len(dataset)

    # Embeddings are computed on-the-fly and stored on dataset
    index1 = fob.compute_similarity(
        dataset,
        embeddings="embeddings",
        model="clip-vit-base32-torch",
        brain_key="img_sim1",
        backend="sklearn",
    )
    assert index1.total_index_size == n
    assert index1.config.supports_prompts is True
    assert "embeddings" not in index1.serialize()

    # Embeddings already exist on dataset
    dataset.compute_embeddings(model, embeddings_field="embeddings2")
    index2 = fob.compute_similarity(
        dataset,
        embeddings="embeddings2",
        model="clip-vit-base32-torch",
        brain_key="img_sim2",
        backend="sklearn",
    )
    assert index2.total_index_size == n
    assert index2.config.supports_prompts is True
    assert "embeddings" not in index2.serialize()

    # Embeddings stored in index itself
    index3 = fob.compute_similarity(
        dataset,
        model="clip-vit-base32-torch",
        brain_key="img_sim3",
        backend="sklearn",
    )
    assert index3.total_index_size == n
    assert index3.config.supports_prompts is True
    assert "embeddings" in index3.serialize()

    # Embeddings stored on dataset (but field doesn't initially exist)
    index4 = fob.compute_similarity(
        dataset,
        embeddings="embeddings4",
        brain_key="img_sim4",
        backend="sklearn",
    )
    embeddings = np.random.randn(n, 512)
    sample_ids = dataset.values("id")
    index4.add_to_index(embeddings, sample_ids)
    assert index4.total_index_size == n
    assert index4.config.supports_prompts is not True
    assert "embeddings" not in index4.serialize()

    dataset.delete()


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


def test_patches_embeddings():
    dataset = foz.load_zoo_dataset("quickstart", max_samples=10)
    model = foz.load_zoo_model("clip-vit-base32-torch")
    n = dataset.count("ground_truth.detections")

    # Embeddings are computed on-the-fly and stored on dataset
    index1 = fob.compute_similarity(
        dataset,
        patches_field="ground_truth",
        embeddings="embeddings",
        model="clip-vit-base32-torch",
        brain_key="gt_sim1",
        backend="sklearn",
    )
    assert index1.total_index_size == n
    assert index1.config.supports_prompts is True
    assert "embeddings" not in index1.serialize()

    # Embeddings already exist on dataset
    dataset.compute_patch_embeddings(
        model, "ground_truth", embeddings_field="embeddings2"
    )
    index2 = fob.compute_similarity(
        dataset,
        patches_field="ground_truth",
        embeddings="embeddings2",
        model="clip-vit-base32-torch",
        brain_key="gt_sim2",
        backend="sklearn",
    )
    assert index2.total_index_size == n
    assert index2.config.supports_prompts is True
    assert "embeddings" not in index2.serialize()

    # Embeddings stored in index itself
    index3 = fob.compute_similarity(
        dataset,
        patches_field="ground_truth",
        model="clip-vit-base32-torch",
        brain_key="gt_sim3",
        backend="sklearn",
    )
    assert index3.total_index_size == n
    assert index3.config.supports_prompts is True
    assert "embeddings" in index3.serialize()

    # Embeddings stored on dataset (but field doesn't initially exist)
    index4 = fob.compute_similarity(
        dataset,
        patches_field="ground_truth",
        embeddings="embeddings4",
        brain_key="gt_sim4",
        backend="sklearn",
    )
    embeddings = np.random.randn(n, 512)
    view = dataset.to_patches("ground_truth")
    sample_ids, label_ids = view.values(["sample_id", "id"])
    index4.add_to_index(embeddings, sample_ids, label_ids=label_ids)
    assert index4.total_index_size == n
    assert index4.config.supports_prompts is not True
    assert "embeddings" not in index4.serialize()

    dataset.delete()


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
