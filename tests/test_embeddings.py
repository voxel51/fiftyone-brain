"""
Embeddings tests.

| Copyright 2017-2026, Voxel51, Inc.
| `voxel51.com <https://voxel51.com/>`_
|
"""
import numpy as np

import fiftyone as fo
import fiftyone.brain.internal.core.utils as fbu


def test_video_frame_embeddings_pool_to_one_per_sample():
    """A video's frame embeddings are mean-pooled to one embedding, and a
    video with none is skipped by _handle_missing_embeddings."""
    dataset = fo.Dataset()
    dataset.add_samples([fo.Sample(filepath="%d.mp4" % i) for i in range(3)])
    ids = dataset.values("id")
    embeddings = {
        ids[0]: np.array([[0.0, 2.0], [2.0, 4.0]]),
        ids[1]: None,
        ids[2]: np.array([[1.0, 1.0]]),
    }

    try:
        pooled, sample_ids, _ = fbu.get_embeddings(
            dataset, embeddings=embeddings
        )

        assert pooled.tolist() == [[1.0, 3.0], [1.0, 1.0]]
        assert list(sample_ids) == [ids[0], ids[2]]
    finally:
        dataset.delete()


def test_patch_embeddings_keep_one_row_per_patch():
    """Patch embeddings of an image collection are not pooled."""
    box = [0.1, 0.1, 0.5, 0.5]
    dataset = fo.Dataset()
    dataset.add_samples(
        [
            fo.Sample(
                filepath="%d.jpg" % n,
                ground_truth=fo.Detections(
                    detections=[
                        fo.Detection(label="x", bounding_box=box)
                        for _ in range(n)
                    ]
                ),
            )
            for n in (2, 3)
        ]
    )
    ids = dataset.values("id")
    embeddings = {ids[0]: np.ones((2, 4)), ids[1]: np.zeros((3, 4))}

    try:
        patches, _, label_ids = fbu.get_embeddings(
            dataset, patches_field="ground_truth", embeddings=embeddings
        )

        assert patches.shape == (5, 4)
        assert len(label_ids) == 5
    finally:
        dataset.delete()
