"""
Representativeness tests.

| Copyright 2017-2026, Voxel51, Inc.
| `voxel51.com <https://voxel51.com/>`_
|
"""
import unittest

import numpy as np
import pytest
import sklearn.cluster as skc

import fiftyone as fo

from fiftyone.brain.internal.core.representativeness import (
    _cluster_ranker,
    _compute_representativeness,
)


def _make_embeddings():
    """Two well-separated Gaussian blobs, 500 points each.

    Large enough that the hardcoded ``KMeans(n_clusters=20)`` used by
    ``_cluster_ranker()`` does not produce degenerate near-empty clusters.
    """
    np.random.seed(0)
    a = np.random.normal(loc=[0, 0], scale=1.0, size=(500, 2))
    b = np.random.normal(loc=[30, 30], scale=1.0, size=(500, 2))
    return np.vstack([a, b]).astype(np.float32)


def _distances_to_assigned_center(embeddings):
    """Each point's distance to its own cluster center, recomputed with the
    same clustering parameters ``_cluster_ranker()`` uses internally.
    """
    clusterer = skc.KMeans(n_clusters=20, random_state=1234).fit(embeddings)
    return np.linalg.norm(
        embeddings - clusterer.cluster_centers_[clusterer.labels_], axis=1
    )


@pytest.mark.parametrize("norm_method", ["local", "global"])
def test_cluster_ranker_is_centerness_not_distance(norm_method):
    """Representativeness must decrease as distance to the cluster center
    increases, for both normalization methods.
    """
    embeddings = _make_embeddings()
    dists = _distances_to_assigned_center(embeddings)

    ranking, _ = _cluster_ranker(embeddings, norm_method=norm_method)

    corr = np.corrcoef(ranking, dists)[0, 1]
    assert corr < -0.5, (
        "representativeness should anti-correlate with distance to cluster "
        "center, got corr=%.4f" % corr
    )

    closest = int(np.argmin(dists))
    farthest = int(np.argmax(dists))
    assert ranking[closest] > ranking[farthest]

    assert ranking.min() >= 0.0
    assert ranking.max() <= 1.0 + 1e-6


def test_cluster_ranker_honors_norm_method():
    """The ``norm_method`` argument must actually be used."""
    embeddings = _make_embeddings()

    local, _ = _cluster_ranker(embeddings, norm_method="local")
    glob, _ = _cluster_ranker(embeddings, norm_method="global")

    assert not np.allclose(local, glob)


def test_cluster_ranker_rejects_bad_norm_method():
    embeddings = _make_embeddings()

    with pytest.raises(ValueError):
        _cluster_ranker(embeddings, norm_method="not-a-method")


@pytest.mark.parametrize(
    "method", ["cluster-center", "cluster-center-downweight"]
)
def test_compute_representativeness_orientation(method):
    """End-to-end: both public methods produce centerness, not distance."""
    embeddings = _make_embeddings()
    dists = _distances_to_assigned_center(embeddings)

    ranking = _compute_representativeness(embeddings, method=method)

    corr = np.corrcoef(ranking, dists)[0, 1]
    assert corr < -0.3, "got corr=%.4f for method=%s" % (corr, method)


if __name__ == "__main__":
    fo.config.show_progress_bars = True
    unittest.main(verbosity=2)
