"""
Representativeness tests.

| Copyright 2017-2026, Voxel51, Inc.
| `voxel51.com <https://voxel51.com/>`_
|
"""
import numpy as np
import pytest

from fiftyone.brain.internal.core.representativeness import _cluster_ranker


def _two_blob_embeddings(seed=0, per_blob=500):
    """Two well-separated Gaussian blobs so that KMeans does not produce
    degenerate near-empty clusters.
    """
    rng = np.random.RandomState(seed)
    blob_a = rng.normal(loc=[0, 0], scale=1.0, size=(per_blob, 2))
    blob_b = rng.normal(loc=[30, 30], scale=1.0, size=(per_blob, 2))
    return np.vstack([blob_a, blob_b]).astype(np.float32)


@pytest.mark.parametrize("norm_method", ["local", "global"])
def test_cluster_ranker_is_centerness_not_distance(norm_method):
    """Representativeness must be a *centerness* score: samples close to their
    cluster center are more representative, so the ranking must be negatively
    correlated with the distance to the assigned cluster center.

    Regression test for the case where the ranking was overwritten with the
    (normalized) raw distance, inverting the semantics, and where
    ``norm_method`` was ignored (hardcoded to ``"local"``).
    """
    embeddings = _two_blob_embeddings()

    ranking, clusterer = _cluster_ranker(embeddings, norm_method=norm_method)

    dists = np.linalg.norm(
        embeddings - clusterer.cluster_centers_[clusterer.labels_], axis=1
    )
    corr = np.corrcoef(ranking, dists)[0, 1]

    # High representativeness <-> small distance to center.
    assert corr < 0, (
        "representativeness should decrease as distance to the cluster center "
        "grows (got correlation %.4f)" % corr
    )


def test_cluster_ranker_global_ranks_most_central_first():
    """With ``"global"`` normalization the single most central sample overall
    should receive the highest representativeness score.
    """
    embeddings = _two_blob_embeddings()

    ranking, clusterer = _cluster_ranker(embeddings, norm_method="global")

    dists = np.linalg.norm(
        embeddings - clusterer.cluster_centers_[clusterer.labels_], axis=1
    )
    assert np.argmax(ranking) == np.argmin(dists)


def test_cluster_ranker_local_is_normalized_per_cluster():
    """The ``"local"`` normalization should scale scores within each cluster
    so that the most central sample of every cluster has a ranking of 1.
    """
    embeddings = _two_blob_embeddings()

    ranking, clusterer = _cluster_ranker(embeddings, norm_method="local")

    cluster_ids = clusterer.labels_
    for unique_id in np.unique(cluster_ids):
        cluster_scores = ranking[cluster_ids == unique_id]
        assert np.isclose(cluster_scores.max(), 1.0)


def test_cluster_ranker_rejects_unknown_norm_method():
    embeddings = _two_blob_embeddings(per_blob=50)

    with pytest.raises(ValueError):
        _cluster_ranker(embeddings, norm_method="unknown")
