"""
Representativeness methods.

| Copyright 2017-2024, Voxel51, Inc.
| `voxel51.com <https://voxel51.com/>`_
|
"""
import logging
import copy

import numpy as np
import sklearn.cluster as skc
from scipy.spatial import cKDTree

import eta.core.utils as etau

import fiftyone.core.brain as fob
import fiftyone.core.fields as fof
import fiftyone.core.labels as fol
import fiftyone.core.validation as fov

import fiftyone.brain.internal.core.utils as fbu
import fiftyone.brain.internal.models as fbm


logger = logging.getLogger(__name__)

_ALLOWED_ROI_FIELD_TYPES = (
    fol.Detection,
    fol.Detections,
    fol.Polyline,
    fol.Polylines,
)

_DEFAULT_MODEL = "simple-resnet-cifar10"
_DEFAULT_BATCH_SIZE = 16


def compute_representativeness(
    samples,
    representativeness_field,
    method,
    roi_field,
    embeddings,
    model,
    model_kwargs,
    force_square,
    alpha,
    batch_size,
    num_workers,
    skip_failures,
    progress,
):
    """See ``fiftyone/brain/__init__.py``."""

    #
    # Algorithm
    #
    # Compute cluster centers with MeanShift. The representativeness will
    # then be a scaled distance to the nearest cluster center. This puts
    # cluster centers which should represent the data the highest with a high
    # ranking and points on the outliers with low ranking.
    #

    fov.validate_image_collection(samples)

    if roi_field is not None:
        fov.validate_collection_label_fields(
            samples, roi_field, _ALLOWED_ROI_FIELD_TYPES
        )

    if etau.is_str(embeddings):
        embeddings_field, embeddings_exist = fbu.parse_embeddings_field(
            samples,
            embeddings,
            patches_field=roi_field,
        )
        embeddings = None
    else:
        embeddings_field = None
        embeddings_exist = None

    if model is None and embeddings is None and not embeddings_exist:
        model = fbm.load_model(_DEFAULT_MODEL)
        if batch_size is None:
            batch_size = _DEFAULT_BATCH_SIZE

    config = RepresentativenessConfig(
        representativeness_field,
        method=method,
        roi_field=roi_field,
        embeddings_field=embeddings_field,
        model=model,
        model_kwargs=model_kwargs,
    )
    brain_key = representativeness_field
    brain_method = config.build()
    brain_method.ensure_requirements()
    brain_method.register_run(samples, brain_key, cleanup=False)

    if roi_field is not None:
        # @todo experiment with mean(), max(), abs().max(), etc
        agg_fcn = lambda e: np.mean(e, axis=0)
    else:
        agg_fcn = None

    embeddings, sample_ids, _ = fbu.get_embeddings(
        samples,
        model=model,
        model_kwargs=model_kwargs,
        patches_field=roi_field,
        embeddings_field=embeddings_field,
        embeddings=embeddings,
        force_square=force_square,
        alpha=alpha,
        handle_missing="image",
        agg_fcn=agg_fcn,
        batch_size=batch_size,
        num_workers=num_workers,
        skip_failures=skip_failures,
        progress=progress,
    )

    logger.info("Computing representativeness...")
    representativeness = _compute_representativeness(embeddings, method=method)

    # Ensure field exists, even if `representativeness` is empty
    samples._dataset.add_sample_field(representativeness_field, fof.FloatField)

    representativeness = {
        _id: u for _id, u in zip(sample_ids, representativeness)
    }
    if representativeness:
        samples.set_values(
            representativeness_field, representativeness, key_field="id"
        )

    brain_method.save_run_results(samples, brain_key, None)

    logger.info("Representativeness computation complete")


def _compute_representativeness(embeddings, method="cluster-center"):
    #
    # @todo experiment on which method for assessing representativeness
    #
    num_embeddings = len(embeddings)
    logger.info(
        "Computing clusters for %d embeddings; this may take awhile...",
        num_embeddings,
    )

    initial_ranking, _ = _cluster_ranker(embeddings)

    if method == "cluster-center":
        final_ranking = initial_ranking
    elif method == "cluster-center-downweight":
        logger.info("Applying iterative downweighting...")
        final_ranking = _adjust_rankings(
            embeddings, initial_ranking, ball_radius=0.5
        )
    else:
        raise ValueError(
            (
                "Method '%s' not supported. Please use one of "
                "['cluster-center', 'cluster-center-downweight']"
            )
            % method
        )

    return final_ranking


def _cluster_ranker(
    embeddings, cluster_algorithm="kmeans", N=20, norm_method="local"
):
    # Cluster
    if cluster_algorithm == "meanshift":
        bandwidth = skc.estimate_bandwidth(
            embeddings, quantile=0.8, n_samples=500
        )
        clusterer = skc.MeanShift(bandwidth=bandwidth, bin_seeding=True).fit(
            embeddings
        )
    elif cluster_algorithm == "kmeans":
        clusterer = skc.KMeans(n_clusters=N, random_state=1234).fit(embeddings)
    else:
        raise ValueError(
            (
                "Clustering algorithm '%s' not supported. Please use one of "
                "['meanshift', 'kmeans']"
            )
            % cluster_algorithm
        )

    cluster_centers = clusterer.cluster_centers_
    cluster_ids = clusterer.labels_

    # Get distance from each point to it's closest cluster center
    sample_dists = np.linalg.norm(
        embeddings - cluster_centers[cluster_ids], axis=1
    )

    centerness_ranking = 1 / (1 + sample_dists)

    # Normalize per cluster vs globally
    norm_method = "local"
    if norm_method == "global":
        centerness_ranking = centerness_ranking / centerness_ranking.max()
    elif norm_method == "local":
        unique_ids = np.unique(cluster_ids)
        for unique_id in unique_ids:
            cluster_indices = np.where(cluster_ids == unique_id)[0]
            cluster_dists = sample_dists[cluster_indices]
            cluster_dists /= cluster_dists.max()
            sample_dists[cluster_indices] = cluster_dists
        centerness_ranking = sample_dists

    return centerness_ranking, clusterer


# Step 3: Adjust rankings to avoid redundancy
def _adjust_rankings(embeddings, initial_ranking, ball_radius=0.5):
    tree = cKDTree(embeddings)
    new_ranking = copy.deepcopy(initial_ranking)

    ordered_ranking = np.argsort(new_ranking)[::-1]
    visited_indices = set()

    for ranked_index in ordered_ranking:
        visited_indices.add(ranked_index)
        query_embedding = embeddings[ranked_index, :]
        nearby_indices = tree.query_ball_point(
            query_embedding, ball_radius, return_sorted=True
        )
        filtered_indices = [
            idx for idx in nearby_indices if idx not in visited_indices
        ]
        visited_indices |= set(filtered_indices)
        new_ranking[filtered_indices] = new_ranking[filtered_indices] * 0.7

    new_ranking = new_ranking / new_ranking.max()
    return new_ranking


# @todo move to `fiftyone/brain/representativeness.py`
# Don't do this hastily; `get_brain_info()` on existing datasets has this
# class's full path in it and may need migration
class RepresentativenessConfig(fob.BrainMethodConfig):
    def __init__(
        self,
        representativeness_field,
        method=None,
        roi_field=None,
        embeddings_field=None,
        model=None,
        model_kwargs=None,
        **kwargs,
    ):
        if model is not None and not etau.is_str(model):
            model = etau.get_class_name(model)

        self.representativeness_field = representativeness_field
        self._method = method
        self.roi_field = roi_field
        self.embeddings_field = embeddings_field
        self.model = model
        self.model_kwargs = model_kwargs
        super().__init__(**kwargs)

    @property
    def type(self):
        return "representativeness"

    @property
    def method(self):
        return self._method

    @classmethod
    def _virtual_attributes(cls):
        # By default 'method' is virtual but we omit so it *IS* serialized
        return ["cls", "type"]


class Representativeness(fob.BrainMethod):
    def ensure_requirements(self):
        pass

    def get_fields(self, samples, brain_key):
        fields = [self.config.representativeness_field]
        if self.config.roi_field is not None:
            fields.append(self.config.roi_field)

        if self.config.embeddings_field is not None:
            fields.append(self.config.embeddings_field)

        return fields

    def cleanup(self, samples, brain_key):
        representativeness_field = self.config.representativeness_field
        samples._dataset.delete_sample_fields(
            representativeness_field, error_level=1
        )

    def _validate_run(self, samples, brain_key, existing_info):
        self._validate_fields_match(
            brain_key, "representativeness_field", existing_info
        )
