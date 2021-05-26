"""
Methods that compute insights related to sample uniqueness.

| Copyright 2017-2021, Voxel51, Inc.
| `voxel51.com <https://voxel51.com/>`_
|
"""
import logging

import numpy as np
from sklearn.neighbors import NearestNeighbors

import eta.core.utils as etau

import fiftyone.core.brain as fob
import fiftyone.core.fields as fof
import fiftyone.core.labels as fol
import fiftyone.core.media as fom
import fiftyone.core.validation as fov
import fiftyone.zoo as foz

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


def compute_uniqueness(
    samples,
    uniqueness_field,
    roi_field,
    embeddings,
    model,
    batch_size,
    force_square,
    alpha,
):
    """See ``fiftyone/brain/__init__.py``."""

    #
    # Algorithm
    #
    # Uniqueness is computed based on a classification model.  Each sample is
    # embedded into a vector space based on the model. Then, we compute the
    # knn's (k is a parameter of the uniqueness function). The uniqueness is
    # then proportional to these distances. The intuition is that a sample is
    # unique when it is far from other samples in the set. This is different
    # than, say, "representativeness" which would stress samples that are core
    # to dense clusters of related samples.
    #

    fov.validate_collection(samples)

    if roi_field is not None:
        fov.validate_collection_label_fields(
            samples, roi_field, _ALLOWED_ROI_FIELD_TYPES
        )

    if samples.media_type == fom.VIDEO:
        raise ValueError("Uniqueness does not yet support video collections")

    if etau.is_str(embeddings):
        embeddings_field = embeddings
        embeddings = None
    else:
        embeddings_field = None

    config = UniquenessConfig(
        uniqueness_field,
        roi_field,
        embeddings_field=embeddings_field,
        model=model,
    )
    brain_key = uniqueness_field
    brain_method = config.build()
    brain_method.register_run(samples, brain_key)

    #
    # Get embeddings
    #

    if model is not None or (embeddings is None and embeddings_field is None):
        if etau.is_str(model):
            model = foz.load_zoo_model(model)
        elif model is None:
            model = fbm.load_model(_DEFAULT_MODEL)
            if batch_size is None:
                batch_size = _DEFAULT_BATCH_SIZE

        logger.info("Generating embeddings...")

        if roi_field is None:
            embeddings = samples.compute_embeddings(
                model, batch_size=batch_size
            )
        else:
            embeddings = samples.compute_patch_embeddings(
                model,
                roi_field,
                handle_missing="image",
                batch_size=batch_size,
                force_square=force_square,
                alpha=alpha,
            )

    if embeddings_field is not None:
        # extracts a potentially huge number of embedding vectors/arrays
        embeddings = samples.values(embeddings_field)

    if isinstance(embeddings, dict):
        _embeddings = []
        for _id in samples.values("id"):
            e = embeddings[_id]
            if roi_field is not None:
                # @todo experiment with mean(), max(), abs().max(), etc
                e = e.max(axis=0)

            _embeddings.append(e)

        embeddings = np.stack(_embeddings)

    logger.info("Computing uniqueness...")
    uniqueness = _compute_uniqueness(embeddings)

    samples._dataset._add_sample_field_if_necessary(
        uniqueness_field, fof.FloatField
    )
    samples.set_values(uniqueness_field, uniqueness)

    brain_method.save_run_results(samples, brain_key, None)

    logger.info("Uniqueness computation complete")


def _compute_uniqueness(embeddings):
    # @todo convert to a parameter with a default, for tuning
    K = 3

    # First column of dists and indices is self-distance
    knns = NearestNeighbors(n_neighbors=K + 1, algorithm="ball_tree").fit(
        embeddings
    )
    dists, _ = knns.kneighbors(embeddings)

    #
    # @todo experiment on which method for assessing uniqueness is best
    #
    # To get something going, for now, just take a weighted mean
    #
    weights = [0.6, 0.3, 0.1]
    sample_dists = np.mean(dists[:, 1:] * weights, axis=1)

    # Normalize to keep the user on common footing across datasets
    sample_dists /= sample_dists.max()

    return sample_dists


class UniquenessConfig(fob.BrainMethodConfig):
    def __init__(
        self,
        uniqueness_field,
        roi_field,
        embeddings_field=None,
        model=None,
        **kwargs,
    ):
        if model is not None and not etau.is_str(model):
            model = etau.get_class_name(model)

        super().__init__(**kwargs)
        self.uniqueness_field = uniqueness_field
        self.roi_field = roi_field
        self.embeddings_field = embeddings_field
        self.model = model

    @property
    def method(self):
        return "uniqueness"


class Uniqueness(fob.BrainMethod):
    def get_fields(self, samples, brain_key):
        fields = [self.config.uniqueness_field]
        if self.config.roi_field is not None:
            fields.append(self.config.roi_field)

        if self.config.embeddings_field is not None:
            fields.append(self.config.embeddings_field)

        return fields

    def cleanup(self, samples, brain_key):
        uniqueness_field = self.config.uniqueness_field
        samples._dataset.delete_sample_fields(uniqueness_field, error_level=1)

    def _validate_run(self, samples, brain_key, existing_info):
        self._validate_fields_match(
            brain_key, "uniqueness_field", existing_info
        )
