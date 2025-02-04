"""
Uniqueness methods.

| Copyright 2017-2025, Voxel51, Inc.
| `voxel51.com <https://voxel51.com/>`_
|
"""
import logging

import numpy as np

import eta.core.utils as etau

import fiftyone.brain as fb
import fiftyone.core.brain as fob
import fiftyone.core.fields as fof
import fiftyone.core.labels as fol
import fiftyone.core.utils as fou
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


def compute_uniqueness(
    samples,
    uniqueness_field,
    roi_field,
    embeddings,
    similarity_index,
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

    if etau.is_str(similarity_index):
        similarity_index = samples.load_brain_results(similarity_index)

    if (
        model is None
        and embeddings is None
        and similarity_index is None
        and not embeddings_exist
    ):
        model = fbm.load_model(_DEFAULT_MODEL)
        if batch_size is None:
            batch_size = _DEFAULT_BATCH_SIZE

    config = UniquenessConfig(
        uniqueness_field,
        roi_field=roi_field,
        embeddings_field=embeddings_field,
        similarity_index=similarity_index,
        model=model,
        model_kwargs=model_kwargs,
    )
    brain_key = uniqueness_field
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
        similarity_index=similarity_index,
        force_square=force_square,
        alpha=alpha,
        handle_missing="image",
        agg_fcn=agg_fcn,
        batch_size=batch_size,
        num_workers=num_workers,
        skip_failures=skip_failures,
        progress=progress,
    )

    if similarity_index is None:
        similarity_index = fb.compute_similarity(
            samples, backend="sklearn", embeddings=False
        )
        similarity_index.add_to_index(embeddings, sample_ids)

    logger.info("Computing uniqueness...")
    uniqueness = _compute_uniqueness(
        embeddings, similarity_index, progress=progress
    )

    # Ensure field exists, even if `uniqueness` is empty
    samples._dataset.add_sample_field(uniqueness_field, fof.FloatField)

    uniqueness = {_id: u for _id, u in zip(sample_ids, uniqueness)}
    if uniqueness:
        samples.set_values(uniqueness_field, uniqueness, key_field="id")

    brain_method.save_run_results(samples, brain_key, None)

    logger.info("Uniqueness computation complete")


def _compute_uniqueness(
    embeddings, similarity_index, batch_size=10, progress=None
):
    K = 3

    num_embeddings = len(embeddings)
    if num_embeddings <= K:
        return [1] * num_embeddings

    if similarity_index.config.method == "sklearn":
        _, dists = similarity_index._kneighbors(k=K + 1, return_dists=True)
    else:
        dists = []
        with fou.ProgressBar(total=num_embeddings, progress=progress) as pb:
            for _embeddings in fou.iter_slices(embeddings, batch_size):
                _, _dists = similarity_index._kneighbors(
                    query=_embeddings, k=K + 1, return_dists=True
                )
                dists.extend(_dists)
                pb.update(len(_dists))

    dists = np.array(dists)

    # @todo experiment on which method for assessing uniqueness is best
    #
    # To get something going, for now, just take a weighted mean
    #
    weights = [0.6, 0.3, 0.1]
    sample_dists = np.mean(dists[:, 1:] * weights, axis=1)

    # Normalize to keep the user on common footing across datasets
    sample_dists /= sample_dists.max()

    return sample_dists


# @todo move to `fiftyone/brain/uniqueness.py`
# Don't do this hastily; `get_brain_info()` on existing datasets has this
# class's full path in it and may need migration
class UniquenessConfig(fob.BrainMethodConfig):
    def __init__(
        self,
        uniqueness_field,
        roi_field=None,
        embeddings_field=None,
        similarity_index=None,
        model=None,
        model_kwargs=None,
        **kwargs,
    ):
        if similarity_index is not None and not etau.is_str(similarity_index):
            similarity_index = similarity_index.key

        if model is not None and not etau.is_str(model):
            model = etau.get_class_name(model)

        self.uniqueness_field = uniqueness_field
        self.roi_field = roi_field
        self.embeddings_field = embeddings_field
        self.similarity_index = similarity_index
        self.model = model
        self.model_kwargs = model_kwargs

        super().__init__(**kwargs)

    @property
    def type(self):
        return "uniqueness"

    @property
    def method(self):
        return "neighbors"


class Uniqueness(fob.BrainMethod):
    def ensure_requirements(self):
        pass

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
