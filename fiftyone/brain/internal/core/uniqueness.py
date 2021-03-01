"""
Methods that compute insights related to sample uniqueness.

| Copyright 2017-2021, Voxel51, Inc.
| `voxel51.com <https://voxel51.com/>`_
|
"""
import logging
import warnings

import numpy as np
from sklearn.neighbors import NearestNeighbors

import eta.core.utils as etau

import fiftyone as fo
import fiftyone.core.brain as fob
import fiftyone.core.fields as fof
import fiftyone.core.labels as fol
import fiftyone.core.media as fom
import fiftyone.core.models as fomo
import fiftyone.core.utils as fou
import fiftyone.core.validation as fov
import fiftyone.zoo as foz

import fiftyone.brain.internal.models as fbm

fou.ensure_torch()
import torch
import fiftyone.utils.torch as fout


logger = logging.getLogger(__name__)


_ALLOWED_ROI_FIELD_TYPES = (
    fol.Detection,
    fol.Detections,
    fol.Polyline,
    fol.Polylines,
)
_DEFAULT_BATCH_SIZE = 16


def compute_uniqueness(
    samples, uniqueness_field, roi_field, embeddings_field, model
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

    config = UniquenessConfig(
        uniqueness_field,
        roi_field,
        embeddings_field=embeddings_field,
        model=model,
    )
    brain_key = uniqueness_field
    brain_method = config.build()
    brain_method.register_run(samples, brain_key)

    if embeddings_field is not None:
        embeddings = samples.values(embeddings_field)
        if roi_field is not None:
            embeddings = [e.max(axis=0) for e in embeddings]

        embeddings = np.stack(embeddings)
    else:
        if etau.is_str(model):
            model = foz.load_zoo_model(model)
        elif model is None:
            model = fbm.load_model("simple-resnet-cifar10")

        # @todo support non-Torch models with ragged batches
        _validate_model(model)

        if roi_field is None:
            embeddings = _compute_embeddings(samples, model)
        else:
            embeddings = _compute_patch_embeddings(samples, model, roi_field)

    uniqueness = _compute_uniqueness(embeddings)

    samples._add_field_if_necessary(uniqueness_field, fof.FloatField)
    samples.set_values(uniqueness_field, uniqueness)

    logger.info("Uniqueness computation complete")


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


def _validate_model(model):
    if not isinstance(model, fomo.Model):
        raise ValueError(
            "Model must be a %s instance; found %s" % (fomo.Model, type(model))
        )

    if model.ragged_batches:
        raise ValueError(
            "This method does not support models with ragged batches"
        )


def _compute_embeddings(samples, model):
    logger.info("Preparing data...")
    data_loader = _make_data_loader(samples, model)

    logger.info("Generating embeddings...")
    embeddings = []
    with fou.ProgressBar(samples) as pb:
        with fou.SetAttributes(model, preprocess=False):
            with model:
                for imgs in data_loader:
                    embeddings_batch = model.embed_all(imgs)
                    embeddings.append(embeddings_batch)

                    pb.set_iteration(pb.iteration + len(imgs))

    return np.concatenate(embeddings)


def _compute_patch_embeddings(samples, model, roi_field):
    logger.info("Preparing data...")
    data_loader = _make_patch_data_loader(samples, model, roi_field)

    logger.info("Generating embeddings...")
    batch_size = fo.config.default_batch_size or _DEFAULT_BATCH_SIZE
    embeddings = []
    with fou.ProgressBar(samples) as pb:
        with fou.SetAttributes(model, preprocess=False):
            with model:
                for patches in pb(data_loader):
                    patch_embeddings = []
                    for patch_batch in fou.iter_slices(patches, batch_size):
                        patch_batch_embeddings = model.embed_all(patch_batch)
                        patch_embeddings.append(patch_batch_embeddings)

                    patch_embeddings = np.concatenate(patch_embeddings)

                    # Aggregate over patches
                    # @todo experiment with mean(), max(), abs().max(), etc
                    embedding = patch_embeddings.max(axis=0)
                    embeddings.append(embedding)

    return np.stack(embeddings)


def _compute_uniqueness(embeddings):
    logger.info("Computing uniqueness...")

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


def _make_data_loader(samples, model):
    image_paths = []
    for sample in samples.select_fields():
        fov.validate_image(sample)
        image_paths.append(sample.filepath)

    dataset = fout.TorchImageDataset(
        image_paths, transform=model.transforms, force_rgb=True
    )

    batch_size = fo.config.default_batch_size or _DEFAULT_BATCH_SIZE
    num_workers = fout.recommend_num_workers()
    return torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, num_workers=num_workers
    )


def _make_patch_data_loader(samples, model, roi_field):
    image_paths = []
    detections = []
    for sample in samples.select_fields(roi_field):
        fov.validate_image(sample)

        rois = _parse_rois(sample, roi_field)
        if rois is None or not rois.detections:
            # Use entire image as ROI
            msg = "Sample found with no ROI; using the entire image..."
            warnings.warn(msg)

            rois = fol.Detections(
                detections=[fol.Detection(bounding_box=[0, 0, 1, 1])]
            )

        image_paths.append(sample.filepath)
        detections.append(rois)

    dataset = fout.TorchImagePatchesDataset(
        image_paths, detections, model.transforms, force_rgb=True
    )

    num_workers = fout.recommend_num_workers()
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        num_workers=num_workers,
        collate_fn=lambda batch: batch[0],  # return patches directly
    )


def _parse_rois(sample, roi_field):
    label = sample[roi_field]

    if isinstance(label, fol.Detections):
        return label

    if isinstance(label, fol.Detection):
        return fol.Detections(detections=[label])

    if isinstance(label, fol.Polyline):
        return fol.Detections(detections=[label.to_detection()])

    if isinstance(label, fol.Polylines):
        return label.to_detections()

    return None
