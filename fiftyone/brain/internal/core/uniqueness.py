"""
Methods that compute insights related to sample uniqueness.

| Copyright 2017-2020, Voxel51, Inc.
| `voxel51.com <https://voxel51.com/>`_
|
"""
import logging
import warnings

import numpy as np
from sklearn.neighbors import NearestNeighbors

import fiftyone as fo
import fiftyone.core.labels as fol
import fiftyone.core.utils as fou
import fiftyone.core.validation as fov

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


def compute_uniqueness(samples, uniqueness_field, roi_field):
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

    model = _load_model()

    if roi_field is None:
        embeddings = _compute_embeddings(samples, model)
    else:
        embeddings = _compute_patch_embeddings(samples, model, roi_field)

    uniqueness = _compute_uniqueness(embeddings)

    logger.info("Saving results...")
    with fou.ProgressBar() as pb:
        for sample, val in zip(pb(samples.select_fields()), uniqueness):
            sample[uniqueness_field] = val
            sample.save()

    logger.info("Uniqueness computation complete")


def _load_model():
    logger.info("Loading uniqueness model...")
    return fbm.load_model("simple-resnet-cifar10")


def _compute_embeddings(samples, model):
    logger.info("Preparing data...")
    data_loader = _make_data_loader(samples, model)

    logger.info("Generating embeddings...")
    embeddings = []
    with fou.ProgressBar(samples) as pb:
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

    # There is a parallelism bug in torch==1.7 on CPU that prevents us from
    # using `num_workers > 0`
    # https://stackoverflow.com/q/64772335
    num_workers = 4 if torch.cuda.is_available() else 0

    if model.ragged_batches:
        kwargs = dict(collate_fn=lambda batch: batch)  # return list
    else:
        kwargs = {}

    batch_size = fo.config.default_batch_size or _DEFAULT_BATCH_SIZE
    return torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, num_workers=num_workers, **kwargs
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
        image_paths,
        detections,
        model.transforms,
        ragged_batches=model.ragged_batches,
        force_rgb=True,
    )

    # There is a parallelism bug in torch==1.7 on CPU that prevents us from
    # using `num_workers > 0`
    # https://stackoverflow.com/q/64772335
    num_workers = 4 if torch.cuda.is_available() else 0

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
