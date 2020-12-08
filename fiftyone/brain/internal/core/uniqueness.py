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

import fiftyone.core.collections as foc
import fiftyone.core.labels as fol
import fiftyone.core.utils as fou

import fiftyone.brain.internal.core.utils as fbu
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

    if roi_field is not None and isinstance(samples, foc.SampleCollection):
        fbu.validate_collection_label_fields(
            samples, [roi_field], _ALLOWED_ROI_FIELD_TYPES
        )

    model = _load_model()

    if roi_field is None:
        embeddings = _compute_embeddings(samples, model)
    else:
        embeddings = _compute_patch_embeddings(samples, model, roi_field)

    uniqueness = _compute_uniqueness(embeddings)

    logger.info("Saving results...")
    with fou.ProgressBar() as pb:
        for sample, val in zip(pb(fbu.optimize_samples(samples)), uniqueness):
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
    embeddings = None
    with fou.ProgressBar(samples) as pb:
        with model:
            for imgs in data_loader:
                vectors = model.embed_all(imgs)

                if embeddings is None:
                    embeddings = vectors
                else:
                    embeddings = np.vstack((embeddings, vectors))

                pb.set_iteration(pb.iteration + len(imgs))

    # `num_samples x dim` array of embeddings
    return embeddings


def _compute_patch_embeddings(samples, model, roi_field):
    logger.info("Preparing data...")
    data_loader = _make_patch_data_loader(samples, model, roi_field)

    logger.info("Generating embeddings...")
    embeddings = None
    with fou.ProgressBar(samples) as pb:
        with model:
            for patches in pb(data_loader):
                patches = torch.squeeze(patches, dim=0)
                patch_vectors = model.embed_all(patches)

                # Aggregate over patches
                # @todo experiment with mean(), max(), abs().max(), etc
                vectors = patch_vectors.max(axis=0)

                if embeddings is None:
                    embeddings = vectors
                else:
                    # @todo if speed is an issue, fix this...
                    embeddings = np.vstack((embeddings, vectors))

    # `num_samples x dim` array of embeddings
    return embeddings


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
    for sample in fbu.optimize_samples(samples):
        fbu.validate_image(sample)

        image_paths.append(sample.filepath)

    dataset = fout.TorchImageDataset(
        image_paths, transform=model.transforms, force_rgb=True
    )

    # There is a parallelism bug in torch==1.7 on CPU that prevents us from
    # using `num_workers > 0`
    # https://stackoverflow.com/q/64772335
    num_workers = 4 if torch.cuda.is_available() else 0

    batch_size = model.batch_size or 1
    return torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, num_workers=num_workers
    )


def _make_patch_data_loader(samples, model, roi_field):
    image_paths = []
    detections = []
    for sample in fbu.optimize_samples(samples, fields=[roi_field]):
        fbu.validate_image(sample)

        rois = _parse_rois(sample, roi_field)
        if not rois.detections:
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

    # There is a parallelism bug in torch==1.7 on CPU that prevents us from
    # using `num_workers > 0`
    # https://stackoverflow.com/q/64772335
    num_workers = 4 if torch.cuda.is_available() else 0

    return torch.utils.data.DataLoader(
        dataset, batch_size=1, num_workers=num_workers
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

    raise ValueError(
        "Sample '%s' field '%s' (%s) is not a valid ROI field; must be a %s "
        "instance"
        % (
            sample.id,
            roi_field,
            label.__class__.__name__,
            set(t.__name__ for t in _ALLOWED_ROI_FIELD_TYPES),
        )
    )
