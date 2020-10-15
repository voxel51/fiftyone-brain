"""
Methods that compute insights related to sample uniqueness.

| Copyright 2017-2020, Voxel51, Inc.
| `voxel51.com <https://voxel51.com/>`_
|
"""
import logging
import os

import numpy as np
from sklearn.neighbors import NearestNeighbors

import eta.core.image as etai
import eta.core.learning as etal

import fiftyone.core.collections as foc
import fiftyone.core.media as fom
import fiftyone.core.utils as fou


torch = fou.lazy_import("torch")
fout = fou.lazy_import("fiftyone.utils.torch")


logger = logging.getLogger(__name__)


def compute_uniqueness(samples, uniqueness_field="uniqueness"):
    """Adds a uniqueness field to each sample scoring how unique it is with
    respect to the rest of the samples.

    This function only uses the pixel data and can therefore process labeled or
    unlabeled samples.

    Args:
        samples: an iterable of :class:`fiftyone.core.sample.Sample` instances
        uniqueness_field ("uniqueness"): the field name to use to store the
            uniqueness value for each sample
    """
    #
    # Algorithm
    #
    # Uniqueness is computed based on a classification model.  Each sample is
    # embedded into a vector space based on the model.  Then, we compute the
    # knn's (k is a parameter of the uniqueness function).  The uniqueness is
    # then proportional to these distances.  The intuition is that a sample is
    # unique when it is far from other samples in the set.  This is different
    # than, say, "representativeness" which would stress samples that are core
    # to dense clusters of related samples.
    #

    # Ensure that `torch` and `torchvision` are installed
    fou.ensure_torch()

    # @todo convert to a parameter with a default, for tuning
    K = 3

    logger.info("Loading uniqueness model...")
    model = etal.load_default_deployment_model("simple_resnet_cifar10")

    sample = _optimize(samples)

    logger.info("Preparing data...")
    data_loader = _make_data_loader(samples, model.transforms)

    # Will be `num_samples x dim`
    embeddings = None

    logger.info("Computing uniqueness...")
    with fou.ProgressBar(samples) as pb:
        with torch.no_grad():
            for imgs in data_loader:
                # @todo the existence of model.embed_all is not well engineered
                vectors = model.embed_all(imgs)

                if embeddings is None:
                    embeddings = vectors
                else:
                    # @todo if speed is an issue, fix this...
                    embeddings = np.vstack((embeddings, vectors))

                pb.set_iteration(pb.iteration + len(imgs))

    logger.info("Analyzing samples...")

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

    logger.info("Saving results...")
    with fou.ProgressBar() as pb:
        for sample, val in zip(pb(samples), sample_dists):
            sample[uniqueness_field] = val
            sample.save()

    logger.info("Uniqueness computation complete")


def _make_data_loader(samples, transforms, batch_size=16):
    """Makes a data loader that can be used for getting the dataset off-disk
    and processed by our model.

    @todo should the class that ultimately wraps the model/weights be required
    to supply a data loader function like this?

    Args:
        samples: an iterable of :class:`fiftyone.core.sample.Sample` instances
        transforms: a torchvision Transform sequence
        batch_size (16): the int size of the batches in the loader

    Returns:
        a ``torch.utils.data.DataLoader``
    """
    image_paths = []
    sample_ids = []
    for sample in samples:
        _validate(sample)

        image_paths.append(sample.filepath)
        sample_ids.append(sample.id)

    dataset = fout.TorchImageDataset(
        image_paths, transform=transforms, force_rgb=True
    )
    return torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, num_workers=4
    )


def _validate(sample):
    if not os.path.exists(sample.filepath):
        raise ValueError(
            "Sample '%s' source media '%s' does not exist on disk"
            % (sample.id, sample.filepath)
        )

    if sample.media_type != fom.IMAGE:
        raise ValueError(
            "Sample '%s' source media '%s' is not a recognized image format"
            % (sample.id, sample.filepath)
        )


def _optimize(samples, fields=None):
    # Selects only the requested fields (and always the default fields)
    if isinstance(samples, foc.SampleCollection):
        return samples.select_fields(fields)

    return samples
