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

import eta.core.learning as etal
import eta.core.utils as etau

import fiftyone.core.utils as fou

torch = fou.lazy_import("torch")
fout = fou.lazy_import("fiftyone.utils.torch")


logger = logging.getLogger(__name__)


def compute_uniqueness(samples, uniqueness_field="uniqueness", validate=False):
    """Adds a uniqueness field to each sample scoring how unique it is with
    respect to the rest of the samples.

    This function only uses the pixel data and can therefore process labeled or
    unlabeled samples.

    Args:
        samples: an iterable of :class:`fiftyone.core.sample.Sample` instances
        uniqueness_field ("uniqueness"): the field name to use to store the
            uniqueness value for each sample
        validate (False): whether to validate that the provided samples have
            the required fields to be processed
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

    if validate:
        _validate(samples)

    model = etal.load_default_deployment_model("simple_resnet_cifar10")

    data_loader = _make_data_loader(samples, model.transforms)

    # Will be `num_samples x dim`
    embeds = None

    num_samples = len(samples)
    logger.info("Computing uniqueness for %d samples...", num_samples)
    with etau.ProgressBar(len(samples), iters_str="samples") as progress:
        with torch.no_grad():
            for imgs in data_loader:
                # @todo the existence of model.embed_all is not well engineered
                vectors = model.embed_all(imgs)

                if embeds is None:
                    embeds = vectors
                else:
                    # @todo if speed is an issue, fix this...
                    embeds = np.vstack((embeds, vectors))

                progress.set_iteration(progress.iteration + len(imgs))

    logger.info("Analyzing samples...")

    # First column of dists and indices is self-distance
    knn = NearestNeighbors(n_neighbors=K + 1, algorithm="ball_tree").fit(
        embeds
    )
    dists, _ = knn.kneighbors(embeds)

    #
    # @todo experiment on which method for assessing uniqueness is best
    #
    # To get something going, for now, just take a weighted mean
    #
    weights = [0.6, 0.3, 0.1]
    sample_dists = np.mean(dists[:, 1:] * weights, axis=1)

    # Normalize to keep the user on common footing across datasets
    sample_dists /= sample_dists.max()

    for sample, val in zip(samples, sample_dists):
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
    """
    image_paths = []
    sample_ids = []
    for sample in samples:
        image_paths.append(sample.filepath)
        sample_ids.append(sample.id)

    dataset = fout.TorchImageDataset(
        image_paths, transform=transforms, force_rgb=True
    )
    return torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, num_workers=4
    )


def _validate(samples):
    """Validates that all samples in the dataset are usable for the uniqueness
    computation by checking that their file-paths are valid.

    @todo When fiftyone extends support to cloud and non-local data, this
    validation will need to change.
    """
    for sample in samples:
        if not os.path.exists(sample.filepath):
            raise ValueError(
                "Sample '%s' failed `compute_uniqueness` validation because it"
                " does not exist on disk at " % sample.filepath
            )
