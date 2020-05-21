"""
Methods that compute insights related to sample uniqueness.

| Copyright 2017-2020, Voxel51, Inc.
| `voxel51.com <https://voxel51.com/>`_
|
"""
# pragma pylint: disable=redefined-builtin
# pragma pylint: disable=unused-wildcard-import
# pragma pylint: disable=wildcard-import
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from builtins import *

# pragma pylint: enable=redefined-builtin
# pragma pylint: enable=unused-wildcard-import
# pragma pylint: enable=wildcard-import

import os.path

import numpy as np
from sklearn.neighbors import NearestNeighbors
import torch
from torch import nn
import torchvision

from fiftyone.brain.models.simple_resnet import *
import fiftyone.utils.torch as fout


# @todo consider moving these outside to some brain utils or config
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
cpu = torch.device("cpu")


def compute_uniqueness(samples, uniqueness_field="uniqueness", validate=False):
    """Adds a uniqueness field to each sample scoring how unique it is with
    respect to the rest of the samples.

    This function only uses the pixel data and can process labeled or unlabeled
    data.

    Args:
        samples: an iterable of :class:`fiftyone.core.sample.Sample` instances
        uniqueness_field ("uniqueness"): the field name to use to store the
        uniqueness value for each samples.
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
    # @todo convert to a parameter with a default, for tuning
    K = 3

    if validate:
        _validate(samples)

    model = etal.load_default_deployment_model("simple_resnet_cifar10")

    loader = _make_data_loader(samples, model.transforms)

    embeds = None
    with torch.no_grad():
        for imgs, _ in loader:
            # @todo the existence of model.embed_all is not well engineered
            vectors = model.embed_all(imgs)

            if embeds is None:
                embeds = vectors
            else:
                # @todo: if speed is an issue, fix this...
                embeds = np.vstack((embeds, vectors))

    # each row of embeddings is a sample from the dataset (via row index)
    # dists and indices have a useless first column ("itself")
    knn = NearestNeighbors(n_neighbors=K+1, algorithm="ball_tree").fit(embeds)
    dists, indices = knn.kneighbors(embeds)

    assert indices.shape[1] == K+1

    # @todo experiment on which method for assessing uniqueness is best
    # to get something going, for now, just take a weight mean
    weights = [0.6, 0.3, 0.1]
    dists = dists[:, 1:]
    dists *= weights
    value_dist = dists.mean(1)

    # need to normalize to keep the user on common footing across datasets
    value_dist /= value_dist.max()

    for index, sample in enumerate(samples.iter_samples()):
        sample[uniqueness_field] = value_dist[index]

    # @todo should these functions return anything?


def _make_data_loader(data, transforms, batch_size=16):
    """Makes a data loader that can be used for getting the dataset off-disk
    and processed by our model.

    XXX @todo should the class that ultimately wraps the model/weights be
    required to supply a data loader function like this?

    Args:
        data: an iterable of :class:`fiftyone.core.sample.Sample` instances
        transforms: a torchvision Transform sequence
        batch_size: the int size of the batches in the loader
    """
    image_paths = []
    sample_ids = []
    for sample in data:
        image_paths.append(sample.filepath)
        sample_ids.append(sample.id)

    dataset = fout.TorchImageDataset(
        image_paths, sample_ids=sample_ids, transform=transforms
    )
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                       num_workers=4)


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
