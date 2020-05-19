"""
Definitions of methods that compute insights related to sample uniqueness

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

import fiftyone.core.insights as foi
import fiftyone.utils.torch as fout


# @todo consider moving these outside to some brain utils or config
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
cpu = torch.device("cpu")


def compute_uniqueness(data, key_label=None, key_insight=None, validate=False):
    """Adds a ``uniqueness`` :class:`fiftyone.core.insight.ScalarInsight` to
    each sample scoring how unique it is with respect to the rest of the
    samples in `data`.

    This function only uses the pixel data and can process labeled or unlabeled
    data.

    Args:
        data: an iterable of :class:`fiftyone.core.sample.Sample` instances
        key_label (None): string denoting what group label to operate for
            getting the label prediction information.  If this is None, then
            all samples in `data` are used.
        key_insight (None): string denoting the group for the insight
            denotation to be specified.  If this is None, then `key_label` is
            used.  If both are none, then `key_insight` is set to `uniqueness`.
        validate (False): whether to validate that the provided samples have
            the required fields to be processed
    """
    """
    **Algorithm:** uniqueness is computed based on a backend model.  Each
    sample is embedded into a vector space based on the model.  Then, we
    compute the knn's (k is a parameter of the uniqueness function).  The
    uniqueness is then proportional to these distances NOT YET DEFINED).
    The intuition is that a sample is unique when it is far from other samples
    in the set.  This is different than, say, "representativeness" which would
    stress samples that are core to dense clusters of related samples.
    """
    # convert to a parameter with a default, for tuning
    K = 8

    if validate:
        _validate(data, key_label)

    if key_insight is None:
        if key_label is None:
            key_insight = "uniqueness"
        else:
            key_insight = key_label

    # load the model first
    # @todo before finalizing this work, make this model downloadable/loadable
    # from somewhere open/general/permanent
    MODEL_PATH="/scratch/jason-model-cache/cifar10-20200507.pth"
    # @todo make this code into some form of a fiftyone.brain class to allow
    # for different models with the same functionality (or similar).  Will use
    # the eta classes for models.  Next step
    model = Network(simple_resnet()).to(device).half()
    model.load_state_dict(torch.load(MODEL_PATH))

    # @todo support filtering down by key_label
    loader = _make_data_loader(data)

    embeds = None
    model.train(False)
    with torch.no_grad():
        for imgs, _ in loader:
            vectors = _embed(model, imgs)

            # take the vectors and then compute knn on them
            if embeds is None:
                embeds = vectors
            else:
                # @todo: if speed is an issue, fix this...
                embeds = np.vstack((embeds, vectors))

    # @todo assess whether or not this is necessary. (input is float16)
    embeds = embeds.astype('float32')

    # each row of embeddings is a sample from the dataset (via row index)
    # dists and indices have a useless first column ("itself")
    knn = NearestNeighbors(n_neighbors=K+1, algorithm="ball_tree").fit(embeds)
    dists, indices = knn.kneighbors(embeds)

    assert(indices.shape[1] == K+1)

    # @todo experiment on which method for assessing uniqueness is best
    # to get something going, for now, just use the min value
    value_dist = dists[:, 1:].min(1)
    value_dist /= value_dist.max()

    # @todo make this only filter down by the key_label
    for index, sample in enumerate(data.iter_samples()):
        insight = foi.ScalarInsight.create(name="uniqueness",
                                           scalar=value_dist[index])
        sample.add_insight(key_insight, insight)

    # @todo should these functions return anything?


def _make_data_loader(data, batch_size=16):
    """Makes a data loader that can be used for getting the dataset off-disk
    and processed by our model.

    XXX @todo should the class that ultimately wraps the model/weights be
    required to supply a data loader function like this?

    Args:
        data: an iterable of :class:`fiftyone.core.sample.Sample` instances
        batch_size: the int size of the batches in the loader
    """
    mean = [0.4914, 0.4822, 0.4465]
    std = [0.2023, 0.1994, 0.2010]
    transforms = torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize([32, 32]),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean, std),
        ]
    )

    # XXX should the fiftyone view be able to supply this operation directly?
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


def _predict(model, imgs):
    """Computes a prediction on the imgs using the model.

    @todo should be part of the actual model instance representation
    """
    inputs = dict(input=imgs.cuda().half())
    outputs = model(inputs)
    logits = outputs['logits'].detach().cpu().numpy()
    predictions = np.argmax(logits, axis=1)
    odds = np.exp(logits)
    confidences = np.max(odds, axis=1) / np.sum(odds, axis=1)
    return predictions, confidences, logits


def _embed(model, imgs):
    """Embeds the imgs into the model's space.

    @todo should be in the actual model instance representation

    XXX unclear if should be flatten or linear;
    """
    inputs = dict(input=imgs.cuda().half())
    outputs = model(inputs)
    return outputs['flatten'].detach().cpu().numpy()


def _validate(data, key_label):
    """Validates that all samples in the dataset are usable for the uniqueness
    computation by checking that their file-paths are valid.

    @todo When fiftyone extends support to cloud and non-local data, this
    validation will need to change.
    """
    # @todo add support for filtering down by key_label first in case the user
    # forgot to do that
    for sample in data:
        if not os.path.exists(sample.filepath):
            raise ValueError(
                "Sample '%s' failed `compute_uniquess` validation because it "
                "does not exist on disk at " % sample.filepath
            )
