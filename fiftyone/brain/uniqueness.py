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
import torch
from torch import nn

from fiftyone.brain.models.simple_resnet import *

import fiftyone.core.insights as foi


def compute_uniqueness(data, key_label, key_insight=None, validate=False):
    """Adds a ``uniqueness`` :class:`fiftyone.core.insight.ScalarInsight` to
    each sample scoring how unique it is with respect to the rest of the
    samples in `data`.

    This function only uses the pixel data and can process labeled or unlabeled
    data.

    Args:
        data: an iterable of :class:`fiftyone.core.sample.Sample` instances
        key: string denoting what group label to operate for getting the label
            prediction information and for adding the insight
        key_insight (None): string denoting the group for the insight
            denotation to be specified only if different than `key`
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
    if validate:
        _validate(data, key_label)

    if key_insight is None:
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

    # should be part of the specific model class

    for sample in data:
        label = sample.get_label(key_label)


        hardness = entropy(_softmax(np.asarray(label.logits)))

        insight = foi.ScalarInsight.create(name="hardness", scalar=hardness)
        sample.add_insight(key_insight, insight)


def _validate(data, key_label):
    """Validates that all samples in the dataset are usable for the uniqueness
    computation by checking that their file-paths are valid.

    @todo When fiftyone extends support to cloud and non-local data, this
    validation will need to change.
    """
    for sample in data:
        if not os.path.exists(sample.filepath):
            raise ValueError(
                "Sample '%s' failed `compute_uniquess` validation because it "
                "does not exist on disk at " % sample.filepath
            )
