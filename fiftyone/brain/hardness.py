"""
Definitions of methods that compute insights related to sample hardness.

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

import numpy as np
from scipy.stats import entropy

import fiftyone.core.insights as foi


def compute_hardness(data, key_label, key_insight=None, validate=False):
    """Adds a ``hardness`` :class:`fiftyone.core.insight.ScalarInsight` to each
    sample scoring the difficulty that ``key_label`` observed in classifying
    the sample.

    Hardness is a measure computed based on model prediction output that
    summarizes a measure of the uncertainty the model had with the sample.
    This makes hardness quantitative and can be used to detect things like
    hard samples, annotation errors during noisy training, and more.

    Args:
        data: an iterable of :class:`fiftyone.core.sample.Sample` instances
        key: string denoting what group label to operate for getting the label
            prediction information and for adding the insight
        key_insight (None): string denoting the group for the insight
            denotation to be specified only if different than `key`
        validate (False): whether to validate that the provided samples have
            the required fields to be processed
    """
    # **Algorithm:** hardness is computed directly as the entropy of the logits.
    if validate:
        _validate(data, key_label)

    if key_insight is None:
        key_insight = key_label

    for sample in data:
        label = sample.get_label(key_label)

        hardness = entropy(_softmax(np.asarray(label.logits)))

        insight = foi.ScalarInsight.create(name="hardness", scalar=hardness)
        sample.add_insight(key_insight, insight)


def _validate(data, key_label):
    """Validates that all samples in the dataset are usable for the hardness
    computation.
    """
    for sample in data:
        label = sample.get_label(key_label)
        if label.logits is None:
            raise ValueError(
                "Sample '%s' failed `compute_hardness` validation because it "
                "has no logits" % sample.id
            )


def _softmax(npa):
    """Computes softmax on the numpy array."""
    # @todo Replace with ``scipy.special.softmax`` after upgrading to scipy as
    #       it is more numerically stable.
    a = np.exp(npa)
    return a / sum(a)
