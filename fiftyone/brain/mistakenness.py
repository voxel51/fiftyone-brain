"""
Definitions of methods that compute insights related to the chance that a label
is a mistake.

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

from math import exp

import numpy as np
from scipy.stats import entropy

import fiftyone.core.insights as foi


def compute_mistakenness(
    data,
    key_prediction,
    key_label="ground_truth",
    key_insight=None,
    validate=False,
):
    """Adds a ``mistakenness`` :class:`fiftyone.core.insight.ScalarInsight`
    to each sample scoring the chance that the sample's ``key_label`` is
    incorrect.

    Mistakenness is computed based on the prediction output of a model (through
    logits) required on data in group ``key_prediction`` in conjunction with
    the label.  This makes the measure quantitative and can be used to detect
    things like annotation errors as well as unusually hard samples.

    **Algorithm:** the chance of a mistake is related to how confident the
    model prediction was as well as whether or not the prediction is correct.
    A prediction that is highly confident and incorrect is likely to be a
    mistake.  A prediction that is low confidence and incorrect is not likely
    to be a mistake.  Let us compute a confidence measure based on negative
    entropy of logits: $c = -entropy(logits)$. (High when low uncertainty, and
    low confidence when high uncertainty.)  Let us define modulator, $m$, based
    on whether or not the answer is correct.  $m = 1$ when the label is correct
    and $0$ otherwise. Then, mistakenness is computed using $exp(m * c)$.

    Args:
        data: an iterable of :class:`fiftyone.core.sample.Sample` instances
        key_prediction: string denoting what group label to operate for getting
            the label prediction information and potentially for adding the
            insight
        key_label: string denoting the "ground truth" label that you want to
            test for a mistake with respect to the prediction output
        key_insight (None): string denoting the group for the insight
            denotation to be specified only if different than `key`
        validate (False): whether to validate that the provided samples have
            the required fields to be processed
    """
    if validate:
        _validate(data, key_prediction, key_label)

    ikey = key_insight or key_prediction
    for sample in data:
        label = sample.get_label(key_prediction)
        check = sample.get_label(key_label)

        c = -1 * entropy(_softmax(np.asarray(label.logits)))
        m = 1 if label.label == check.label else 0
        value = exp(m * c)

        insight = foi.ScalarInsight.create(name="mistakenness", scalar=value)
        sample.add_insight(ikey, insight)


def _validate(data, key_prediction, key_label):
    """Validates that all samples in the dataset are usable for the
    mistakenness computation.
    """
    for sample in data:
        label = sample.get_label(key_prediction)
        if label.logits is None:
            raise ValueError(
                "Sample '%s' failed `compute_mistakenness` validation because "
                "it has no logits" % sample.id
            )

        label = sample.get_label(key_label)
        if label is None:
            raise ValueError(
                "Sample '%s' failed `compute_mistakenness` validation because "
                "it has no '%s' label" % (sample.id, key_label)
            )


def _softmax(npa):
    """Computes softmax on the numpy array.

    @todo Replace with ``scipy.special.softmax`` after upgrading to scipy as it
    is more numerically stable.
    """
    a = np.exp(npa)
    return a / sum(a)
