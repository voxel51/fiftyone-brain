"""
Definitions of methods that compute insights related to sample hardness.

| Copyright 2017-2020, Voxel51, Inc.
| `voxel51.com <https://voxel51.com/>`_
|
"""

import numpy as np
from scipy.stats import entropy

import fiftyone.core.insights as foi


def _softmax(npa):
    """Computes softmax on the numpy array npa.

    @todo Replace with scipy.special.softmax after upgrading to scipy as it is
    more numerically stable.
    """
    a = np.exp(npa)
    return a / sum(a)


def _validate(data, key):
    """Check if all samples in the dataset are usable for the hardness
    computation.

    Raise ValueError if not.
    """
    for sample in data.iter_samples():
        label = sample.get_label(key)
        if label.logits is None:
            raise ValueError("sample " + sample.id +
                " failed compute_hardness validation because it has no logits")


def compute_hardness(data, key, key_insight=None, validate=False):
    """Computes a :class:`fiftyone.core.insight.ScalarInsight` scoring the
    difficulty in which each sample was to classify.

    Will add an insight to each sample describing its "hardness" (see below)
    and associate them with the insight group `key_insight`.

    Hardness is a measure computed based on model prediction output that
    summarizes a measure of the uncertainty the model had with the sample.
    This makes hardness quantitative and can be used to detect things like
    hard samples, annotation errors during noisy training, and more.

    Algorithm: Currently, hardness is computed as a direct measure on the
    entropy of the logits.

    Args:
        data: a :class:`fiftyone.core.collection:SampleCollection`
        key: string denoting what group label to operate for getting the label
            prediction information and for adding the insight
        key_insight (None): string denoting the group for the insight
            denotation to be specified only if different than `key`
        validate (False): validate correctness of samples in data
    """
    if validate:
        _validate(data, key)

    ikey = key_insight or key
    for sample in data.iter_samples():
        label = sample.get_label(key)

        hardness = entropy(_softmax(np.asarray(label.logits)))

        insight = foi.ScalarInsight.create(name="hardness", scalar=hardness)
        sample.add_insight(ikey, insight)
