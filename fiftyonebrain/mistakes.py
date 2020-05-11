"""
Definitions of methods that compute insights related to annotation mistakes.

| Copyright 2017-2020, Voxel51, Inc.
| `voxel51.com <https://voxel51.com/>`_
|
"""

import fiftyone.core.dataset as fod
import fiftyone.core.sample as fos


def compute_label_mistakes(data, validate=False):
    """Computes a :class:`fiftyone.core.insight.ScalarInsight` scoring the
    chance there is a mistake in each sample's label.

    Args:
        data: a :class:`fiftyone.core.collection:SampleCollection`
        validate (False): validate correctness of samples in data


    Works for classification data.
    Expects
    Will validate
    MORE
    """
    #todo add mechanism for validating the samples
