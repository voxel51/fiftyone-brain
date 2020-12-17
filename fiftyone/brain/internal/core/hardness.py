"""
Methods that compute insights related to sample hardness.

| Copyright 2017-2020, Voxel51, Inc.
| `voxel51.com <https://voxel51.com/>`_
|
"""
import logging

import numpy as np
from scipy.special import softmax
from scipy.stats import entropy

import fiftyone.core.collections as foc
import fiftyone.core.labels as fol
import fiftyone.core.utils as fou
import fiftyone.core.validation as fov


logger = logging.getLogger(__name__)


_ALLOWED_TYPES = (fol.Classification, fol.Classifications)


def compute_hardness(samples, label_field, hardness_field):
    """See ``fiftyone/brain/__init__.py``."""

    #
    # Algorithm
    #
    # Hardness is computed directly as the entropy of the logits
    #

    fov.validate_collection(samples)
    fov.validate_collection_label_fields(samples, label_field, _ALLOWED_TYPES)

    samples = samples.select_fields(label_field)

    logger.info("Computing hardness...")
    with fou.ProgressBar() as pb:
        for sample in pb(samples):
            label = _get_data(sample, label_field)
            hardness = entropy(softmax(np.asarray(label.logits)))
            sample[hardness_field] = hardness
            sample.save()

    logger.info("Hardness computation complete")


def _get_data(sample, label_field):
    label = fov.get_field(
        sample, label_field, allowed_types=_ALLOWED_TYPES, allow_none=False
    )

    if label.logits is None:
        raise ValueError(
            "Sample '%s' field '%s' has no logits" % (sample.id, label_field)
        )

    return label
