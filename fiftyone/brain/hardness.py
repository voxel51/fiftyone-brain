"""
Methods that compute insights related to sample hardness.

| Copyright 2017-2020, Voxel51, Inc.
| `voxel51.com <https://voxel51.com/>`_
|
"""
import logging

import numpy as np
from scipy.stats import entropy

import fiftyone.core.collections as foc
import fiftyone.core.labels as fol
import fiftyone.core.utils as fou


logger = logging.getLogger(__name__)


def compute_hardness(
    samples, label_field, hardness_field="hardness", validate=False
):
    """Adds a hardness field to each sample scoring the difficulty that the
    specified label field observed in classifying the sample.

    Hardness is a measure computed based on model prediction output that
    summarizes a measure of the uncertainty the model had with the sample.
    This makes hardness quantitative and can be used to detect things like
    hard samples, annotation errors during noisy training, and more.

    Args:
        samples: an iterable of :class:`fiftyone.core.sample.Sample` instances
        label_field: the :class:`fiftyone.core.labels.Classification` label
            field to use from each sample
        hardness_field ("hardness"): the field name to use to store the
            hardness value for each sample
        validate (False): whether to validate that the provided samples have
            the required fields prior to processing them
    """
    #
    # Algorithm
    #
    # Hardness is computed directly as the entropy of the logits
    #

    if validate:
        _validate(samples, label_field)

    logger.info("Computing hardness...")
    with fou.ProgressBar() as pb:
        for sample in pb(samples):
            label = sample[label_field]
            hardness = entropy(_softmax(np.asarray(label.logits)))
            sample[hardness_field] = hardness
            sample.save()

    logger.info("Hardness computation complete")


def _validate(samples, label_field):
    logger.info("Validating samples...")
    with fou.ProgressBar() as pb:
        for sample in pb(_optimize(samples, label_field)):
            label = sample[label_field]

            if not isinstance(label, fol.Classification):
                raise ValueError(
                    "Sample '%s' failed validation because its '%s' field is "
                    "not a %s instance; expected %s"
                    % (
                        sample.id,
                        label_field,
                        label.__class__,
                        fol.Classification,
                    )
                )

            if label.logits is None:
                raise ValueError(
                    "Sample '%s' failed validation because its '%s' field has "
                    "no logits" % (sample.id, label)
                )


def _softmax(npa):
    # @todo replace with ``scipy.special.softmax`` after upgrading to scipy as
    # it is more numerically stable
    a = np.exp(npa)
    return a / sum(a)


def _optimize(samples, fields=None):
    # Selects only the requested fields (and always the default fields)
    if isinstance(samples, foc.SampleCollection):
        return samples.select_fields(fields)

    return samples
