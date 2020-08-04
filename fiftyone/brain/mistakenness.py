"""
Methods that compute insights related to the chance that a label is a mistake.

| Copyright 2017-2020, Voxel51, Inc.
| `voxel51.com <https://voxel51.com/>`_
|
"""
import logging
from math import exp

import numpy as np
from scipy.stats import entropy

import eta.core.utils as etau


logger = logging.getLogger(__name__)


def compute_mistakenness(
    samples,
    pred_field,
    label_field="ground_truth",
    mistakenness_field="mistakenness",
    validate=False,
):
    """Adds a mistakenness field to each sample scoring the chance that the
    specified label field is incorrect.

    Mistakenness is computed based on the prediction output of a model (through
    logits) provided in the ``pred_field`` field of the sample in conjunction
    with the reference "ground truth" label in the ``label_field`` field.
    This makes the measure quantitative and can be used to detect things like
    annotation errors as well as unusually hard samples.

    Args:
        samples: an iterable of :class:`fiftyone.core.sample.Sample` instances
        pred_field: the name of the predicted
            :class:`fiftyone.core.labels.Classification` label field to use
            from each sample
        label_field ("ground_truth"): the name of the "ground truth"
            :class:`fiftyone.core.labels.Classification` label field that you
            want to test for a mistake with respect to the prediction output
        mistakenness_field ("mistakenness"): the field name to use to store the
            mistakenness value for each sample
        validate (False): whether to validate that the provided samples have
            the required fields to be processed
    """
    #
    # Algorithm
    #
    # The chance of a mistake is related to how confident the model prediction
    # was as well as whether or not the prediction is correct. A prediction
    # that is highly confident and incorrect is likely to be a mistake. A
    # prediction that is low confidence and incorrect is not likely to be a
    # mistake.
    #
    # Let us compute a confidence measure based on negative entropy of logits:
    # $c = -entropy(logits)$. This value is large when there is low uncertainty
    # and small when there is high uncertainty. Let us define modulator, $m$,
    # based on whether or not the answer is correct. $m = 1$ when the label is
    # correct and $0$ otherwise. Then, mistakenness is computed as $exp(m * c)$
    #
    if validate:
        _validate(samples, pred_field, label_field)

    num_samples = len(samples)
    logger.info("Computing mistakenness for %d samples...", num_samples)
    with etau.ProgressBar(iters_str="samples") as progress:
        for sample in progress(samples):
            label = sample[pred_field]
            check = sample[label_field]

            c = -1 * entropy(_softmax(np.asarray(label.logits)))
            m = 1 if label.label == check.label else 0
            mistakenness = exp(m * c)

            sample[mistakenness_field] = mistakenness
            sample.save()

    logger.info("Mistakenness computation complete")


def _validate(samples, pred_field, label_field):
    """Validates that all samples in the dataset are usable for the
    mistakenness computation.
    """
    for sample in samples:
        label = sample[pred_field]
        if label.logits is None:
            raise ValueError(
                "Sample '%s' failed `compute_mistakenness` validation because "
                "it has no logits" % sample.id
            )

        label = sample[label_field]
        if label is None:
            raise ValueError(
                "Sample '%s' failed `compute_mistakenness` validation because "
                "it has no '%s' label" % (sample.id, label_field)
            )


def _softmax(npa):
    """Computes softmax on the numpy array."""
    # @todo Replace with ``scipy.special.softmax`` after upgrading to scipy as
    #       it is more numerically stable.
    a = np.exp(npa)
    return a / sum(a)
