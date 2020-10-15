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

import fiftyone.core.collections as foc
import fiftyone.core.labels as fol
import fiftyone.core.utils as fou


logger = logging.getLogger(__name__)


_ALLOWED_TYPES = (fol.Classification, fol.Classifications)


def compute_mistakenness(
    samples,
    pred_field,
    label_field="ground_truth",
    mistakenness_field="mistakenness",
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
            :class:`fiftyone.core.labels.Classification` or
            :class:`fiftyone.core.labels.Classifications` label field to use
            from each sample
        label_field ("ground_truth"): the name of the "ground truth"
            :class:`fiftyone.core.labels.Classification` or
            :class:`fiftyone.core.labels.Classifications` label field that you
            want to test for a mistake with respect to the prediction output
        mistakenness_field ("mistakenness"): the field name to use to store the
            mistakenness value for each sample
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

    samples = _optimize(samples, (pred_field, label_field))

    logger.info("Computing mistakenness...")
    with fou.ProgressBar() as pb:
        for sample in pb(samples):
            pred_label, label = _get_data(sample, pred_field, label_field)

            if isinstance(pred_label, fol.Classifications):
                # For multilabel problems, all labels must match
                pred_labels = set(c.label for c in pred_label.classifications)
                labels = set(c.label for c in label.classifications)
                m = float(pred_labels == labels)
            else:
                m = float(pred_label.label == label.label)

            c = -1.0 * entropy(_softmax(np.asarray(pred_label.logits)))
            mistakenness = exp(m * c)

            sample[mistakenness_field] = mistakenness
            sample.save()

    logger.info("Mistakenness computation complete")


def _get_data(sample, pred_field, label_field):
    pred_label = sample[pred_field]
    label = sample[label_field]

    if not isinstance(pred_label, _ALLOWED_TYPES):
        raise ValueError(
            "Sample '%s' field '%s' is not a %s instance; found %s"
            % (sample.id, pred_field, _ALLOWED_TYPES, pred_label.__class__,)
        )

    if pred_label.logits is None:
        raise ValueError(
            "Sample '%s' field '%s' has no logits" % (sample.id, pred_field)
        )

    if not isinstance(label, _ALLOWED_TYPES):
        raise ValueError(
            "Sample '%s' field '%s' is not a %s instance; found %s"
            % (sample.id, label_field, _ALLOWED_TYPES, label.__class__,)
        )

    if type(pred_label) is not type(label):
        raise ValueError(
            "Sample '%s' fields %s (%s) and %s (%s) do not have the same type"
            % (
                sample.id,
                pred_field,
                type(pred_label),
                label_field,
                type(label),
            )
        )

    return pred_label, label


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
