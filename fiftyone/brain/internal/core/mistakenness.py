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

import fiftyone.brain.internal.core.utils as fbu


logger = logging.getLogger(__name__)


_ALLOWED_TYPES = (fol.Classification, fol.Classifications)


def compute_mistakenness(
    samples,
    pred_field,
    label_field="ground_truth",
    mistakenness_field="mistakenness",
):
    """See :mod:`fiftyone.brain` for documentation."""

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

    if isinstance(samples, foc.SampleCollection):
        fbu.validate_collection_label_fields(
            samples, (pred_field, label_field), _ALLOWED_TYPES, same_type=True
        )

    samples = fbu.optimize_samples(samples, fields=(pred_field, label_field))

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
    pred_label, label = fbu.get_fields(
        sample,
        (pred_field, label_field),
        allowed_types=_ALLOWED_TYPES,
        same_type=True,
        allow_none=False,
    )

    if pred_label.logits is None:
        raise ValueError(
            "Sample '%s' field '%s' has no logits" % (sample.id, pred_field)
        )

    return pred_label, label


def _softmax(npa):
    # @todo replace with ``scipy.special.softmax`` after upgrading to scipy as
    # it is more numerically stable
    a = np.exp(npa)
    return a / sum(a)
