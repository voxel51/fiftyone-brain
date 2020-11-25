"""
Methods that compute insights related to the chance that a label is a mistake.

| Copyright 2017-2020, Voxel51, Inc.
| `voxel51.com <https://voxel51.com/>`_
|
"""
import logging
from math import exp

import numpy as np
from scipy.special import softmax
from scipy.stats import entropy

import fiftyone.core.collections as foc
import fiftyone.core.labels as fol
import fiftyone.core.utils as fou

import fiftyone.utils.eval as foue

import fiftyone.brain.internal.core.utils as fbu


logger = logging.getLogger(__name__)


_ALLOWED_TYPES = (fol.Classification, fol.Classifications, fol.Detections)
_MISSED_CONFIDENCE_THRESHOLD = 0.95
_DETECTION_IOU = 0.5
_DETECTION_IOU_STR = str(_DETECTION_IOU).replace(".", "_")


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
    # based on whether or not the answer is correct. $m = -1$ when the label is
    # correct and $1$ otherwise. Then, mistakenness is computed as
    # $(m * exp(c) + 1) / 2$ so that high confidence correct predictions result
    # in low mistakenness, high confidence incorrect predictions result in high
    # mistakenness, and low confidence predictions result in middling
    # mistakenness.
    #

    if isinstance(samples, foc.SampleCollection):
        fbu.validate_collection_label_fields(
            samples, (pred_field, label_field), _ALLOWED_TYPES, same_type=True
        )

    samples = fbu.optimize_samples(samples, fields=(pred_field, label_field))

    if samples and isinstance(next(iter(samples))[pred_field], fol.Detections):
        foue.evaluate_detections(
            samples,
            pred_field,
            label_field,
            save_sample_fields=False,
            classwise=False,
            iou=_DETECTION_IOU,
        )

    logger.info("Computing mistakenness...")
    with fou.ProgressBar() as pb:
        for sample in pb(samples):
            pred_label, label = _get_data(sample, pred_field, label_field)

            if isinstance(pred_label, fol.Detections):
                possible_spurious = 0
                possible_missing = 0
                missing_detections = {}
                sample_mistakenness = []
                pred_map = {}
                for pred_det in pred_label.detections:
                    pred_map[pred_det.id] = pred_det
                    ent = entropy(softmax(np.asarray(pred_det["logits"])))
                    gt_id = pred_det[label_field + "_eval"]["matches"][
                        _DETECTION_IOU_STR
                    ]["gt_id"]
                    conf = pred_det["confidence"]
                    if gt_id == -1 and conf > _MISSED_CONFIDENCE_THRESHOLD:
                        pred_det["possible_missing"] = True
                        possible_missing += 1
                        missing_detections[pred_det.id] = pred_det

                for gt_det in label.detections:

                    # Avoid adding the same predictions again upon multiple
                    # runs of this method
                    if "possible_missing" in gt_det:
                        if gt_det.id in missing_detections:
                            del missing_detections[gt_det.id]
                        continue

                    matches = gt_det[pred_field + "_eval"]["matches"]
                    pred_id = matches[_DETECTION_IOU_STR]["pred_id"]
                    iou = matches[_DETECTION_IOU_STR]["iou"]
                    if pred_id == -1:
                        gt_det["possible_spurious"] = True
                        possible_spurious += 1

                    else:
                        pred_det = pred_map[pred_id]
                        m = float(gt_det["label"] == pred_det["label"])
                        mistakenness_class = _compute_mistakenness_class(
                            pred_det["logits"], m
                        )
                        mistakenness_loc = _compute_mistakenness_loc(
                            pred_det["logits"], iou
                        )

                        gt_det[mistakenness_field + "_loc"] = mistakenness_loc
                        gt_det[mistakenness_field] = mistakenness_class
                        sample_mistakenness.append(mistakenness_class)

                label.detections += missing_detections

                if sample_mistakenness:
                    sample["max_" + mistakenness_field] = np.max(
                        sample_mistakenness
                    )

                sample["possible_missing"] = possible_missing
                sample["possible_spurious"] = possible_spurious

            else:
                if isinstance(pred_label, fol.Classifications):
                    # For multilabel problems, all labels must match
                    pred_labels = set(
                        c.label for c in pred_label.classifications
                    )
                    labels = set(c.label for c in label.classifications)
                    m = float(pred_labels == labels)
                else:
                    m = float(pred_label.label == label.label)

                mistakenness = _compute_mistakenness_class(
                    pred_label.logits, m
                )
                sample[mistakenness_field] = mistakenness

            sample.save()

    logger.info("Mistakenness computation complete")


def _compute_mistakenness_class(logits, m):
    # constrain m to either 1 (incorrect) or -1 (correct)
    m = m * -2.0 + 1.0

    c = -1.0 * entropy(softmax(np.asarray(logits)))
    mistakenness = (m * exp(c) + 1.0) / 2.0

    return mistakenness


def _compute_mistakenness_loc(logits, iou):
    # i = 0 for high iou, i = 1 for low iou
    i = (1.0 / (1.0 - _DETECTION_IOU)) * (1.0 - iou)

    # c = 0 for low confidence, c = 1 for high confidence
    c = exp(-1.0 * entropy(softmax(np.asarray(logits))))

    # mistakenness = i when c = i, mistakenness = 0.5 if c = 0
    # mistakenness is higher with lower IoU and closer to 0 or 1 with higher
    # confidence
    mistakenness = (c * ((2.0 * i) - 1.0) + 1.0) / 2.0

    return mistakenness


def _get_data(sample, pred_field, label_field):
    pred_label, label = fbu.get_fields(
        sample,
        (pred_field, label_field),
        allowed_types=_ALLOWED_TYPES,
        same_type=True,
        allow_none=False,
    )

    if isinstance(pred_label, fol.Detections):
        for det in pred_label.detections:
            if det.logits is None:
                raise ValueError(
                    "A detection in Sample '%s' field '%s' has no logits"
                    % (sample.id, pred_field)
                )

    else:
        if pred_label.logits is None:
            raise ValueError(
                "Sample '%s' field '%s' has no logits"
                % (sample.id, pred_field)
            )

    return pred_label, label
