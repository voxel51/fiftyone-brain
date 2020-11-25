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
    missing_field="possible_missing",
    spurious_field="possible_spurious",
    use_logits=True,
):
    """Computes the mistakenness of the labels in the specified
    ``label_field``, scoring the chance that the labels are incorrect.

    Mistakenness is computed based on the predictions in the ``pred_field``,
    through its ``logits`` or ``confidence``. This measure can be used to
    detect things like annotation errors and unusually hard samples.

    This method supports both classifications and detections.

    For classifications, a ``mistakenness_field`` field is populated on each
    sample that quantifies the likelihood that the label in the ``label_field``
    of that sample is incorrect.

    For detections, the mistakenness of each detection in ``label_field`` is
    computed, using :meth:`fiftyone.utils.evaluation.evaluate_detections` to
    locate corresponding detections in ``pred_field``. Three types of mistakes
    are identified:

    -   (Mistakes) Detections with a match in ``pred_field`` are assigned a
        mistakenness value in their ``mistakenness_field``, which captures the
        likelihood that the detection in ``label_field`` is a mistake. Such
        mistakes may be due to either the class label or localization of the
        detection

    -   (Missing) Detections in ``pred_field`` with no matches in
        ``label_field`` but which are likely to be correct are *added* to
        ``label_field`` and given a value of ``True`` in their
        ``missing_field`` attribute

    -   (Spurious) Detections in ``label_field`` with no matches in
        ``pred_field`` but which are likely to be incorrect are given a value
        of ``True`` in their ``spurious_field`` attribute

    These per-detection data are then aggregated at the sample-level as
    follows:

    -   (Mistakes) The ``mistakenness_field`` of each sample is populated with
        the maximum mistakenness of the detections in ``label_field``

    -   (Missing) The ``missing_field`` of each sample is populated with the
        number of missing detections that were deemed missing and thus added
        to ``label_field``

    -   (Spurious) The ``spurious_field`` of each sample is populated with the
        number of detections in ``label_field`` that were given deemed spurious

    Args:
        samples: an iterable of :class:`fiftyone.core.sample.Sample` instances
        pred_field: the name of the predicted label field to use from each
            sample. Can be of type
                :class:`fiftyone.core.labels.Classification`,
                :class:`fiftyone.core.labels.Classifications`, or
                :class:`fiftyone.core.labels.Detections`
        label_field ("ground_truth"): the name of the "ground truth" label
            field that you want to test for mistakes with respect to the
            predictions in ``pred_field``. Must have the same type as
            ``pred_field``
        mistakenness_field ("mistakenness"): the field name to use to store the
            mistakenness value for each sample
        missing_field ("possible_missing): the field in which to store
            per-sample counts of potential missing detections. Only applicable
            for :class:`fiftyone.core.labels.Detections` labels
        spurious_field ("possible_spurious): the field in which to store
            per-sample counts of potential spurious detections. Only applicable
            for :class:`fiftyone.core.labels.Detections` labels
        use_logits (True): whether to use logits (True) or confidence (False)
            to compute mistakenness. Logits typically yield better results,
            when they are available
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
    # See the docstring above for additional handling of missing and spurious
    # detections.
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
            pred_label, label = _get_data(
                sample, pred_field, label_field, use_logits
            )

            if isinstance(pred_label, fol.Detections):
                possible_spurious = 0
                possible_missing = 0
                missing_detections = {}
                sample_mistakenness = []
                pred_map = {}
                for pred_det in pred_label.detections:
                    pred_map[pred_det.id] = pred_det
                    gt_id = pred_det[label_field + "_eval"]["matches"][
                        _DETECTION_IOU_STR
                    ]["gt_id"]
                    conf = pred_det.confidence
                    if gt_id == -1 and conf > _MISSED_CONFIDENCE_THRESHOLD:
                        pred_det[missing_field] = True
                        possible_missing += 1
                        missing_detections[pred_det.id] = pred_det

                for gt_det in label.detections:
                    # Avoid adding the same predictions again upon multiple
                    # runs of this method
                    if gt_det.has_field(missing_field):
                        if gt_det.id in missing_detections:
                            del missing_detections[gt_det.id]

                        continue

                    matches = gt_det[pred_field + "_eval"]["matches"]
                    pred_id = matches[_DETECTION_IOU_STR]["pred_id"]
                    iou = matches[_DETECTION_IOU_STR]["iou"]
                    if pred_id == -1:
                        gt_det[spurious_field] = True
                        possible_spurious += 1

                    else:
                        pred_det = pred_map[pred_id]
                        m = float(gt_det.label == pred_det.label)
                        if use_logits:
                            mistakenness_class = _compute_mistakenness_class(
                                pred_det.logits, m
                            )
                            mistakenness_loc = _compute_mistakenness_loc(
                                pred_det.logits, iou
                            )
                        else:
                            mistakenness_class = _compute_mistakenness_class_conf(
                                pred_det.confidence, m
                            )
                            mistakenness_loc = _compute_mistakenness_loc_conf(
                                pred_det.confidence, iou
                            )

                        gt_det[mistakenness_field + "_loc"] = mistakenness_loc
                        gt_det[mistakenness_field] = mistakenness_class
                        sample_mistakenness.append(mistakenness_class)

                label.detections += list(missing_detections.values())

                if sample_mistakenness:
                    sample[mistakenness_field] = np.max(sample_mistakenness)

                sample[missing_field] = possible_missing
                sample[spurious_field] = possible_spurious

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

                if use_logits:
                    mistakenness = _compute_mistakenness_class(
                        pred_label.logits, m
                    )
                else:
                    mistakenness = _compute_mistakenness_class_conf(
                        pred_label.confidence, m
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


def _compute_mistakenness_class_conf(confidence, m):
    # constrain m to either 1 (incorrect) or -1 (correct)
    m = m * -2.0 + 1.0

    mistakenness = (m * confidence + 1.0) / 2.0

    return mistakenness


def _compute_mistakenness_loc_conf(confidence, iou):
    # i = 0 for high iou, i = 1 for low iou
    i = (1.0 / (1.0 - _DETECTION_IOU)) * (1.0 - iou)

    # c = 0 for low confidence, c = 1 for high confidence
    c = confidence

    # mistakenness = i when c = i, mistakenness = 0.5 if c = 0
    # mistakenness is higher with lower IoU and closer to 0 or 1 with higher
    # confidence
    mistakenness = (c * ((2.0 * i) - 1.0) + 1.0) / 2.0

    return mistakenness


def _get_data(sample, pred_field, label_field, use_logits):
    pred_label, label = fbu.get_fields(
        sample,
        (pred_field, label_field),
        allowed_types=_ALLOWED_TYPES,
        same_type=True,
        allow_none=False,
    )

    if isinstance(pred_label, fol.Detections):
        for det in pred_label.detections:
            # We always need confidence for detections
            if det.confidence is None:
                raise ValueError(
                    "Detection '%s' in Sample '%s' field '%s' has no "
                    "confidence" % (det.id, sample.id, pred_field)
                )

            if use_logits and det.logits is None:
                raise ValueError(
                    "Detection '%s' in Sample '%s' field '%s' has no "
                    "logits" % (det.id, sample.id, pred_field)
                )

    elif use_logits:
        if pred_label.logits is None:
            raise ValueError(
                "Sample '%s' field '%s' has no logits"
                % (sample.id, pred_field)
            )

    else:
        if pred_label.confidence is None:
            raise ValueError(
                "Sample '%s' field '%s' has no confidence"
                % (sample.id, pred_field)
            )

    return pred_label, label