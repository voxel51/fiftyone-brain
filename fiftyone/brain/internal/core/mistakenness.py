"""
Methods that compute insights related to the chance that a label is a mistake.

| Copyright 2017-2021, Voxel51, Inc.
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
import fiftyone.core.validation as fov

import fiftyone.utils.eval as foue

from fiftyone.utils.eval.base import (
    Evaluation,
    EvaluationInfo,
    EvaluationConfig,
    save_evaluation_info,
    validate_evaluation,
)

from fiftyone import ViewField as F


logger = logging.getLogger(__name__)


_ALLOWED_TYPES = (fol.Classification, fol.Classifications, fol.Detections)
_MISSED_CONFIDENCE_THRESHOLD = 0.95
_DETECTION_IOU = 0.5
_DETECTION_IOU_STR = str(_DETECTION_IOU).replace(".", "_")


def compute_mistakenness(
    samples,
    pred_field,
    label_field,
    mistakenness_field,
    missing_field,
    spurious_field,
    use_logits,
    copy_missing,
):
    """See ``fiftyone/brain/__init__.py``."""

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

    fov.validate_collection_label_fields(
        samples, (pred_field, label_field), _ALLOWED_TYPES, same_type=True
    )

    frame_pred_field, is_frame_field = samples._handle_frame_field(pred_field)

    if is_frame_field:
        is_det = "Detection" in str(
            samples.get_frame_field_schema()[frame_pred_field]
        )

    else:
        is_det = "Detection" in str(samples.get_field_schema()[pred_field])

    config = MistakennessEvaluationConfig(
        mistakenness_field=mistakenness_field,
        missing_field=missing_field,
        spurious_field=spurious_field,
        use_logits=use_logits,
        copy_missing=copy_missing,
        is_detection=is_det,
    )
    eval_info = EvaluationInfo(
        mistakenness_field, pred_field, label_field, config
    )
    validate_evaluation(samples, eval_info)
    eval_method = config.build()

    pred_field, processing_frames = samples._handle_frame_field(pred_field)
    label_field, _ = samples._handle_frame_field(label_field)

    if not processing_frames:
        iter_samples = samples.select_fields([label_field, pred_field])
    else:
        iter_samples = samples

    det_eval_key = None

    if samples and is_det:
        # Find a temporary eval key for detections that isn't already in use
        # This will be deleted at the end of this method
        det_eval_key = "fob_det_eval"
        if det_eval_key in samples.list_evaluations():
            det_eval_key += "_"

        foue.evaluate_detections(
            samples,
            pred_field,
            label_field,
            eval_key=det_eval_key,
            classwise=False,
            iou=_DETECTION_IOU,
        )

    logger.info("Computing mistakenness...")
    with fou.ProgressBar() as pb:
        for sample in pb(iter_samples):
            if processing_frames:
                images = sample.frames.values()
            else:
                images = [sample]

            sample_mistakenness = []
            sample_miss = 0
            sample_spur = 0
            for image in images:
                (
                    img_mistakenness,
                    img_mis,
                    img_spur,
                ) = eval_method.evaluate_image(
                    image, pred_field, label_field, det_eval_key=det_eval_key
                )
                sample_mistakenness.append(img_mistakenness)
                if is_det:
                    sample_miss += img_mis
                    sample_spur += img_spur

                if processing_frames:
                    image[config.mistakenness_field] = img_mistakenness

                    if is_det:
                        image[config.missing_field] = img_mis
                        image[config.spurious_field] = img_spur

            sample[config.mistakenness_field] = np.max(sample_mistakenness)
            if is_det:
                sample[config.missing_field] = sample_miss
                sample[config.spurious_field] = sample_spur

            sample.save()

    samples.delete_evaluation(det_eval_key)
    save_evaluation_info(samples, eval_info)
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
    pred_label, label = fov.get_fields(
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


def _compute_detection_mistakenness(
    sample_or_frame, pred_field, gt_field, config, det_eval_key,
):

    missing_field = config.missing_field
    spurious_field = config.spurious_field
    mistakenness_field = config.mistakenness_field
    copy_missing = config.copy_missing
    use_logits = config.use_logits

    pred_label, gt_label = _get_data(
        sample_or_frame, pred_field, gt_field, config.use_logits
    )

    possible_spurious = 0
    possible_missing = 0
    missing_detections = {}
    image_mistakenness = []
    pred_map = {}
    for pred_det in pred_label.detections:
        pred_map[pred_det.id] = pred_det
        gt_id = pred_det[det_eval_key + "_id"]
        conf = pred_det.confidence
        if gt_id == "" and conf > _MISSED_CONFIDENCE_THRESHOLD:
            # Unmached FP with high conf are missing
            pred_det[missing_field] = True
            possible_missing += 1
            missing_detections[pred_det.id] = pred_det

    for gt_det in gt_label.detections:
        # Avoid adding the same unmatched FP predictions to gt
        # again upon multiple runs of this method
        if copy_missing and gt_det.has_field(missing_field):
            if gt_det.id in missing_detections:
                del missing_detections[gt_det.id]

            continue

        pred_id = gt_det[det_eval_key + "_id"]
        iou = gt_det[det_eval_key + "_iou"]
        if pred_id == "":
            # FN may be spurious
            gt_det[spurious_field] = True
            possible_spurious += 1

        else:
            # For matched FP, compute mistakenness
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
            image_mistakenness.append(mistakenness_class)

    if copy_missing:
        gt_label.detections += list(missing_detections.values())

    if image_mistakenness:
        mistakenness = np.max(image_mistakenness)
    else:
        mistakenness = -1

    return mistakenness, possible_missing, possible_spurious


def _compute_classification_mistakenness(
    sample_or_frame, pred_field, gt_field, config,
):

    mistakenness_field = config.mistakenness_field
    use_logits = config.use_logits

    pred_label, gt_label = _get_data(
        sample_or_frame, pred_field, gt_field, use_logits
    )

    if isinstance(pred_label, fol.Classifications):
        # For multilabel problems, all labels must match
        pred_labels = set(c.label for c in pred_label.classifications)
        gt_labels = set(c.label for c in gt_label.classifications)
        m = float(pred_labels == gt_labels)
    else:
        m = float(pred_label.label == gt_label.label)

    if use_logits:
        mistakenness = _compute_mistakenness_class(pred_label.logits, m)
    else:
        mistakenness = _compute_mistakenness_class_conf(
            pred_label.confidence, m
        )

    return mistakenness, None, None


class MistakennessEvaluationConfig(EvaluationConfig):
    """Base class for configuring :class:`MistakennessEvaluation` instances.
    Args:

    """

    def __init__(
        self,
        mistakenness_field="mistakenness",
        missing_field="possible_missing",
        spurious_field="possible_spurious",
        use_logits=True,
        copy_missing=False,
        is_detection=False,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.is_detection = is_detection
        self.mistakenness_field = mistakenness_field
        if is_detection:
            self.missing_field = missing_field
            self.spurious_field = spurious_field
            self.copy_missing = copy_missing

        self.use_logits = use_logits

    @property
    def method(self):
        return "mistakenness"


class MistakennessEvaluation(Evaluation):
    """Evaluation mistakenness of labels
    Args:
        config: a :class:`MistakennessEvaluationConfig`
    """

    def __init__(self, config):
        super().__init__(config)

    def evaluate_image(
        self, sample_or_frame, pred_field, gt_field, det_eval_key=None
    ):
        """Evaluates the ground truth and predicted objects in an image.
        Args:
            sample_or_frame: a :class:`fiftyone.core.Sample` or
                :class:`fiftyone.core.frame.Frame`
            pred_field: the name of the field containing the predicted
                :class:`fiftyone.core.labels.Detections` instances
            gt_field: the name of the field containing the ground truth
                :class:`fiftyone.core.labels.Detections` instances
            eval_key (None): an evaluation key for this evaluation
        Returns:
            a list of ``(mistakenness, possible_spurious, possible_missing)`` tuples
        """
        if self.config.is_detection:
            if not det_eval_key:
                raise ValueError(
                    "You mist pass in a det_eval_key when computing mistakenness on Detections."
                )
            # Detections mistakenness
            mistakes = _compute_detection_mistakenness(
                sample_or_frame,
                pred_field,
                gt_field,
                self.config,
                det_eval_key,
            )
        else:
            # Classification and Classifications mistakennes
            mistakes = _compute_classification_mistakenness(
                sample_or_frame, pred_field, gt_field, self.config
            )

        return mistakes

    def get_fields(self, samples, mistakenness_field):
        eval_info = samples.get_evaluation_info(mistakenness_field)

        pred_field = eval_info.pred_field
        gt_field = eval_info.gt_field

        eval_fields = [
            eval_info.config.mistakenness_field,
        ]
        if eval_info.config.is_detection:
            eval_fields.extend(
                [
                    eval_info.config.spurious_field,
                    eval_info.config.missing_field,
                    "%s.detections.%s"
                    % (eval_info.pred_field, eval_info.config.missing_field),
                    "%s.detections.%s"
                    % (
                        eval_info.gt_field,
                        eval_info.config.mistakenness_field,
                    ),
                    "%s.detections.%s_loc"
                    % (
                        eval_info.pred_field,
                        eval_info.config.mistakenness_field,
                    ),
                ]
            )
            if eval_info.config.copy_missing:
                eval_fields.append(
                    "%s.detections.%s"
                    % (eval_info.gt_field, eval_info.config.missing_field)
                )

        if samples._is_frame_field(eval_info.gt_field):
            eval_fields.append(
                "frames.%s" % eval_info.config.mistakenness_field,
            )
            if eval_info.config.is_detection:
                eval_fields.extend(
                    [
                        "frames.%s" % eval_info.config.spurious_field,
                        "frames.%s" % eval_info.config.missing_field,
                    ]
                )

        return eval_fields

    def cleanup(self, samples, mistakenness_field):
        eval_info = samples.get_evaluation_info(mistakenness_field)

        pred_field, is_frame_field = samples._handle_frame_field(
            eval_info.pred_field
        )
        gt_field, _ = samples._handle_frame_field(eval_info.gt_field)

        fields = [
            eval_info.config.mistakenness_field,
        ]
        if eval_info.config.is_detection:
            fields.extend(
                [
                    eval_info.config.spurious_field,
                    eval_info.config.missing_field,
                    "%s.detections.%s"
                    % (eval_info.pred_field, eval_info.config.missing_field),
                    "%s.detections.%s"
                    % (
                        eval_info.gt_field,
                        eval_info.config.mistakenness_field,
                    ),
                    "%s.detections.%s_loc"
                    % (
                        eval_info.gt_field,
                        eval_info.config.mistakenness_field,
                    ),
                ]
            )
            if eval_info.config.copy_missing:
                # Remove the detections that were copied from predictions to
                # gt field
                missing_gt_field = gt_field
                if is_frame_field:
                    missing_gt_field = "frames." + missing_gt_field

                samples._dataset.filter_labels(
                    missing_gt_field,
                    ~F(eval_info.config.missing_field).exists(),
                ).save()

        if is_frame_field:
            sample_fields = [eval_info.config.mistakenness_field]

            if eval_info.config.is_detection:
                sample_fields.extend(
                    [
                        eval_info.config.spurious_field,
                        eval_info.config.missing_field,
                    ]
                )
            samples._dataset.delete_sample_fields(sample_fields)
            samples._dataset.delete_frame_fields(fields)
        else:
            samples._dataset.delete_sample_fields(fields)
