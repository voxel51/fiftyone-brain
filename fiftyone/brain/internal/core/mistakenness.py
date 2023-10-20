"""
Mistakenness methods.

| Copyright 2017-2023, Voxel51, Inc.
| `voxel51.com <https://voxel51.com/>`_
|
"""
import logging
from math import exp

import numpy as np
from scipy.special import softmax
from scipy.stats import entropy

from fiftyone import ViewField as F
import fiftyone.core.brain as fob
import fiftyone.core.labels as fol
import fiftyone.core.media as fom
import fiftyone.core.utils as fou
import fiftyone.core.validation as fov


logger = logging.getLogger(__name__)


_ALLOWED_TYPES = (
    fol.Classification,
    fol.Classifications,
    fol.Detections,
    fol.Polylines,
    fol.Keypoints,
    fol.TemporalDetections,
)
_MISSED_CONFIDENCE_THRESHOLD = 0.95
_DETECTION_IOU = 0.5


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

    fov.validate_collection_label_fields(
        samples, (pred_field, label_field), _ALLOWED_TYPES, same_type=True
    )

    if samples.media_type == fom.VIDEO:
        mistakenness_field, _ = samples._handle_frame_field(mistakenness_field)
        missing_field, _ = samples._handle_frame_field(missing_field)
        spurious_field, _ = samples._handle_frame_field(spurious_field)

    is_objects = samples._is_label_field(
        pred_field,
        (fol.Detections, fol.Polylines, fol.Keypoints, fol.TemporalDetections),
    )
    if is_objects:
        eval_key = _make_eval_key(samples, mistakenness_field)
        config = DetectionMistakennessConfig(
            pred_field,
            label_field,
            mistakenness_field,
            missing_field,
            spurious_field,
            use_logits,
            copy_missing,
            eval_key,
        )
    else:
        eval_key = None
        config = ClassificationMistakennessConfig(
            pred_field, label_field, mistakenness_field, use_logits
        )

    brain_key = mistakenness_field
    brain_method = config.build()
    brain_method.ensure_requirements()
    brain_method.register_run(samples, brain_key)
    brain_method.register_samples(samples)

    if is_objects:
        samples.evaluate_detections(
            pred_field,
            gt_field=label_field,
            eval_key=eval_key,
            classwise=False,
            iou=_DETECTION_IOU,
        )

    view = samples.select_fields([label_field, pred_field])
    processing_frames = samples._is_frame_field(label_field)

    logger.info("Computing mistakenness...")
    for sample in view.iter_samples(progress=True):
        if processing_frames:
            images = sample.frames.values()
        else:
            images = [sample]

        sample_mistakenness = []
        num_missing = 0
        num_spurious = 0
        for image in images:
            if is_objects:
                (
                    img_mistakenness,
                    img_missing,
                    img_spurious,
                ) = brain_method.process_image(image, eval_key)

                num_missing += img_missing
                num_spurious += img_spurious
                if processing_frames:
                    image[missing_field] = img_missing
                    image[spurious_field] = img_spurious
            else:
                img_mistakenness = brain_method.process_image(image)

            if img_mistakenness is not None:
                sample_mistakenness.append(img_mistakenness)

            if processing_frames:
                image[mistakenness_field] = img_mistakenness

        if sample_mistakenness:
            sample[mistakenness_field] = np.max(sample_mistakenness)
        else:
            sample[mistakenness_field] = None

        if is_objects:
            sample[missing_field] = num_missing
            sample[spurious_field] = num_spurious

        sample.save()

    if eval_key is not None:
        samples.delete_evaluation(eval_key)

    brain_method.save_run_results(samples, brain_key, None)

    logger.info("Mistakenness computation complete")


# @todo move to `fiftyone/brain/mistakenness.py`
# Don't do this hastily; `get_brain_info()` on existing datasets has this
# class's full path in it and may need migration
class MistakennessMethodConfig(fob.BrainMethodConfig):
    def __init__(self, pred_field, label_field, mistakenness_field, **kwargs):
        super().__init__(**kwargs)
        self.pred_field = pred_field
        self.label_field = label_field
        self.mistakenness_field = mistakenness_field

    @property
    def type(self):
        return "mistakenness"


class MistakennessMethod(fob.BrainMethod):
    def __init__(self, config):
        super().__init__(config)
        self.pred_field = None
        self.label_field = None
        self.label_type = None

    def ensure_requirements(self):
        pass

    def register_samples(self, samples):
        self.pred_field, _ = samples._handle_frame_field(
            self.config.pred_field
        )
        self.label_field, _ = samples._handle_frame_field(
            self.config.label_field
        )
        self.label_type = samples._get_label_field_type(self.config.pred_field)

    def _validate_run(self, samples, brain_key, existing_info):
        self._validate_fields_match(brain_key, "pred_field", existing_info)
        self._validate_fields_match(brain_key, "label_field", existing_info)
        self._validate_fields_match(
            brain_key, "mistakenness_field", existing_info
        )


# @todo move to `fiftyone/brain/mistakenness.py`
# Don't do this hastily; `get_brain_info()` on existing datasets has this
# class's full path in it and may need migration
class ClassificationMistakennessConfig(MistakennessMethodConfig):
    def __init__(
        self, pred_field, label_field, mistakenness_field, use_logits, **kwargs
    ):
        super().__init__(pred_field, label_field, mistakenness_field, **kwargs)
        self.use_logits = use_logits

    @property
    def method(self):
        return "classification"


class ClassificationMistakenness(MistakennessMethod):
    def process_image(self, sample_or_frame):
        use_logits = self.config.use_logits

        pred_label, gt_label = _get_data(
            sample_or_frame, self.pred_field, self.label_field, use_logits
        )

        if pred_label is None and gt_label is None:
            return None

        if pred_label is None or gt_label is None:
            m = 1.0
        elif isinstance(pred_label, fol.Classifications):
            # For multilabel problems, all labels must match
            pred_labels = set(c.label for c in pred_label.classifications)
            gt_labels = set(c.label for c in gt_label.classifications)
            m = float(pred_labels == gt_labels)
        else:
            m = float(pred_label.label == gt_label.label)

        if pred_label is None:
            mistakenness = 1.0
        elif use_logits:
            mistakenness = _compute_mistakenness_class(pred_label.logits, m)
        else:
            mistakenness = _compute_mistakenness_class_conf(
                pred_label.confidence, m
            )

        return mistakenness

    def get_fields(self, samples, brain_key):
        pred_field = self.config.pred_field
        label_field = self.config.label_field
        mistakenness_field = self.config.mistakenness_field

        fields = [pred_field, label_field, mistakenness_field]

        if samples._is_frame_field(label_field):
            fields.append(samples._FRAMES_PREFIX + mistakenness_field)

        return fields

    def cleanup(self, samples, brain_key):
        label_field = self.config.label_field
        mistakenness_field = self.config.mistakenness_field

        samples._dataset.delete_sample_fields(
            mistakenness_field, error_level=1
        )

        if samples._is_frame_field(label_field):
            samples._dataset.delete_frame_fields(
                mistakenness_field, error_level=1
            )


# @todo move to `fiftyone/brain/mistakenness.py`
# Don't do this hastily; `get_brain_info()` on existing datasets has this
# class's full path in it and may need migration
class DetectionMistakennessConfig(MistakennessMethodConfig):
    def __init__(
        self,
        pred_field,
        label_field,
        mistakenness_field,
        missing_field,
        spurious_field,
        use_logits,
        copy_missing,
        eval_key,
        **kwargs
    ):
        super().__init__(pred_field, label_field, mistakenness_field, **kwargs)
        self.missing_field = missing_field
        self.spurious_field = spurious_field
        self.use_logits = use_logits
        self.copy_missing = copy_missing
        self.eval_key = eval_key

    @property
    def method(self):
        return "detection"


class DetectionMistakenness(MistakennessMethod):
    def process_image(self, sample_or_frame, eval_key):
        missing_field = self.config.missing_field
        spurious_field = self.config.spurious_field
        mistakenness_field = self.config.mistakenness_field
        copy_missing = self.config.copy_missing
        use_logits = self.config.use_logits

        pred_label, gt_label = _get_data(
            sample_or_frame, self.pred_field, self.label_field, use_logits
        )

        list_field = self.label_type._LABEL_LIST_FIELD

        if pred_label is None:
            pred_label = self.label_type()

        if gt_label is None:
            gt_label = self.label_type()

        num_spurious = 0
        num_missing = 0
        missing_objects = {}
        image_mistakenness = []
        pred_map = {}
        for pred_obj in pred_label[list_field]:
            pred_map[pred_obj.id] = pred_obj
            gt_id = pred_obj[eval_key + "_id"]
            conf = pred_obj.confidence
            if gt_id == "" and conf > _MISSED_CONFIDENCE_THRESHOLD:
                # Unmached FP with high confidence are missing
                pred_obj[missing_field] = True
                num_missing += 1
                missing_objects[pred_obj.id] = pred_obj

        for gt_obj in gt_label[list_field]:
            # Avoid adding the same unmatched FP predictions upon multiple runs
            # of this method
            if copy_missing and gt_obj.has_field(missing_field):
                if gt_obj.id in missing_objects:
                    del missing_objects[gt_obj.id]

                continue

            pred_id = gt_obj[eval_key + "_id"]
            if pred_id == "":
                # FN may be spurious
                gt_obj[spurious_field] = True
                num_spurious += 1
            else:
                # For matched FP, compute mistakenness
                iou = gt_obj[eval_key + "_iou"]
                pred_obj = pred_map[pred_id]
                m = float(gt_obj.label == pred_obj.label)
                if use_logits:
                    mistakenness_class = _compute_mistakenness_class(
                        pred_obj.logits, m
                    )
                    mistakenness_loc = _compute_mistakenness_loc(
                        pred_obj.logits, iou
                    )
                else:
                    mistakenness_class = _compute_mistakenness_class_conf(
                        pred_obj.confidence, m
                    )
                    mistakenness_loc = _compute_mistakenness_loc_conf(
                        pred_obj.confidence, iou
                    )

                gt_obj[mistakenness_field] = mistakenness_class
                gt_obj[mistakenness_field + "_loc"] = mistakenness_loc
                image_mistakenness.append(mistakenness_class)

        if copy_missing:
            gt_label[list_field].extend(missing_objects.values())
            sample_or_frame[self.label_field] = gt_label

        if image_mistakenness:
            mistakenness = np.max(image_mistakenness)
        else:
            mistakenness = -1

        return mistakenness, num_missing, num_spurious

    def get_fields(self, samples, brain_key):
        pred_field = self.config.pred_field
        label_field = self.config.label_field
        mistakenness_field = self.config.mistakenness_field
        missing_field = self.config.missing_field
        spurious_field = self.config.spurious_field

        label_type = samples._get_label_field_type(pred_field)
        list_field = label_type._LABEL_LIST_FIELD

        fields = [
            mistakenness_field,
            missing_field,
            spurious_field,
            "%s.%s.%s" % (label_field, list_field, mistakenness_field),
            "%s.%s.%s_loc" % (label_field, list_field, mistakenness_field),
            "%s.%s.%s" % (pred_field, list_field, missing_field),
            "%s.%s.%s" % (label_field, list_field, spurious_field),
        ]

        if samples._is_frame_field(pred_field):
            fields.extend(
                [
                    samples._FRAMES_PREFIX + mistakenness_field,
                    samples._FRAMES_PREFIX + missing_field,
                    samples._FRAMES_PREFIX + spurious_field,
                ]
            )

        return fields

    def cleanup(self, samples, brain_key):
        pred_field = self.config.pred_field
        label_field = self.config.label_field
        mistakenness_field = self.config.mistakenness_field
        missing_field = self.config.missing_field
        spurious_field = self.config.spurious_field
        eval_key = self.config.eval_key

        label_type = samples._get_label_field_type(pred_field)
        list_field = label_type._LABEL_LIST_FIELD

        pred_field, is_frame_field = samples._handle_frame_field(pred_field)
        label_field, _ = samples._handle_frame_field(label_field)

        fields = [
            mistakenness_field,
            missing_field,
            spurious_field,
            "%s.%s.%s" % (label_field, list_field, mistakenness_field),
            "%s.%s.%s_loc" % (label_field, list_field, mistakenness_field),
            "%s.%s.%s" % (pred_field, list_field, missing_field),
            "%s.%s.%s" % (label_field, list_field, spurious_field),
        ]

        if self.config.copy_missing:
            # Remove objects that were added to `label_field`
            samples._dataset.filter_labels(
                self.config.label_field, F(missing_field).exists(False)
            ).save()

        if is_frame_field:
            samples._dataset.delete_sample_fields(
                [mistakenness_field, spurious_field, missing_field],
                error_level=1,
            )
            samples._dataset.delete_frame_fields(fields, error_level=1)
        else:
            samples._dataset.delete_sample_fields(fields, error_level=1)

        if eval_key in samples.list_evaluations():
            samples.delete_evaluation(eval_key)

    def _validate_run(self, samples, brain_key, existing_info):
        super()._validate_run(samples, brain_key, existing_info)
        self._validate_fields_match(brain_key, "missing_field", existing_info)
        self._validate_fields_match(brain_key, "spurious_field", existing_info)
        self._validate_fields_match(brain_key, "copy_missing", existing_info)


def _make_eval_key(samples, brain_key):
    existing_eval_keys = samples.list_evaluations()
    eval_key = brain_key + "_eval"
    if eval_key not in existing_eval_keys:
        return eval_key

    idx = 2
    while eval_key + str(idx) in existing_eval_keys:
        idx += 1

    return eval_key + str(idx)


def _get_data(sample, pred_field, label_field, use_logits):
    pred_label = sample[pred_field]
    label = sample[label_field]

    if pred_label is None:
        return pred_label, label

    if isinstance(pred_label, fol.Detections):
        for det in pred_label.detections:
            if det.confidence is None:
                raise ValueError(
                    "Detection '%s' in sample '%s' field '%s' has no "
                    "confidence" % (det.id, sample.id, pred_field)
                )
    elif isinstance(pred_label, fol.Polylines):
        for poly in pred_label.polylines:
            if poly.confidence is None:
                raise ValueError(
                    "Polyline '%s' in sample '%s' field '%s' has no "
                    "confidence" % (poly.id, sample.id, pred_field)
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
