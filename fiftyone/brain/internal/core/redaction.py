"""
Redaction methods.

| Copyright 2017-2025, Voxel51, Inc.
| `voxel51.com <https://voxel51.com/>`_
"""
import os
import logging
import shutil
from typing import Tuple

import numpy as np
import cv2

import fiftyone as fo
import fiftyone.core.storage as fos
import fiftyone.core.brain as fob
import fiftyone.core.labels as fol
import fiftyone.core.validation as fov


logger = logging.getLogger(__name__)


_ALLOWED_TYPES = (
    fol.Detections,
    fol.Polylines,
    fol.Keypoints,
    fol.TemporalDetections,
)

_GAUSSIAN_BLUR_KERNEL_SIZE = lambda w: (w // 10) * 2 + 1  # ensure ksize is odd
_STACK_BLUR_KERNEL_SIZE = lambda w: min(w, 50) * 2 + 1  # ensure ksize is odd


def create_redaction(
    samples,
    label_field,
    label_classes,
    redaction_type,
    redaction_method,
    force_recreate,
    create_as_new_sample,
    num_workers,
    progress,
):
    """See ``fiftyone/brain/__init__.py``."""

    #
    # Algorithm
    #
    # TODO(neeraja): Add algorithm description
    #

    fov.validate_collection(samples)
    fov.validate_collection_label_fields(samples, label_field, _ALLOWED_TYPES)

    config = RedactionConfig(
        label_field,
        label_classes,
        redaction_type,
        redaction_method,
        force_recreate,
        create_as_new_sample,
    )
    brain_key = config.redaction_field
    brain_method = config.build()
    brain_method.ensure_requirements()
    brain_method.register_run(samples, brain_key, cleanup=False)
    brain_method.register_samples(samples)

    select_fields = [
        label_field,
        config.redaction_field,
        "original_sample_id",
        "redacted_sample_ids",
    ]
    select_fields = [
        ff for ff in select_fields if samples.get_field(ff) is not None
    ]
    view = samples.select_fields(select_fields)

    logger.info("Computing redaction...")
    view._dataset.persistent = True
    for sample in view.iter_samples(progress=progress):
        brain_method.process_sample(sample)
        sample.save()

    if not create_as_new_sample:
        logger.info(
            f"Adding redacted media field: {config.redaction_field}_filepath"
        )
        if (
            f"{config.redaction_field}_filepath"
            not in view._dataset.app_config.media_fields
        ):
            view._dataset.app_config.media_fields.append(
                f"{config.redaction_field}_filepath"
            )

    view._dataset.save()
    brain_method.save_run_results(samples, brain_key, None)

    logger.info("Redaction computation complete")


# @todo move to `fiftyone/brain/redaction.py`
# Don't do this hastily; `get_brain_info()` on existing datasets has this
# class's full path in it and may need migration
class RedactionConfig(fob.BrainMethodConfig):
    def __init__(
        self,
        label_field,
        label_classes,
        redaction_type,
        redaction_method,
        force_recreate,
        create_as_new_sample,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.label_field = label_field
        self.label_classes = [
            label_name.strip()
            for label_name in label_classes.strip("[").strip("]").split(",")
        ]
        self.redaction_type = redaction_type
        self.redaction_method = redaction_method
        self.force_recreate = force_recreate
        self.create_as_new_sample = create_as_new_sample
        redaction_labels = "_".join(label_classes)
        redaction_field = f"redacted_{label_field}_{redaction_labels}_{redaction_type}_{redaction_method}"
        self.redaction_field = redaction_field

    @property
    def type(self):
        return "redaction"

    @property
    def method(self):
        # TODO(neeraja): is this just a unique ID?
        return self.redaction_field


class Redaction(fob.BrainMethod):
    def __init__(self, config):
        super().__init__(config)
        self.label_field = None
        self.label_classes = None
        self.redaction_type = None
        self.redaction_method = None
        self.force_recreate = None
        self.create_as_new_sample = None
        self.redaction_field = None

    def ensure_requirements(self):
        pass

    def register_samples(self, samples):
        self.label_field, _ = samples._handle_frame_field(
            self.config.label_field
        )
        self.label_type = samples._get_label_field_type(
            self.config.label_field
        )

    def process_sample(self, sample):
        redacted_media_path = _get_outpath(
            sample.filepath,
            rel_output_dir=self.redaction_field,
        )
        if self.create_as_new_sample:
            if (
                ("redacted_sample_ids" in sample)
                and (self.redaction_field in sample["redacted_sample_ids"])
                and (
                    os.path.exists(
                        sample._dataset[
                            sample["redacted_sample_ids"][self.redaction_field]
                        ]["filepath"]
                    )
                )
            ):
                logger.debug(
                    f"Redaction already exists for sample {sample.id} --> {sample['redacted_sample_ids'][self.redaction_field]}"
                )
                if self.force_recreate:
                    sample._dataset.delete_sample(
                        sample["redacted_sample_ids"][self.redaction_field]
                    )
                    sample["redacted_sample_ids"].pop(self.redaction_field)
                else:
                    return
            if self.redaction_field in sample.tags:
                logger.debug(
                    f"This is a redaction of sample {sample['original_sample_id']}"
                )
                return
        else:
            if (f"{self.redaction_field}_filepath" in sample) and (
                os.path.exists(sample[f"{self.redaction_field}_filepath"])
            ):
                logger.debug(
                    f"Redaction already exists for sample {sample.id}"
                )
                if self.force_recreate:
                    sample[f"{self.redaction_field}_filepath"] = None
                    sample.tags.remove(self.redaction_field)
                else:
                    return

        shutil.copy(sample.filepath, redacted_media_path)

        processing_frames = sample._is_frame_field(self.label_field)
        if processing_frames:
            # TODO(neeraja): handle this for frame fields
            # images = sample.frames.values()
            raise NotImplementedError("Frame fields are not supported yet")
        else:
            self.redact_file_at(redacted_media_path, sample[self.label_field])

        if self.create_as_new_sample:
            redacted_sample = sample.copy()
            redacted_sample[self.label_field] = self.filter_detections(
                sample[self.label_field], keep_matching=False
            )
            redacted_sample["filepath"] = redacted_media_path
            redacted_sample.tags.append(self.redaction_field)
            redacted_sample["original_sample_id"] = sample.id
            if "redacted_sample_ids" not in sample:
                sample["redacted_sample_ids"] = {}
            sample["redacted_sample_ids"][
                self.redaction_field
            ] = redacted_sample.id
            logger.info(
                f"Adding redacted sample at filepath: {redacted_media_path}"
            )
            sample._dataset.add_sample(redacted_sample)
        else:
            # add the redacted filepath to the sample
            sample.tags.append(self.redaction_field)
            sample[f"{self.redaction_field}_filepath"] = redacted_media_path

        sample.save()
        return

    def redact_file_at(self, redacted_path, detections_object):
        image = cv2.imread(redacted_path)
        if self.redaction_method == "gaussian_blur":
            ksize = int(
                _GAUSSIAN_BLUR_KERNEL_SIZE(min(image.shape[0], image.shape[1]))
            )
            redacted_image = cv2.GaussianBlur(image, (ksize, ksize), 0)
        elif self.redaction_method == "stack_blur":
            ksize = int(
                _STACK_BLUR_KERNEL_SIZE(min(image.shape[0], image.shape[1]))
            )
            redacted_image = cv2.stackBlur(image, (ksize, ksize))
        elif self.redaction_method == "mask":
            redacted_image = np.zeros_like(image)
        else:
            raise ValueError(
                f"Unimplemented redaction method: {self.redaction_method}"
            )

        for detection in self.filter_detections(detections_object).detections:
            x1, y1, x2, y2 = get_corners_from_bbox(
                detection.bounding_box, image.shape
            )
            mask = detection.get_mask()
            if self.redaction_type == "segmentation_mask" and mask is not None:
                mask = fit_mask_to_bbox(mask, (y2 - y1, x2 - x1))
                image[y1:y2, x1:x2][mask] = redacted_image[y1:y2, x1:x2][mask]
            elif self.redaction_type == "bounding_box":
                image[y1:y2, x1:x2] = redacted_image[y1:y2, x1:x2]
            elif self.redaction_type == "segmentation_mask" and mask is None:
                logger.warning(f"No mask found for detection: {detection}")
                continue
            else:
                raise ValueError(
                    f"Unknown redaction type: {self.redaction_type}"
                )

        cv2.imwrite(redacted_path, image)

    def filter_detections(
        self, detections_object: fol.Detections, keep_matching: bool = True
    ) -> fol.Detections:
        """
        Filters the detections object to keep only the detections that match the redaction labels.
        Args:
            detections_object: fol.Detections object
            keep_matching: bool to keep matching detections (True) or remove matching detections (False)
        Returns:
            fo.Detections object
        """
        filtered_detections = fo.Detections()
        if (not hasattr(detections_object, "detections")) or (
            detections_object.detections is None
        ):
            return filtered_detections
        for detection in detections_object.detections:
            if keep_matching and (
                detection.label
                in self.label_classes  # pylint: disable=unsupported-membership-test
            ):
                filtered_detections.detections.append(detection)
            elif not keep_matching and (
                detection.label
                not in self.label_classes  # pylint: disable=unsupported-membership-test
            ):
                filtered_detections.detections.append(detection)
        return filtered_detections

    def get_fields(self, samples, brain_key):
        fields = [
            self.config.label_field,
            self.config.redaction_field,
            "original_sample_id",
            "redacted_sample_ids",
        ]

        if samples._is_frame_field(self.config.label_field):
            fields.append(samples._FRAMES_PREFIX + self.config.redaction_field)

        return fields

    def cleanup(self, samples, brain_key):
        label_field = self.config.label_field
        redaction_field = self.config.redaction_field

        samples._dataset.delete_sample_fields(redaction_field, error_level=1)

        if samples._is_frame_field(label_field):
            samples._dataset.delete_frame_fields(
                redaction_field, error_level=1
            )

    def _validate_run(self, samples, brain_key, existing_info):
        self._validate_fields_match(brain_key, "label_field", existing_info)
        self._validate_fields_match(
            brain_key, "redaction_field", existing_info
        )


def fit_mask_to_bbox(
    mask: np.ndarray, bbox_size: Tuple[int, int]
) -> np.ndarray:
    """
    Pads or crops the mask to the bounding box size.
    Args:
        mask: np.ndarray of shape (mask_height, mask_width)
        bbox_size: Tuple[int, int] of the bounding box size (height, width)
    Returns:
        np.ndarray of shape (height, width)
    """
    return np.pad(
        mask,
        [
            (0, max(0, bbox_size[0] - mask.shape[0])),
            (0, max(0, bbox_size[1] - mask.shape[1])),
        ],
    )[: bbox_size[0], : bbox_size[1]]


def get_corners_from_bbox(
    bbox: Tuple[float, float, float, float], image_shape: Tuple[int, int]
) -> Tuple[int, int, int, int]:
    """
    Returns the corners of the bounding box in image coordinates.

    Args:
        bbox: Tuple[float, float, float, float] of the bounding box (top-left-x, top-left-y, width, height)
        image_shape: Tuple[int, int] of the image shape (height, width)

    Returns:
        Tuple[int, int, int, int] of the corners of the bounding box
        in image coordinates (left, top, right, bottom) i.e. (x1, y1, x2, y2)
    """
    x1 = int(bbox[0] * image_shape[1])
    y1 = int(bbox[1] * image_shape[0])
    x2 = int((bbox[0] + bbox[2]) * image_shape[1])
    y2 = int((bbox[1] + bbox[3]) * image_shape[0])
    return (x1, y1, x2, y2)


def _get_outpath(inpath, rel_output_dir):
    dir_path = os.path.dirname(inpath)
    dir_path = fos.normalize_path(dir_path)
    filename = os.path.basename(inpath)
    return os.path.join(dir_path, rel_output_dir, filename)
