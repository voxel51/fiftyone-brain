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

from eta.core.video import VideoProcessor

import fiftyone as fo
import fiftyone.core.storage as fos
import fiftyone.core.brain as fob
import fiftyone.core.fields as fof
import fiftyone.core.labels as fol
import fiftyone.core.validation as fov
from fiftyone import ViewField as F


logger = logging.getLogger(__name__)


_ALLOWED_TYPES = (
    fol.Detection,
    fol.Detections,
)

_GAUSSIAN_BLUR_KERNEL_SIZE = lambda w: (w // 10) * 2 + 1  # ensure ksize is odd
_STACK_BLUR_KERNEL_SIZE = lambda w: min(w, 50) * 2 + 1  # ensure ksize is odd


def create_redaction(
    samples,
    label_field,
    label_classes,
    redaction_type,
    redaction_method,
    redaction_field,
    progress,
):
    """See ``fiftyone/brain/__init__.py``."""

    #
    # Algorithm
    #
    # Generates a blurred/masked image based on the redaction_method.
    # Identifies areas of the image (based on the redaction_type = bounding_box/segmentation_mask)
    # corresponding to the label_classes from the label_field
    # and replaces the regions with the redaction_method image
    #

    fov.validate_collection(samples)
    fov.validate_collection_label_fields(samples, label_field, _ALLOWED_TYPES)

    config = RedactionConfig(
        label_field,
        label_classes,
        redaction_type,
        redaction_method,
        redaction_field,
    )
    brain_key = config.redaction_field
    brain_method = config.build()
    brain_method.ensure_requirements()
    brain_method.register_run(samples, brain_key, cleanup=False)
    brain_method.register_samples(samples)

    redacted_filepath_field = f"{config.redaction_field}_filepath"
    if samples.get_field(redacted_filepath_field) is None:
        samples._dataset.add_sample_field(
            redacted_filepath_field, fof.StringField
        )
    select_fields = [label_field, "filepath", redacted_filepath_field]
    view = samples.select_fields(select_fields)

    logger.info("Computing redaction...")

    view.set_values(redacted_filepath_field, view.values("filepath"))
    redaction_view = view.filter_labels(
        f"{config.label_field}.detections",
        F("label").is_in(config.label_classes),
        only_matches=True,
    )

    redaction_view.map_samples(
        brain_method.process_sample_fn,
    )

    logger.info(f"Adding redacted media field: {brain_key}_filepath")
    if f"{brain_key}_filepath" not in samples.app_config.media_fields:
        samples.app_config.media_fields.append(f"{brain_key}_filepath")
        samples.save()

    results = RedactionResults(samples, config, brain_key, brain_method)
    brain_method.save_run_results(samples, brain_key, results)

    logger.info("Redaction computation complete")
    return results


class RedactionConfig(fob.BrainMethodConfig):
    def __init__(
        self,
        label_field,
        label_classes,
        redaction_type,
        redaction_method,
        redaction_field,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.label_field = label_field
        self.label_classes = label_classes
        self.redaction_type = redaction_type
        self.redaction_method = redaction_method
        self.redaction_field = redaction_field
        self.validate_label_classes()
        self.make_redaction_field()

    def validate_label_classes(self):
        try:
            self.label_classes = list(map(str.strip, self.label_classes))
        except:
            raise ValueError(
                f"Invalid label classes type: {type(self.label_classes)}"
            )

    def make_redaction_field(self):
        redaction_field_string = f"redacted_{self.label_field.replace('.', '_')}_{self.redaction_type}_{self.redaction_method}"
        self.redaction_field = (
            redaction_field_string
            if self.redaction_field is None
            else self.redaction_field
        )

    @property
    def type(self):
        return "redaction"

    @property
    def method(self):
        return self.redaction_method


class Redaction(fob.BrainMethod):
    def __init__(self, config):
        super().__init__(config)
        self.label_type = None
        self.label_field = self.config.label_field
        self.label_classes = self.config.label_classes
        self.redaction_type = self.config.redaction_type
        self.redaction_method = self.config.redaction_method
        self.redaction_field = self.config.redaction_field
        self.processing_frames = False

    def ensure_requirements(self):
        pass

    def register_samples(self, samples):
        _, self.processing_frames = samples._handle_frame_field(
            self.label_field
        )
        self.label_type = samples._get_label_field_type(self.label_field)

    def process_sample_fn(self, sample):
        redacted_media_path = _get_outpath(
            sample.filepath,
            rel_output_dir=self.redaction_field,
        )
        fos.copy_file(sample.filepath, redacted_media_path)
        if self.processing_frames:
            self.redact_video_file_at(redacted_media_path, self.label_field)
        else:
            self.redact_image_file_at(redacted_media_path, self.label_field)

    def redact_image_file_at(self, redacted_path, detections_object):
        if not detections_object.detections:
            return

        image = cv2.imread(redacted_path)
        redacted_image = self._redact_entire_image(image)
        image = self._apply_redaction_to_image(
            image, redacted_image, detections_object
        )
        cv2.imwrite(redacted_path, image)

    def redact_video_file_at(self, redacted_path, detections_object_list):
        if all(
            [
                not detections_object.detections
                for detections_object in detections_object_list
            ]
        ):
            return

        suffix = redacted_path.split(".")[-1]
        temp_redacted_path = redacted_path.replace(
            f".{suffix}", f"_temp.{suffix}"
        )
        with VideoProcessor(
            redacted_path,
            out_video_path=temp_redacted_path,
            out_opts=[
                "-pix_fmt",
                "yuv420p",
                "-c:v",
                "libopenh264",  # this increases the video quality
            ],
        ) as vp:
            if vp.total_frame_count != len(detections_object_list):
                raise ValueError(
                    f"Number of video frames ({vp.total_frame_count}) does not match the number of detections ({len(detections_object_list)})"
                )
            for frame, detections_object in zip(vp, detections_object_list):
                redacted_image = self._redact_entire_image(frame)
                frame = self._apply_redaction_to_image(
                    frame, redacted_image, detections_object
                )
                vp.write(frame)
        fos.move_file(temp_redacted_path, redacted_path)

    def _redact_entire_image(self, image):
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
        return redacted_image

    def _apply_redaction_to_image(
        self, image, redacted_image, detections_object
    ):
        for detection in detections_object.detections:
            x1, y1, x2, y2 = get_corners_from_bbox(
                detection.bounding_box, image.shape
            )
            mask = detection.get_mask()
            if self.redaction_type == "segmentation_mask":
                if mask is None:
                    logger.warning(f"No mask found for detection: {detection}")
                    continue
                mask = fit_mask_to_bbox(mask, (y2 - y1, x2 - x1))
                image[y1:y2, x1:x2][mask] = redacted_image[y1:y2, x1:x2][mask]
            elif self.redaction_type == "bounding_box":
                image[y1:y2, x1:x2] = redacted_image[y1:y2, x1:x2]
            else:
                raise ValueError(
                    f"Unknown redaction type: {self.redaction_type}"
                )
        return image

    def get_fields(self, brain_key):
        fields = [
            self.label_field,
            f"{brain_key}_filepath",
        ]
        return fields

    def cleanup(self, samples, brain_key):
        label_field = self.label_field

        samples._dataset.delete_sample_field(
            f"{brain_key}_filepath", error_level=1
        )

        if samples._is_frame_field(label_field):
            samples._dataset.delete_frame_fields(brain_key, error_level=1)

    def _validate_run(self, samples, brain_key, existing_info):
        self._validate_fields_match(brain_key, "label_field", existing_info)
        self._validate_fields_match(
            brain_key, "redaction_field", existing_info
        )


class RedactionResults(fob.BrainResults):
    """Class for storing the results of :meth:`fiftyone.brain.create_redaction`.

    Args:
        samples: the :class:`fiftyone.core.collections.SampleCollection` used
        config: the :class:`RedactionConfig` used
        brain_key: the brain key
        backend (None): a :class:`Redaction` backend
    """

    def __init__(self, samples, config, brain_key, backend=None):
        super().__init__(samples, config, brain_key, backend=backend)

    def generate_redacted_dataset(self, name=None, overwrite=False):
        """
        Generates a new dataset with the redacted media only.
        Args:
            name: name of the redacted dataset
            overwrite: whether to overwrite the redacted dataset if it already exists
        Returns:
            fo.Dataset
        """
        redacted_dataset_name = (
            name
            if name is not None
            else f"{self.samples._dataset.name}_{self.samples.name}_redacted"
        )

        select_fields = self.backend.get_fields(self.key)
        redacted_view = self.samples.select_fields(select_fields)
        # exclude detections of type label_field from the redacted dataset
        redacted_view = redacted_view.filter_labels(
            f"{self.backend.config.label_field}.detections",
            ~F("label").is_in(self.backend.label_classes),
            only_matches=False,
        )

        if overwrite and redacted_dataset_name in fo.list_datasets():
            fo.delete_dataset(redacted_dataset_name)
        redacted_dataset = redacted_view.clone(name=redacted_dataset_name)
        redacted_dataset.set_values(
            "filepath", redacted_view.values(f"{self.key}_filepath")
        )

        # remove the redacted field from the redacted dataset
        redacted_dataset.delete_sample_field(
            f"{self.key}_filepath", error_level=1
        )

        logger.info(f"Redacted dataset {redacted_dataset.name} created")
        return redacted_dataset


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
    filename = os.path.basename(inpath)
    dir_path = os.path.join(dir_path, rel_output_dir)
    os.makedirs(dir_path, exist_ok=True)
    return os.path.join(dir_path, filename)
