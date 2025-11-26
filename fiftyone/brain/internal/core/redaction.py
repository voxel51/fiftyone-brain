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

# from eta.core.video import VideoProcessor

import fiftyone as fo
import fiftyone.core.storage as fos
import fiftyone.core.brain as fob
import fiftyone.core.labels as fol
import fiftyone.core.validation as fov
from fiftyone import ViewField as F


logger = logging.getLogger(__name__)


_ALLOWED_TYPES = (
    fol.Detection,
    fol.Detections,
    fol.TemporalDetection,
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
    redaction_field,
    force_recreate,
    create_as_new_sample,
    num_workers,
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
        brain_key,
        "original_sample_id",
        "redacted_sample_ids",
    ]
    select_fields = [
        ff for ff in select_fields if samples.get_field(ff) is not None
    ]
    view = samples.select_fields(select_fields)

    logger.info("Computing redaction...")
    for sample in view.iter_samples(progress=progress):
        brain_method.process_sample(sample)
        sample.save()

    logger.info(f"Adding redacted media field: {brain_key}_filepath")
    if f"{brain_key}_filepath" not in view._dataset.app_config.media_fields:
        view._dataset.app_config.media_fields.append(f"{brain_key}_filepath")

    view._dataset.save()

    results = RedactionResults(samples, config, brain_key, brain_method)
    brain_method.save_run_results(samples, brain_key, results)

    logger.info("Redaction computation complete")
    return results


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
        redaction_field,
        force_recreate,
        create_as_new_sample,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.label_field = label_field
        self.label_classes = label_classes
        self.redaction_type = redaction_type
        self.redaction_method = redaction_method
        self.redaction_field = redaction_field
        self.force_recreate = force_recreate
        self.create_as_new_sample = create_as_new_sample
        self.make_redaction_field()

    def make_redaction_field(self):
        if isinstance(self.label_classes, str):
            self.label_classes = [
                label_name.strip()
                for label_name in self.label_classes.strip("[")
                .strip("]")
                .split(",")
            ]
        else:
            try:
                self.label_classes = list(map(str, self.label_classes))
            except:
                raise ValueError(
                    f"Invalid label classes type: {type(self.label_classes)}"
                )
        redaction_labels = "_".join(self.label_classes)
        redaction_field_string = f"redacted_{self.label_field.replace('.', '_')}_{redaction_labels}_{self.redaction_type}_{self.redaction_method}"
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
        return self.redaction_field


class Redaction(fob.BrainMethod):
    def __init__(self, config):
        super().__init__(config)
        self.label_field = None
        self.label_type = None
        self.label_classes = None
        self.redaction_type = None
        self.redaction_method = None
        self.force_recreate = None
        self.create_as_new_sample = None
        self.redaction_field = None
        self.processing_frames = False

    def ensure_requirements(self):
        pass

    def register_samples(self, samples):
        self.label_field, _ = samples._handle_frame_field(
            self.config.label_field
        )
        self.label_type = samples._get_label_field_type(
            self.config.label_field
        )
        self.label_classes = self.config.label_classes

        self.redaction_type = self.config.redaction_type
        self.redaction_method = self.config.redaction_method
        self.redaction_field = self.config.redaction_field
        self.force_recreate = self.config.force_recreate
        self.create_as_new_sample = self.config.create_as_new_sample
        self.processing_frames = samples._is_frame_field(
            self.config.label_field
        )

    def process_sample(self, sample):
        redacted_media_path = _get_outpath(
            sample.filepath,
            rel_output_dir=self.redaction_field,
        )
        # if self.create_as_new_sample:
        #     if (
        #         ("redacted_sample_ids" in sample)
        #         and (sample["redacted_sample_ids"] is not None)
        #         and (self.redaction_field in sample["redacted_sample_ids"])
        #         and (
        #             os.path.exists(
        #                 sample._dataset[
        #                     sample["redacted_sample_ids"][self.redaction_field]
        #                 ]["filepath"]
        #             )
        #         )
        #     ):
        #         logger.debug(
        #             f"Redaction already exists for sample {sample.id} --> {sample['redacted_sample_ids'][self.redaction_field]}"
        #         )
        #         if self.force_recreate:
        #             sample._dataset.delete_sample(
        #                 sample["redacted_sample_ids"][self.redaction_field]
        #             )
        #             sample["redacted_sample_ids"].pop(self.redaction_field)
        #         else:
        #             return
        #     if self.redaction_field in sample.tags:
        #         logger.debug(
        #             f"This is a redaction of sample {sample['original_sample_id']}"
        #         )
        #         return

        if (
            (f"{self.redaction_field}_filepath" in sample)
            and (sample[f"{self.redaction_field}_filepath"] is not None)
            and (os.path.exists(sample[f"{self.redaction_field}_filepath"]))
        ):
            logger.debug(f"Redaction already exists for sample {sample.id}")
            if self.force_recreate:
                sample[f"{self.redaction_field}_filepath"] = None
                sample.tags.remove(self.redaction_field)
            else:
                return

        shutil.copy(sample.filepath, redacted_media_path)

        if self.processing_frames:
            detections_object_list = [
                frame[self.label_field.replace("frames.", "")]
                for frame in sample.frames.values()
            ]
            self.redact_video_file_at(
                redacted_media_path, detections_object_list
            )
        else:
            self.redact_file_at(redacted_media_path, sample[self.label_field])

        # if self.create_as_new_sample:
        #     redacted_sample = sample.copy()
        #     redacted_sample[self.label_field] = self.filter_detections(
        #         sample[self.label_field], keep_matching=False
        #     )
        #     redacted_sample["filepath"] = redacted_media_path
        #     if self.redaction_field not in redacted_sample.tags:
        #         redacted_sample.tags.append(self.redaction_field)
        #     redacted_sample["original_sample_id"] = sample.id
        #     logger.debug(
        #         f"Adding redacted sample at filepath: {redacted_media_path}"
        #     )
        #     sample._dataset.add_sample(redacted_sample)
        #     if ("redacted_sample_ids" not in sample) or (
        #         sample["redacted_sample_ids"] is None
        #     ):
        #         sample["redacted_sample_ids"] = {}
        #     sample["redacted_sample_ids"][
        #         self.redaction_field
        #     ] = redacted_sample.id

        # add the redacted filepath to the sample
        if self.redaction_field not in sample.tags:
            sample.tags.append(self.redaction_field)
        sample[f"{self.redaction_field}_filepath"] = redacted_media_path

        sample.save()
        return

    def redact_file_at(self, redacted_path, detections_object):
        image = cv2.imread(redacted_path)
        redacted_image = self._redact_entire_image(image)
        image = self._apply_redaction_to_image(
            image, redacted_image, detections_object
        )
        cv2.imwrite(redacted_path, image)

    def redact_video_file_at(self, redacted_path, detections_object_list):
        video_frames, fps, (width, height) = read_video_frames(redacted_path)
        if len(video_frames) != len(detections_object_list):
            raise ValueError(
                f"Number of video frames ({len(video_frames)}) does not match the number of detections ({len(detections_object_list)})"
            )

        redacted_video_frames = []
        for video_frame, detections_object in zip(
            video_frames, detections_object_list
        ):
            redacted_image = self._redact_entire_image(video_frame)
            redacted_image = self._apply_redaction_to_image(
                video_frame, redacted_image, detections_object
            )
            redacted_video_frames.append(redacted_image)

        write_video_frames(
            redacted_path, redacted_video_frames, fps, (width, height)
        )

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
        return image

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
            self.label_field,
            f"{brain_key}_filepath",
            # "original_sample_id",
            # "redacted_sample_ids",
        ]

        # if samples._is_frame_field(self.label_field):
        #     fields.append(samples._FRAMES_PREFIX + f"{brain_key}_filepath")

        return fields

    def cleanup(self, samples, brain_key):
        label_field = self.label_field

        # samples._dataset.delete_sample_fields(brain_key, error_level=1)
        samples._dataset.delete_sample_fields(
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
        redacted_dataset = fo.Dataset(
            name=redacted_dataset_name, overwrite=overwrite
        )

        select_fields = self.backend.get_fields(self.samples, self.key)
        redacted_view = self.samples.select_fields(select_fields)

        for redacted_sample in redacted_view.iter_samples():
            redacted_sample["filepath"] = redacted_sample[
                f"{self.key}_filepath"
            ]
            redacted_sample[
                self.backend.label_field
            ] = self.backend.filter_detections(
                redacted_sample[self.backend.label_field], keep_matching=False
            )

            redacted_sample.tags.append(self.key)
            redacted_dataset.add_sample(redacted_sample)

        redacted_dataset.delete_sample_fields(
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
    # dir_path = fos.normalize_path(dir_path)
    filename = os.path.basename(inpath)
    dir_path = os.path.join(dir_path, rel_output_dir)
    os.makedirs(dir_path, exist_ok=True)
    return os.path.join(dir_path, filename)


def read_video_frames(filepath):
    cap = cv2.VideoCapture(filepath)

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames, fps, (width, height)


def write_video_frames(
    filepath, frames, fps: int, frame_width_height: Tuple[int, int]
):
    video_file = cv2.VideoWriter(
        filepath, cv2.VideoWriter_fourcc(*"mp4v"), fps, frame_width_height
    )
    for frame in frames:
        video_file.write(frame)
    video_file.release()
    return
