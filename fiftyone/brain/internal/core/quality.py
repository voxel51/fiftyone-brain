"""
Core infrastructure for computing qualities of images and video frames.

| Copyright 2017-2020, Voxel51, Inc.
| `voxel51.com <https://voxel51.com/>`_
|
"""
import os

from eta.core.config import Config
import eta.core.data as etad
import eta.core.image as etai
import eta.core.video as etav

import fiftyone.brain.internal.core.image as fobi


class FrameQualityConfig(Config):
    """Frame quality configuration settings.

    Attributes:
        method: the frame quality method to use. Can be any value supported by
            :meth:`fiftyone.brain.internal.core.image.compute_quality`. The
            default is ``"laplacian-stdev"``
        attr_name: the name of the numeric attribute in which to store the
            frame quality values. The default is ``"quality"``
    """

    def __init__(self, d):
        self.method = self.parse_string(d, "method", default="laplacian-stdev")
        self.attr_name = self.parse_string(d, "attr_name", default="quality")


def compute_video_frame_qualities(
    video_reader, video_labels=None, quality_config=None
):
    """Computes the qualities of the frames of the given video.

    Args:
        video_reader: an ``eta.core.video.VideoReader`` that is ready to read
            the frames of the video
        video_labels (None): an optional ``eta.core.video.VideoLabels`` to
            which to add the labels. By default, a new instance is created
        quality_config (None): a :class:`FrameQualityConfig` describing the
            method to use. If omitted, the default config is used

    Returns:
        a ``eta.core.video.VideoLabels`` containing the frame qualities
    """
    if quality_config is None:
        quality_config = FrameQualityConfig.default()

    # Parse config
    method = quality_config.method
    attr_name = quality_config.attr_name

    if video_labels is None:
        video_labels = etav.VideoLabels()

    # Compute frame qualities
    with video_reader as vr:
        for img in vr:
            attr = _compute_quality(img, method, attr_name)
            video_labels.add_frame_attribute(attr, vr.frame_number)

    return video_labels


def compute_image_quality(img, image_labels=None, quality_config=None):
    """Computes the quality of the given image.

    Args:
        img: an image
        image_labels (None): an optional ``eta.core.image.ImageLabels`` to
            which to add the label. By default, a new instance is created
        quality_config (None): a :class:`FrameQualityConfig` describing the
            method to use. If omitted, the default config is used

    Returns:
        an ``eta.core.image.ImageLabels`` containing the frame quality
    """
    if quality_config is None:
        quality_config = FrameQualityConfig.default()

    # Parse config
    method = quality_config.method
    attr_name = quality_config.attr_name

    if image_labels is None:
        image_labels = etai.ImageLabels()

    # Compute frame quality
    attr = _compute_quality(img, method, attr_name)
    image_labels.add_attribute(attr)

    return image_labels


def compute_image_set_qualities(
    image_paths, image_set_labels=None, quality_config=None
):
    """Computes the quality of the given set of images.

    Args:
        image_paths: an iterator over image paths to process
        image_set_labels (None): an optional ``eta.core.image.ImageSetLabels``
            to which to add the labels. By default, a new instance is created
        quality_config (None): a :class:`FrameQualityConfig` describing the
            method to use. If omitted, the default config is used

    Returns:
        an ``eta.core.image.ImageSetLabels`` containing the frame qualities
    """
    if quality_config is None:
        quality_config = FrameQualityConfig.default()

    # Parse config
    method = quality_config.method
    attr_name = quality_config.attr_name

    if image_set_labels is None:
        image_set_labels = etai.ImageSetLabels()

    # Compute frame qualities
    for image_path in image_paths:
        filename = os.path.basename(image_path)
        img = etai.read(image_path)
        attr = _compute_quality(img, method, attr_name)
        image_set_labels[filename].add_attribute(attr)

    return image_set_labels


def _compute_quality(img, method, attr_name):
    quality = fobi.compute_quality(img, method=method)
    return etad.NumericAttribute(attr_name, quality)
