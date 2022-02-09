"""
Core infrastructure for computing frame motion in videos.

| Copyright 2017-2020, Voxel51, Inc.
| `voxel51.com <https://voxel51.com/>`_
|
"""
import numpy as np

from eta.core.config import Config
import eta.core.data as etad
import eta.core.utils as etau
import eta.core.video as etav


# @todo add support for customizing the BackgroundSubtractor parameters
class FrameMotionConfig(Config):
    """Frame motion configuration settings.

    Attributes:
        background_subtractor: the fully-qualified name of the
            ``eta.core.primitives.BackgroundSubtractor`` class to use. The
            default is ``"eta.core.primitives.MOG2BackgroundSubtractor"``
        motion_method: the method to use to compute the frame motion. Supported
            values are ``("fgsupport")``. The default is ``"fgsupport"``
        init_buffer_frames: the number of initial buffer frames before frame
            motion should be trusted and reported. The default is 5 frames
        attr_name: the name of the numeric attribute in which to store the
            frame motion values. The default is ``"motion"``
    """

    def __init__(self, d):
        self.background_subtractor = self.parse_string(
            d,
            "background_subtractor",
            default="eta.core.primitives.MOG2BackgroundSubtractor",
        )
        self.motion_method = self.parse_string(
            d, "motion_method", default="fgsupport"
        )
        self.init_buffer_frames = self.parse_number(
            d, "init_buffer_frames", default=5
        )
        self.attr_name = self.parse_string(d, "attr_name", default="motion")


def compute_frame_motion(video_reader, video_labels=None, motion_config=None):
    """Computes the frame motion for the frames of the given video.

    Args:
        video_reader: an ``eta.core.video.VideoReader`` that is ready to read
            the frames of the video
        video_labels (None): an optional ``eta.core.video.VideoLabels`` to
            which to add the labels. By default, a new instance is created
        motion_config (None): a :class:`FrameMotionConfig` describing the
            method to use. If omitted, the default config is used

    Returns:
        a ``eta.core.video.VideoLabels`` containing the frame motions
    """
    if motion_config is None:
        motion_config = FrameMotionConfig.default()

    if motion_config.motion_method != "fgsupport":
        raise ValueError(
            "Unsupported motion_method = '%s'" % motion_config.motion_method
        )

    background_subtractor_cls = etau.get_class(
        motion_config.background_subtractor
    )
    init_buffer_frames = motion_config.init_buffer_frames
    attr_name = motion_config.attr_name

    if video_labels is None:
        video_labels = etav.VideoLabels()

    with background_subtractor_cls() as bgs:
        with video_reader as vr:
            for idx, img in enumerate(vr):
                attr = _compute_motion(img, bgs, attr_name)
                if idx >= init_buffer_frames:
                    # Only store motion after initial buffer
                    video_labels.add_frame_attribute(attr, vr.frame_number)

    return video_labels


def _compute_motion(img, bgs, attr_name):
    # Motion = proportion of foreground pixels
    fgmask, _ = bgs.process_frame(img)
    motion = np.count_nonzero(fgmask) / fgmask.size
    return etad.NumericAttribute(attr_name, motion)
