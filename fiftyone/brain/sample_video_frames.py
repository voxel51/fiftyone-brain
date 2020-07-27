"""
Methods for adaptively sampling video frames.

| Copyright 2017-2020, Voxel51, Inc.
| `voxel51.com <https://voxel51.com/>`_
|
"""
# pragma pylint: disable=redefined-builtin
# pragma pylint: disable=unused-wildcard-import
# pragma pylint: disable=wildcard-import
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from builtins import *

# pragma pylint: enable=redefined-builtin
# pragma pylint: enable=unused-wildcard-import
# pragma pylint: enable=wildcard-import

import logging
import os

from eta.core.config import Config
import eta.core.image as etai
import eta.core.video as etav

import fiftyone as fo
import fiftyone.brain.core.motion as fobm
import fiftyone.brain.core.sampling as fobs
import fiftyone.brain.core.quality as fobq


logger = logging.getLogger(__name__)


def sample_best_video_frames(
    video_path,
    out_frames_dir,
    target_num_frames=None,
    target_accel=None,
    target_fps=None,
    size=None,
    max_size=None,
):
    """Adaptively samples the best frames from the input video.

    The "best" video frames at a given sampling density are defined as the
    frames with highest image quality that are most representative of the
    visual content in the video.

    Provide one of ``target_num_frames``, ``target_accel``, or ``target_fps``
    to perform the sampling.

    Args:
        video_path: the path to the video to process
        out_frames_dir: a directory to write the sampled frames
        target_num_frames (None): the target number of frames to sample
        target_accel (None): a desired target acceleration factor to apply when
            sampling frames. For example, a target acceleration of 2x would
            correspond to sampling every other frame, on average
        target_fps (None): a desired target sampling rate, which must be less
            than the frame rate of the input video
        size (None): a desired output ``(width, height)`` for the sampled
            frames. Dimensions can be -1, in which case the input aspect ratio
            is preserved. By default, the input frame size is maintained
        max_size (None): a maximum ``(width, height)`` allowed for the sampled
            frames. Frames are resized as necessary to meet this limit, and
            ``size`` is decreased (aspect-preserving) if necessary to satisfy
            this constraint. Dimensions can be -1, in which case no limit is
            applied to them. By default, no maximum frame size is imposed

    Returns:
        a dict mapping frame numbers to the paths to the sampled frames in
        ``out_frames_dir``
    """
    #
    # Algorithm
    #
    # Divides the video into bins of equal motion, and samples the highest
    # quality frame from each bin. A blowout parameter is used to avoid
    # sampling consecutive frames within a given radius
    #
    # Performance
    #
    # Requires a single pass over the frames of the video
    #
    parameters = SampleBestVideoFramesParameters.default()

    video_metadata = etav.VideoMetadata.build_for(video_path)

    # Compute target acceleration
    target_accel = _compute_target_accel(
        video_metadata, target_num_frames, target_accel, target_fps
    )

    # Compute output frame size
    size = _compute_output_frame_size(video_metadata, size, max_size)

    # Sample frames
    logger.info(
        "Performing adaptive sampling with target acceleration %g",
        target_accel,
    )
    output_patt = os.path.join(
        out_frames_dir,
        fo.config.default_sequence_idx + fo.config.default_image_ext,
    )
    video_labels = etav.VideoLabels()  # currently used internally only
    frames = _perform_adaptive_sampling(
        video_path, video_labels, target_accel, output_patt, size, parameters
    )

    return frames


class SampleBestVideoFramesParameters(Config):
    """Internal parameters for :meth:`sample_best_video_frames`.

    Args:
        use_motion: (True) whether to choose sample windows whose widths are
            proportionate to frame motion
        motion_config: (None) a
            :class:`fiftyone.brain.core.motion.FrameMotionConfig` describing
            the motion method to use. Only used when ``use_motion == True``
        use_quality: (True) whether to use frame quality to inform which frame
            from each sampling window to select
        quality_config: (None) a
            :class:`fiftyone.brain.core.quality.FrameQualityConfig` describing
            the quality method to use. Only used when ``use_quality == True``
        blowout: (0.25) a blowout factor in ``(0, 1)`` that defines a region in
            which subsequent samples cannot be taken. For example, a blowout
            factor of 0.25 corresponds to a minimum sampling distance of
            ``0.25 * target_accel``. A blowout factor of 1 or greater would
            force uniform sampling
        delta: (None) a multiple of frame quality standard deviation increments
            that earns equal weight to a one frame deviation from the sampling
            point recommended by the target acceleration. Setting delta to a
            small value, say ``1e-5``, will always sample the maximum quality
            frame within each sampling window. Setting delta to a large value,
            say ``1e5``, will reduce to uniform sampling at the target rate.
            The default value is ``1 / target_accel``
        offset: (None) an optional offset from the beginning of the video to
            choose the initial sampling window
        always_sample_first: (False) whether to always sample the first frame
            of the video
        always_sample_last: (False) whether to always sample the last frame of
            the video
    """

    def __init__(self, d):
        self.use_motion = self.parse_bool(d, "use_motion", default=True)
        self.motion_config = self.parse_object(
            d, "motion_config", fobm.FrameMotionConfig, default=None
        )
        self.use_quality = self.parse_bool(d, "use_quality", default=True)
        self.quality_config = self.parse_object(
            d, "quality_config", fobq.FrameQualityConfig, default=None
        )
        self.blowout = self.parse_number(d, "blowout", default=0.25)
        self.delta = self.parse_number(d, "delta", default=None)
        self.offset = self.parse_number(d, "offset", default=None)
        self.always_sample_first = self.parse_bool(
            d, "always_sample_first", default=False
        )
        self.always_sample_last = self.parse_bool(
            d, "always_sample_last", default=False
        )


def _compute_target_accel(
    video_metadata, target_num_frames, target_accel, target_fps
):
    ifps = video_metadata.frame_rate
    iframe_count = video_metadata.total_frame_count

    if target_num_frames is not None:
        if target_num_frames > iframe_count:
            raise ValueError(
                "Target number of frames %d cannot be greater than total "
                "frame count %d" % (target_num_frames, iframe_count)
            )

        target_accel = iframe_count / target_num_frames
    elif target_accel is not None:
        if target_accel < 1:
            raise ValueError(
                "Acceleration factor must be greater than 1; found %d"
                % target_accel
            )
    elif target_fps is not None:
        if target_fps > ifps:
            raise ValueError(
                "Target frame rate (%g) cannot be greater than input frame "
                "rate (%g)" % (target_fps, ifps)
            )

        target_accel = ifps / target_fps
    else:
        raise ValueError(
            "Either `target_num_frames`, `target_accel`, or `target_fps` must "
            "be specified"
        )

    return target_accel


def _compute_output_frame_size(video_metadata, size, max_size):
    isize = video_metadata.frame_size

    # Compute output frame size
    if size is not None:
        psize = etai.parse_frame_size(size)
        osize = etai.infer_missing_dims(psize, isize)
    else:
        osize = isize

    if max_size is not None:
        msize = etai.parse_frame_size(max_size)
        osize = etai.clamp_frame_size(osize, msize)

    # Avoid resizing if possible
    resize_frames = osize != isize
    if resize_frames:
        owidth, oheight = osize
        logger.info("Resizing sampled frames to %d x %d", owidth, oheight)

    return osize if resize_frames else None


def _perform_adaptive_sampling(
    video_path,
    video_labels,
    target_accel,
    output_frames_patt,
    size,
    parameters,
):
    # Parse parameters
    offset = parameters.offset
    use_motion = parameters.use_motion
    motion_config = parameters.motion_config
    use_quality = parameters.use_quality
    quality_config = parameters.quality_config
    blowout = parameters.blowout
    delta = parameters.delta
    always_sample_first = parameters.always_sample_first
    always_sample_last = parameters.always_sample_last

    # Sample frames
    if not use_motion and not use_quality:
        logger.info("Uniformly sampling frames")
        frames = fobs.uniform_sample_frames(
            video_path,
            output_frames_patt,
            target_accel,
            offset=offset,
            always_sample_first=always_sample_first,
            always_sample_last=always_sample_last,
            size=size,
        )
    elif use_motion and not use_quality:
        logger.info("Using frame motion to inform sampling")
        _, frames, _ = fobs.adaptive_sample_frames_by_motion(
            video_path,
            output_frames_patt,
            target_accel,
            video_labels=video_labels,
            blowout=blowout,
            offset=offset,
            always_sample_first=always_sample_first,
            always_sample_last=always_sample_last,
            size=size,
            motion_config=motion_config,
        )
    elif not use_motion and use_quality:
        logger.info("Using frame quality to inform sampling")
        _, frames, _ = fobs.adaptive_sample_frames_by_quality(
            video_path,
            output_frames_patt,
            target_accel,
            video_labels=video_labels,
            blowout=blowout,
            delta=delta,
            offset=offset,
            always_sample_first=always_sample_first,
            always_sample_last=always_sample_last,
            size=size,
            quality_config=quality_config,
        )
    elif use_motion and use_quality:
        logger.info("Using frame quality and motion to inform sampling")
        _, frames, _ = fobs.adaptive_sample_frames_by_quality_and_motion(
            video_path,
            output_frames_patt,
            target_accel,
            video_labels=video_labels,
            blowout=blowout,
            delta=delta,
            offset=offset,
            always_sample_first=always_sample_first,
            always_sample_last=always_sample_last,
            size=size,
            quality_config=quality_config,
            motion_config=motion_config,
        )
    else:
        raise ValueError("Invalid sampling parameters...")

    return {f: output_frames_patt % f for f in frames}