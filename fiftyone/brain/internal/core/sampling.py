"""
Core infrastructure for adaptively sampling frames of videos.

@todo add common interface for frame sampling methods defined here

| Copyright 2017-2020, Voxel51, Inc.
| `voxel51.com <https://voxel51.com/>`_
|
"""
from copy import copy
import logging
import os

import numpy as np

from eta.core.config import Config
import eta.core.data as etad
import eta.core.image as etai
import eta.core.utils as etau
import eta.core.video as etav

import fiftyone as fo

import fiftyone.brain.internal.core.image as fobi
import fiftyone.brain.internal.core.motion as fobm
import fiftyone.brain.internal.core.quality as fobq


logger = logging.getLogger(__name__)


class SampleBestVideoFramesParameters(Config):
    """Internal parameters for
    :meth:`fiftyone.sample_video_frames.sample_best_video_frames`.

    Args:
        use_motion: (True) whether to choose sample windows whose widths are
            proportionate to frame motion
        motion_config: (None) a
            :class:`fiftyone.brain.internal.core.motion.FrameMotionConfig`
            describing the motion method to use. Only used when
            ``use_motion == True``
        use_quality: (True) whether to use frame quality to inform which frame
            from each sampling window to select
        quality_factor: (1): a quality factor in ``[0, 1]`` defining the target
            frame qualities to sample. Only used when ``use_quality == True``
        quality_config: (None) a
            :class:`fiftyone.brain.internal.core.quality.FrameQualityConfig`
            describing the quality method to use. Only used when
            ``use_quality == True``
        blowout: (0.25) a blowout factor in ``(0, 1)`` that defines a region in
            which subsequent samples cannot be taken. For example, a blowout
            factor of 0.25 corresponds to a minimum sampling distance of
            ``0.25 * target_accel``. A blowout factor of 1 or greater would
            force uniform sampling
        alpha: (0.9) a frame quality factor in ``(0, 1)`` that determines how
            much weight to give to frame quality relative to uniform sampling
            when sampling frames. Setting ``alpha == 1`` corresponds to always
            selecting the frame with the desired quality, while ``alpha == 0``
            corresponds to uniform sampling at the target rate
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
        self.quality_factor = self.parse_number(d, "quality_factor", default=1)
        self.quality_config = self.parse_object(
            d, "quality_config", fobq.FrameQualityConfig, default=None
        )
        self.blowout = self.parse_number(d, "blowout", default=0.25)
        self.alpha = self.parse_number(d, "alpha", default=0.9)
        self.offset = self.parse_number(d, "offset", default=None)
        self.always_sample_first = self.parse_bool(
            d, "always_sample_first", default=False
        )
        self.always_sample_last = self.parse_bool(
            d, "always_sample_last", default=False
        )


def sample_best_video_frames(
    video_path,
    out_frames_dir,
    quality_factor=1,
    target_num_frames=None,
    target_accel=None,
    target_fps=None,
    size=None,
    max_size=None,
):
    """Adaptively samples the best frames from the input video.

    The "best" video frames at a given sampling density are defined as the
    frames that are most representative of the visual content of the video.

    The ``quality_factor`` parameter in ``[0, 1]`` defines the desired image
    quality of the frames to be sampled. A `quality_factor == k` means to
    sample frames whose quality are in the ``k x 100%`` percentile of quality
    in their temporal neighborhood of frames.

    Provide one of ``target_num_frames``, ``target_accel``, or ``target_fps``
    to perform the sampling.

    Args:
        video_path: the path to the video to process
        out_frames_dir: a directory to write the sampled frames
        quality_factor (1): a quality factor in ``[0, 1]`` specifying the
            target frame qualities to sample
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

    parameters = SampleBestVideoFramesParameters.default()
    parameters.quality_factor = quality_factor

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


def uniform_sample_frames(
    video_path,
    output_frames_patt,
    target_accel,
    offset=None,
    always_sample_first=False,
    always_sample_last=False,
    size=None,
):
    """Uniformly samples frames from the given video to achieve the specified
    target acceleration.

    This implementation uniformly samples frames from the video, with the first
    frame sampled being ``offset + 1``.

    Args:
        video_path: the path to the video
        output_frames_patt: an output pattern like ``/path/to/frames/%06d.jpg``
            to which to write the sampled frames
        target_accel: the target acceleration factor to apply when sampling
            frames. For example, a target acceleration of 2x would correspond
            to sampling every other frame, on average
        offset (None): an offset from the beginning of the video to choose the
            initial sample. By default, the first frame is sampled
        always_sample_first (False): whether to always sample the first frame
            of the video
        always_sample_last (False): whether to always sample the last frame of
            the video
        size (None): a desired output ``(width, height)`` of the sampled
            frames. Dimensions can be -1, in which case the input aspect ratio
            is preserved

    Returns:
        a sorted list of frame numbers that were sampled
    """
    # Select frames to sample
    total_frame_count = etav.get_frame_count(video_path)
    frames = _select_frames_uniformly(
        total_frame_count,
        target_accel,
        offset,
        always_sample_first,
        always_sample_last,
    )

    # Sample the frames
    etav.sample_select_frames(
        video_path,
        frames,
        output_patt=output_frames_patt,
        size=size,
        fast=True,
    )

    return frames


def adaptive_sample_frames_by_quality(
    video_path,
    output_frames_patt,
    target_accel,
    video_labels=None,
    blowout=None,
    alpha=0.9,
    offset=None,
    always_sample_first=False,
    always_sample_last=False,
    size=None,
    quality_factor=1,
    quality_config=None,
):
    """Adaptively samples frames from the given video to achieve the specified
    target acceleration.

    This implementation uses frame quality to inform which frame from each
    uniformly spaced interval of the given video to sample.

    The frame quality of every frame is stored in the output labels.

    Args:
        video_path: the path to the video
        output_frames_patt: an output pattern like ``/path/to/frames/%06d.jpg``
            to which to write the sampled frames
        target_accel: the target acceleration factor to apply when sampling
            frames. For example, a target acceleration of 2x would correspond
            to sampling every other frame, on average
        video_labels (None): an optional ``eta.core.video.VideoLabels`` to
            which to add the labels. By default, a new instance is created
        blowout (None): a blowout factor in ``(0, 1)`` that defines a region in
            which subsequent samples cannot be taken. For example, a blowout
            factor of 0.25 corresponds to a minimum sampling distance of
            ``0.25 * target_accel``. A blowout factor of 1 or greater would
            force uniform sampling
        alpha: (0.9) a frame quality factor in ``(0, 1)`` that determines how
            much weight to give to frame quality relative to uniform sampling
            when sampling frames. Setting ``alpha == 1`` corresponds to always
            selecting the frame with the desired quality, while ``alpha == 0``
            corresponds to uniform sampling at the target rate
        offset (None): an optional offset from the beginning of the video to
            set the initial sampling window. By default, there is no offset
        always_sample_first (False): whether to always sample the first frame
            of the video
        always_sample_last (False): whether to always sample the last frame of
            the video
        size (None): a desired output ``(width, height)`` of the sampled
            frames. Dimensions can be -1, in which case the input aspect ratio
            is preserved
        quality_factor (1): a quality factor in ``[0, 1]`` specifying the
            target frame qualities to sample
        quality_config (None): a
            :class:`fiftyone.brain.internal.core.quality.FrameQualityConfig`
            describing the frame quality method to use

    Returns:
        a tuple of

        -   video_labels: a ``eta.core.video.VideoLabels`` containing the frame
            qualities and motions for all frames of the video
        -   frames: a sorted list of frames that were sampled
        -   sample_edges: a sorted list of sample boundaries used for the
            sampling
    """
    #
    # Parse parameters
    #

    if video_labels is None:
        video_labels = etav.VideoLabels()

    if offset is None:
        offset = 0

    # Compute blowout width, in frames
    blowout_width = _compute_blowout_width(blowout, target_accel)

    if quality_config is None:
        quality_config = fobq.FrameQualityConfig.default()

    # Lookback window size over which to compute frame quality statistics
    lookback_size = int(10 * target_accel)

    #
    # Sample frames
    #

    with etav.FFmpegVideoReader(video_path) as vr:
        total_frame_count = vr.total_frame_count

        def get_decision_frame(num):
            return min(
                int(round(num * target_accel)) + offset, total_frame_count
            )

        last_sample = None
        num_windows = 1
        blowout_boundary = offset
        next_decision_frame = get_decision_frame(num_windows)
        sample_edges = [1 + offset, next_decision_frame]

        imgs = []
        frame_qualities = []
        frames = []
        for img in vr:
            frame_number = vr.frame_number
            if frame_number > blowout_boundary:
                imgs.append(img)

            # Compute frame quality
            attr = _compute_frame_quality(
                img, quality_config.attr_name, quality_config.method
            )
            video_labels.add_frame_attribute(attr, frame_number)
            frame_qualities.append(attr.value)

            # Always sample first/last frame, if requested
            always_sample = ((frame_number == 1) and always_sample_first) or (
                (frame_number == total_frame_count) and always_sample_last
            )
            if always_sample:
                frames.append(frame_number)
                _sample_frame(img, frame_number, output_frames_patt, size=size)

            if frame_number < next_decision_frame:
                continue

            # Pick frame to sample
            window_size = len(imgs)
            sample_frame_number, idx = _select_best_frame_in_window(
                frame_qualities,
                last_sample,
                frame_number,
                window_size,
                lookback_size,
                target_accel,
                alpha,
                quality_factor,
            )
            sample_img = imgs[idx]

            # Sample frame
            num_windows += 1
            if not always_sample or (sample_frame_number != frame_number):
                frames.append(sample_frame_number)
                _sample_frame(
                    sample_img,
                    sample_frame_number,
                    output_frames_patt,
                    size=size,
                )

            if frame_number >= total_frame_count:
                break

            # Set next decision frame
            next_decision_frame = get_decision_frame(num_windows)
            sample_edges.append(next_decision_frame)
            last_sample = sample_frame_number
            blowout_boundary = min(
                last_sample + blowout_width, next_decision_frame - 1
            )
            imgs = []

    # The last two frames may be out of order if `always_sample_last == True`
    if always_sample_last:
        frames = sorted(frames)

    return video_labels, frames, sample_edges


def adaptive_sample_frames_by_motion(
    video_path,
    output_frames_patt,
    target_accel,
    video_labels=None,
    blowout=None,
    offset=None,
    always_sample_first=False,
    always_sample_last=False,
    size=None,
    motion_config=None,
):
    """Adaptively samples frames from the given video to achieve the specified
    target acceleration.

    This implementation uses frame motion to choose sample windows whose widths
    are proportionate to frame motion. In particular, the first frame
    (or ``1 + offset`` when an offset is provided) and the last frame of the
    video is always sampled.

    The frame motion of every frame is stored in the output labels.

    Args:
        video_path: the path to the video
        output_frames_patt: an output pattern like ``/path/to/frames/%06d.jpg``
            to which to write the sampled frames
        target_accel: the target acceleration factor to apply when sampling
            frames. For example, a target acceleration of 2x would correspond
            to sampling every other frame, on average
        video_labels (None): an optional ``eta.core.video.VideoLabels`` to
            which to add the labels. By default, a new instance is created
        blowout (None): a blowout factor in ``(0, 1)`` that defines a region in
            which subsequent samples cannot be taken. For example, a blowout
            factor of 0.25 corresponds to a minimum sampling distance of
            ``0.25 * target_accel``. A blowout factor of 1 or greater would
            force uniform sampling
        offset (None): an optional offset from the beginning of the video to
            set the initial sampling window. By default, there is no offset
        always_sample_first (False): whether to always sample the first frame
            of the video
        always_sample_last (False): whether to always sample the last frame of
            the video
        size (None): a desired output ``(width, height)`` of the sampled
            frames. Dimensions can be -1, in which case the input aspect ratio
            is preserved
        motion_config (None): a
            :class:`fiftyone.brain.internal.core.motion.FrameMotionConfig`
            describing the frame motion method to use

    Returns:
        a tuple of

        -   video_labels: an ``eta.core.video.VideoLabels`` containing the
            frame qualities and motions for all frames of the video
        -   frames: a sorted list of frames that were sampled
        -   sample_edges: a sorted list of sample boundaries used for the
            sampling
    """
    #
    # Parse parameters
    #

    if motion_config is None:
        motion_config = fobm.FrameMotionConfig.default()

    if video_labels is None:
        video_labels = etav.VideoLabels()

    # Compute motion of every frame
    frame_motions = _compute_video_motion(
        video_path, video_labels, motion_config
    )

    # Select frames to sample
    frames, sample_edges = _select_frames_by_motion(
        frame_motions,
        target_accel,
        blowout,
        offset,
        always_sample_first,
        always_sample_last,
    )

    # Sample the frames
    etav.sample_select_frames(
        video_path,
        frames,
        output_patt=output_frames_patt,
        size=size,
        fast=True,
    )

    return video_labels, frames, sample_edges


def adaptive_sample_frames_by_quality_and_motion(
    video_path,
    output_frames_patt,
    target_accel,
    video_labels=None,
    blowout=None,
    alpha=0.9,
    offset=None,
    always_sample_first=False,
    always_sample_last=False,
    size=None,
    quality_factor=1,
    quality_config=None,
    motion_config=None,
):
    """Adaptively samples frames from the given video to achieve the specified
    target acceleration.

    This implementation uses frame motion to choose sample windows whose widths
    are proportionate to frame motion; within each sampling window, frame
    quality is used to inform which frame sample.

    The frame quality and motion of every frame is stored in the output labels.

    Args:
        video_path: the path to the video
        output_frames_patt: an output pattern like ``/path/to/frames/%06d.jpg``
            to which to write the sampled frames
        target_accel: the target acceleration factor to apply when sampling
            frames. For example, a target acceleration of 2x would correspond
            to sampling every other frame, on average
        video_labels (None): an optional ``eta.core.video.VideoLabels`` to
            which to add the labels. By default, a new instance is created
        blowout (None): a blowout factor in ``(0, 1)`` that defines a region in
            which subsequent samples cannot be taken. For example, a blowout
            factor of 0.25 corresponds to a minimum sampling distance of
            ``0.25 * target_accel``. A blowout factor of 1 or greater would
            force uniform sampling
        alpha: (0.9) a frame quality factor in ``(0, 1)`` that determines how
            much weight to give to frame quality relative to uniform sampling
            when sampling frames. Setting ``alpha == 1`` corresponds to always
            selecting the frame with the desired quality, while ``alpha == 0``
            corresponds to uniform sampling at the target rate
        offset (None): an optional offset from the beginning of the video to
            set the initial sampling window. By default, there is no offset
        always_sample_first (False): whether to always sample the first frame
            of the video
        always_sample_last (False): whether to always sample the last frame of
            the video
        size (None): a desired output ``(width, height)`` of the sampled
            frames. Dimensions can be -1, in which case the input aspect ratio
            is preserved
        quality_factor (1): a quality factor in ``[0, 1]`` specifying the
            target frame qualities to sample
        quality_config (None): a
            :class:`fiftyone.brain.internal.core.quality.FrameQualityConfig`
            describing the frame quality method to use
        motion_config (None): a
            :class:`fiftyone.brain.internal.core.motion.FrameMotionConfig`
            describing the frame motion method to use

    Returns:
        a tuple of

        -   video_labels: a ``eta.core.video.VideoLabels`` containing the frame
            qualities and motions for all frames of the video
        -   frames: a sorted list of frames that were sampled
        -   sample_edges: a sorted list of sample boundaries used for the
            sampling
    """
    #
    # Parse parameters
    #

    if quality_config is None:
        quality_config = fobq.FrameQualityConfig.default()

    if motion_config is None:
        motion_config = fobm.FrameMotionConfig.default()

    if video_labels is None:
        video_labels = etav.VideoLabels()

    # Compute frame qualities and motions of every frame
    frame_qualities, frame_motions = _compute_video_quality_and_motion(
        video_path, video_labels, quality_config, motion_config
    )

    # Select frames to sample
    frames, sample_edges = _select_frames_by_quality_and_motion(
        frame_qualities,
        frame_motions,
        target_accel,
        alpha,
        blowout,
        offset,
        quality_factor,
        always_sample_first,
        always_sample_last,
    )

    # Sample the frames
    etav.sample_select_frames(
        video_path,
        frames,
        output_patt=output_frames_patt,
        size=size,
        fast=True,
    )

    return video_labels, frames, sample_edges


def partition_sample_weights(weights, num_bins, min_bin_width=0):
    """Partitions the given weights into bins such that each bin contains
    approximately equal weight.

    Args:
        weights: an array of weights, which must be nonnegative and not all
            zero
        num_bins: the desired number of bins
        min_bin_width (0): the minimum allowed bin width, defined as the
            difference between the left and right bin edges (exclusive of the
            endpoints)

    Returns:
        a sorted list of length ``num_bins + 1`` containing the bin edges. Note
        that, if a large minimum bin width is used, the length of the returned
        list may be less than the target number
    """
    weights = np.array(weights, dtype=np.float)  # copy weights
    if min(weights) < 0 or max(weights) <= 0:
        raise ValueError("Weights must be nonnegative and not all zero")

    edges = np.array([], dtype=int)
    while True:
        # Select new edges
        cum_weights = np.cumsum(weights)
        num_edges = num_bins - len(edges)
        edge_weights = np.linspace(
            0, cum_weights[-1], num=num_edges, endpoint=False
        )
        new_edges = np.searchsorted(cum_weights, edge_weights, side="right")
        edges = np.concatenate((edges, new_edges))
        edges.sort()

        # Marked sampled points
        weights[edges] = 0

        # Apply blowout
        keep_inds = []
        last_edge = -float("inf")
        for idx, edge in enumerate(edges):
            if edge - last_edge - 1 >= min_bin_width:
                keep_inds.append(idx)
                last_edge = edge

        edges = edges[keep_inds]

        # Check exit criteria
        if len(edges) >= num_bins or max(weights) <= 0:
            break

    edges = list(edges)
    end = len(weights) - 1
    if not edges or edges[-1] != end:
        edges.append(end)

    return edges


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
    quality_factor = parameters.quality_factor
    quality_config = parameters.quality_config
    blowout = parameters.blowout
    alpha = parameters.alpha
    always_sample_first = parameters.always_sample_first
    always_sample_last = parameters.always_sample_last

    # Sample frames
    if not use_motion and not use_quality:
        logger.info("Uniformly sampling frames")
        frames = uniform_sample_frames(
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
        _, frames, _ = adaptive_sample_frames_by_motion(
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
        _, frames, _ = adaptive_sample_frames_by_quality(
            video_path,
            output_frames_patt,
            target_accel,
            video_labels=video_labels,
            blowout=blowout,
            alpha=alpha,
            offset=offset,
            always_sample_first=always_sample_first,
            always_sample_last=always_sample_last,
            size=size,
            quality_factor=quality_factor,
            quality_config=quality_config,
        )
    elif use_motion and use_quality:
        logger.info("Using frame quality and motion to inform sampling")
        _, frames, _ = adaptive_sample_frames_by_quality_and_motion(
            video_path,
            output_frames_patt,
            target_accel,
            video_labels=video_labels,
            blowout=blowout,
            alpha=alpha,
            offset=offset,
            always_sample_first=always_sample_first,
            always_sample_last=always_sample_last,
            size=size,
            quality_factor=quality_factor,
            quality_config=quality_config,
            motion_config=motion_config,
        )
    else:
        raise ValueError("Invalid sampling parameters...")

    return {f: output_frames_patt % f for f in frames}


def _compute_blowout_width(blowout, target_accel):
    if blowout is None:
        return 0

    blowout_width = int((target_accel - 1) * blowout)
    return max(0, min(blowout_width, int(target_accel) - 1))


def _select_frames_uniformly(
    total_frame_count,
    target_accel,
    offset,
    always_sample_first,
    always_sample_last,
):
    if total_frame_count <= 0:
        return []

    if offset is None:
        offset = 0

    sample_points = np.arange(offset + 1, total_frame_count + 1, target_accel)
    sample_frames = set(int(round(x)) for x in sample_points)

    if always_sample_first:
        sample_frames.add(1)

    if always_sample_last:
        sample_frames.add(total_frame_count)

    return sorted(sample_frames)


def _select_frames_by_motion(
    frame_motions,
    target_accel,
    blowout,
    offset,
    always_sample_first,
    always_sample_last,
):
    if not frame_motions:
        return [], []

    if offset is None:
        offset = 0

    # Compute blowout width, in frames
    blowout_width = _compute_blowout_width(blowout, target_accel)

    # Partition frames by motion
    total_frame_count = len(frame_motions)
    num_samples = int(round(max(total_frame_count - offset, 0) / target_accel))
    if num_samples > 1:
        edges = partition_sample_weights(
            frame_motions[offset:],
            num_samples - 1,
            min_bin_width=blowout_width,
        )
        sample_edges = [e + 1 + offset for e in edges]
    else:
        sample_edges = [min(1 + offset, total_frame_count)]

    # Sample edges
    frames = copy(sample_edges)

    if always_sample_first and frames[0] != 1:
        frames = [1] + frames

    if always_sample_last and frames[-1] != total_frame_count:
        frames.append(total_frame_count)

    return frames, sample_edges


def _select_frames_by_quality_and_motion(
    frame_qualities,
    frame_motions,
    target_accel,
    alpha,
    blowout,
    offset,
    quality_factor,
    always_sample_first,
    always_sample_last,
):
    if not frame_qualities:
        return [], []

    if offset is None:
        offset = 0

    # Compute blowout width, in frames
    blowout_width = _compute_blowout_width(blowout, target_accel)

    # Lookback window size over which to compute frame quality statistics
    lookback_size = int(10 * target_accel)

    # Partition frames by motion
    total_frame_count = len(frame_qualities)
    num_samples = int(round(max(total_frame_count - offset, 0) / target_accel))
    if num_samples >= 1:
        edges = partition_sample_weights(
            frame_motions[offset:], num_samples, min_bin_width=blowout_width
        )
        sample_edges = [e + 1 + offset for e in edges]
    else:
        sample_edges = [min(1 + offset, total_frame_count), total_frame_count]

    # Select frames
    frames = []
    last_sample = None
    for last_boundary, decision_frame in zip(sample_edges, sample_edges[1:]):
        if last_sample is not None:
            blowout_boundary = max(last_sample + blowout_width, last_boundary)
        else:
            blowout_boundary = last_boundary

        window_size = max(1, decision_frame - blowout_boundary)
        frame_scores = frame_qualities[:decision_frame]
        sample_frame_number, _ = _select_best_frame_in_window(
            frame_scores,
            last_sample,
            decision_frame,
            window_size,
            lookback_size,
            target_accel,
            alpha,
            quality_factor,
        )

        frames.append(sample_frame_number)
        last_sample = sample_frame_number

    # Always sample first frame, if requested
    if always_sample_first and (not frames or frames[0] != 1):
        frames = [1] + frames

    # Always sample last frame, if requested
    if always_sample_last and (not frames or frames[-1] != total_frame_count):
        frames.append(total_frame_count)

    return frames, sample_edges


def _select_best_frame_in_window(
    frame_scores,
    last_sample,
    frame_number,
    window_size,
    lookback_size,
    target_accel,
    alpha,
    quality_factor,
):
    frame_scores = np.asarray(frame_scores)
    lookback_scores = frame_scores[-lookback_size:]
    window_scores = frame_scores[-window_size:]
    window_size_actual = len(window_scores)

    # +1 point for closeness to `quality_factor`, up to a maximum of
    # `window_size_actual` points
    lookback_scores.sort()
    score_percentile = np.searchsorted(
        lookback_scores, window_scores, side="right"
    ) / len(lookback_scores)
    pos_scores = (
        window_size_actual
        * max(quality_factor, 1 - quality_factor)
        * (1 - np.abs(score_percentile - quality_factor))
    )

    # -1 point for each frame away from non-uniform spacing
    if last_sample is not None:
        last_gap = frame_number - last_sample
        sample_gaps = np.array(range(last_gap - window_size_actual, last_gap))
        neg_scores = np.abs(sample_gaps - target_accel)
    else:
        neg_scores = 0

    # Select frame with maximal score
    scores = alpha * pos_scores - (1 - alpha) * neg_scores
    idx = np.argmax(scores)
    sample_frame_number = frame_number + 1 - window_size_actual + idx

    return sample_frame_number, idx


def _sample_frame(img, frame_number, output_frames_patt, size=None):
    # Resize if necessary
    if size is not None:
        img = etai.resize(img, *size)

    # Sample frame
    frame_path = output_frames_patt % frame_number
    etai.write(img, frame_path)


def _compute_video_motion(video_path, video_labels, motion_config):
    if motion_config.motion_method != "fgsupport":
        raise ValueError(
            "Unsupported motion_method = '%s'" % motion_config.motion_method
        )

    background_subtractor_cls = etau.get_class(
        motion_config.background_subtractor
    )

    # Compute frame quality and motion
    frame_motions = []
    with background_subtractor_cls() as bgs:
        with etav.FFmpegVideoReader(video_path) as vr:
            for idx, img in enumerate(vr):
                # Compute frame motion
                motion_attr = _compute_frame_motion(
                    img, bgs, motion_config.attr_name
                )
                frame_motions.append(motion_attr.value)

                if idx >= motion_config.init_buffer_frames:
                    # Only store motion after initial buffer
                    video_labels.add_frame_attribute(
                        motion_attr, vr.frame_number
                    )

    return frame_motions


def _compute_video_quality_and_motion(
    video_path, video_labels, quality_config, motion_config
):
    if motion_config.motion_method != "fgsupport":
        raise ValueError(
            "Unsupported motion_method = '%s'" % motion_config.motion_method
        )

    background_subtractor_cls = etau.get_class(
        motion_config.background_subtractor
    )

    # Compute frame quality and motion
    frame_qualities = []
    frame_motions = []
    with background_subtractor_cls() as bgs:
        with etav.FFmpegVideoReader(video_path) as vr:
            for idx, img in enumerate(vr):
                # Compute frame quality
                quality_attr = _compute_frame_quality(
                    img, quality_config.attr_name, quality_config.method
                )
                frame_qualities.append(quality_attr.value)
                video_labels.add_frame_attribute(quality_attr, vr.frame_number)

                # Compute frame motion
                motion_attr = _compute_frame_motion(
                    img, bgs, motion_config.attr_name
                )
                frame_motions.append(motion_attr.value)

                if idx >= motion_config.init_buffer_frames:
                    # Only store motion after initial buffer
                    video_labels.add_frame_attribute(
                        motion_attr, vr.frame_number
                    )

    return frame_qualities, frame_motions


def _compute_frame_quality(img, attr_name, method):
    quality = fobi.compute_quality(img, method=method)
    return etad.NumericAttribute(attr_name, quality)


def _compute_frame_motion(img, bgs, attr_name):
    # Motion = proportion of foreground pixels
    fgmask, _ = bgs.process_frame(img)
    motion = np.count_nonzero(fgmask) / fgmask.size
    return etad.NumericAttribute(attr_name, motion)
