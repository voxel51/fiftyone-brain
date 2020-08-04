"""
Core infrastructure for adaptively sampling frames of videos.

@todo add common interface for frame sampling methods defined here

| Copyright 2017-2020, Voxel51, Inc.
| `voxel51.com <https://voxel51.com/>`_
|
"""
from copy import copy

import numpy as np

from eta.core.config import Config
import eta.core.data as etad
import eta.core.image as etai
import eta.core.utils as etau
import eta.core.video as etav

import fiftyone.brain.internal.core.image as fobi
import fiftyone.brain.internal.core.motion as fobm
import fiftyone.brain.internal.core.quality as fobq


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
        quality_config: (None) a
            :class:`fiftyone.brain.internal.core.quality.FrameQualityConfig`
            describing the quality method to use. Only used when
            ``use_quality == True``
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
    delta=None,
    offset=None,
    always_sample_first=False,
    always_sample_last=False,
    size=None,
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
        delta (None): a multiple of frame quality standard deviation increments
            that earns equal weight to a one frame deviation from the uniform
            sampling gap defined by the target acceleration. Setting delta to a
            small alue, say ``1e-5``, will always sample the maximum quality
            frame within each sampling window. Setting delta to a large value,
            say ``1e5``, will reduce to uniform sampling at the target rate.
            The default value is ``1 / target_accel``
        offset (None): an optional offset from the beginning of the video to
            set the initial sampling window. By default, there is no offset
        always_sample_first (False): whether to always sample the first frame
            of the video
        always_sample_last (False): whether to always sample the last frame of
            the video
        size (None): a desired output ``(width, height)`` of the sampled
            frames. Dimensions can be -1, in which case the input aspect ratio
            is preserved
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

    if delta is None:
        delta = 1 / target_accel

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
                delta,
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
    delta=None,
    offset=None,
    always_sample_first=False,
    always_sample_last=False,
    size=None,
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
        delta (None): a multiple of frame quality standard deviation increments
            that earns equal weight to a one frame deviation from the uniform
            sampling gap defined by the target acceleration. Setting delta to a
            small alue, say ``1e-5``, will always sample the maximum quality
            frame within each sampling window. Setting delta to a large value,
            say ``1e5``, will reduce to uniform sampling at the target rate.
            The default value is ``1 / target_accel``
        offset (None): an optional offset from the beginning of the video to
            set the initial sampling window. By default, there is no offset
        always_sample_first (False): whether to always sample the first frame
            of the video
        always_sample_last (False): whether to always sample the last frame of
            the video
        size (None): a desired output ``(width, height)`` of the sampled
            frames. Dimensions can be -1, in which case the input aspect ratio
            is preserved
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

    if delta is None:
        delta = 1 / target_accel

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
        delta,
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
    delta,
    blowout,
    offset,
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
            delta,
        )

        frames.append(sample_frame_number)
        last_sample = sample_frame_number

    # Always sample first frame, if requested
    if always_sample_first and not frames or frames[0] != 1:
        frames = [1] + frames

    # Always sample last frame, if requested
    if always_sample_last and not frames or frames[-1] != total_frame_count:
        frames.append(total_frame_count)

    return frames, sample_edges


def _select_best_frame_in_window(
    frame_scores,
    last_sample,
    frame_number,
    window_size,
    lookback_size,
    target_accel,
    delta,
):
    frame_scores = np.asarray(frame_scores)
    lookback_scores = frame_scores[-lookback_size:]
    window_scores = frame_scores[-window_size:]
    window_size_actual = len(window_scores)

    # +/-1 point for frame metric in increments of `delta` standard deviations
    # above mean over lookback window
    fqmean = np.mean(lookback_scores)
    fqstd = np.std(lookback_scores)
    scale = delta * fqstd
    if scale > 0:
        pos_scores = (window_scores - fqmean) / scale
    else:
        pos_scores = 0

    # -1 point for each frame away from non-uniform spacing
    if last_sample is not None:
        last_gap = frame_number - last_sample
        sample_gaps = np.array(range(last_gap - window_size_actual, last_gap))
        neg_scores = np.abs(sample_gaps - target_accel)
    else:
        neg_scores = 0

    # Select frame with maximal score
    scores = pos_scores - neg_scores
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
