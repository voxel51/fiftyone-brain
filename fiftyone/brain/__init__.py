"""
The brains behind FiftyOne: a powerful package for dataset curation, analysis,
and visualization.

See https://github.com/voxel51/fiftyone for more information.

| Copyright 2017-2020, Voxel51, Inc.
| `voxel51.com <https://voxel51.com/>`_
|
"""


def compute_hardness(samples, label_field, hardness_field="hardness"):
    """Adds a hardness field to each sample scoring the difficulty that the
    specified label field observed in classifying the sample.

    Hardness is a measure computed based on model prediction output (through
    logits) that summarizes a measure of the uncertainty the model had with the
    sample. This makes hardness quantitative and can be used to detect things
    like hard samples, annotation errors during noisy training, and more.

    Args:
        samples: a :class:`fiftyone.core.collections.SampleCollection`
        label_field: the :class:`fiftyone.core.labels.Classification` or
            :class:`fiftyone.core.labels.Classifications` field to use from
            each sample
        hardness_field ("hardness"): the field name to use to store the
            hardness value for each sample
    """
    import fiftyone.brain.internal.core.hardness as fbh

    fbh.compute_hardness(samples, label_field, hardness_field)


def compute_mistakenness(
    samples,
    pred_field,
    label_field="ground_truth",
    mistakenness_field="mistakenness",
    missing_field="possible_missing",
    spurious_field="possible_spurious",
    use_logits=True,
):
    """Computes the mistakenness of the labels in the specified
    ``label_field``, scoring the chance that the labels are incorrect.

    Mistakenness is computed based on the predictions in the ``pred_field``,
    through its ``logits`` or ``confidence``. This measure can be used to
    detect things like annotation errors and unusually hard samples.

    This method supports both classifications and detections.

    For classifications, a ``mistakenness_field`` field is populated on each
    sample that quantifies the likelihood that the label in the ``label_field``
    of that sample is incorrect.

    For detections, the mistakenness of each detection in ``label_field`` is
    computed, using :meth:`fiftyone.utils.evaluation.evaluate_detections` to
    locate corresponding detections in ``pred_field``. Three types of mistakes
    are identified:

    -   **(Mistakes)** Detections with a match in ``pred_field`` are assigned a
        mistakenness value in their ``mistakenness_field``, which captures the
        likelihood that the detection in ``label_field`` is a mistake. Such
        mistakes may be due to either the class label or localization of the
        detection

    -   **(Missing)** Detections in ``pred_field`` with no matches in
        ``label_field`` but which are likely to be correct are *added* to
        ``label_field`` and given a value of ``True`` in their
        ``missing_field`` attribute

    -   **(Spurious)** Detections in ``label_field`` with no matches in
        ``pred_field`` but which are likely to be incorrect are given a value
        of ``True`` in their ``spurious_field`` attribute

    These per-detection data are then aggregated at the sample-level as
    follows:

    -   **(Mistakes)** The ``mistakenness_field`` of each sample is populated
        with the maximum mistakenness of the detections in ``label_field``

    -   **(Missing)** The ``missing_field`` of each sample is populated with
        the number of missing detections that were deemed missing and thus
        added to ``label_field``

    -   **(Spurious)** The ``spurious_field`` of each sample is populated with
        the number of detections in ``label_field`` that were given deemed
        spurious

    Args:
        samples: a :class:`fiftyone.core.collections.SampleCollection`
        pred_field: the name of the predicted label field to use from each
            sample. Can be of type
            :class:`fiftyone.core.labels.Classification`,
            :class:`fiftyone.core.labels.Classifications`, or
            :class:`fiftyone.core.labels.Detections`
        label_field ("ground_truth"): the name of the "ground truth" label
            field that you want to test for mistakes with respect to the
            predictions in ``pred_field``. Must have the same type as
            ``pred_field``
        mistakenness_field ("mistakenness"): the field name to use to store the
            mistakenness value for each sample
        missing_field ("possible_missing): the field in which to store
            per-sample counts of potential missing detections. Only applicable
            for :class:`fiftyone.core.labels.Detections` labels
        spurious_field ("possible_spurious): the field in which to store
            per-sample counts of potential spurious detections. Only applicable
            for :class:`fiftyone.core.labels.Detections` labels
        use_logits (True): whether to use logits (True) or confidence (False)
            to compute mistakenness. Logits typically yield better results,
            when they are available
    """
    import fiftyone.brain.internal.core.mistakenness as fbm

    fbm.compute_mistakenness(
        samples,
        pred_field,
        label_field,
        mistakenness_field,
        missing_field,
        spurious_field,
        use_logits,
    )


def compute_uniqueness(samples, uniqueness_field="uniqueness", roi_field=None):
    """Adds a uniqueness field to each sample scoring how unique it is with
    respect to the rest of the samples.

    This function only uses the pixel data and can therefore process labeled or
    unlabeled samples.

    Args:
        samples: a :class:`fiftyone.core.collections.SampleCollection`
        uniqueness_field ("uniqueness"): the field name to use to store the
            uniqueness value for each sample
        roi_field (None): an optional :class:`fiftyone.core.labels.Detection`,
            :class:`fiftyone.core.labels.Detections`,
            :class:`fiftyone.core.labels.Polyline`, or
            :class:`fiftyone.core.labels.Polylines` field defining a region of
            interest within each image to use to compute uniqueness
    """
    import fiftyone.brain.internal.core.uniqueness as fbu

    fbu.compute_uniqueness(samples, uniqueness_field, roi_field)


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
    import fiftyone.brain.internal.core.sampling as fbs

    return fbs.sample_best_video_frames(
        video_path,
        out_frames_dir,
        quality_factor=quality_factor,
        target_num_frames=target_num_frames,
        target_accel=target_accel,
        target_fps=target_fps,
        size=size,
        max_size=max_size,
    )
