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
        samples: an iterable of :class:`fiftyone.core.sample.Sample` instances
        label_field: the :class:`fiftyone.core.labels.Classification` or
            :class:`fiftyone.core.labels.Classifications` field to use from
            each sample
        hardness_field ("hardness"): the field name to use to store the
            hardness value for each sample
    """
    import fiftyone.brain.internal.core.hardness as fbh

    fbh.compute_hardness(samples, label_field, hardness_field=hardness_field)


def compute_mistakenness(
    samples,
    pred_field,
    label_field="ground_truth",
    mistakenness_field="mistakenness",
):
    """Adds a mistakenness field to each sample scoring the chance that the
    specified label field is incorrect.

    Mistakenness is computed based on the prediction output of a model (through
    logits) provided in the ``pred_field`` field of the sample in conjunction
    with the reference "ground truth" label in the ``label_field`` field.
    This makes the measure quantitative and can be used to detect things like
    annotation errors as well as unusually hard samples.

    Args:
        samples: an iterable of :class:`fiftyone.core.sample.Sample` instances
        pred_field: the name of the predicted
            :class:`fiftyone.core.labels.Classification` or
            :class:`fiftyone.core.labels.Classifications` label field to use
            from each sample
        label_field ("ground_truth"): the name of the "ground truth"
            :class:`fiftyone.core.labels.Classification` or
            :class:`fiftyone.core.labels.Classifications` label field that you
            want to test for a mistake with respect to the prediction output
        mistakenness_field ("mistakenness"): the field name to use to store the
            mistakenness value for each sample
    """
    import fiftyone.brain.internal.core.mistakenness as fbm

    fbm.compute_mistakenness(
        samples,
        pred_field,
        label_field=label_field,
        mistakenness_field=mistakenness_field,
    )


def compute_uniqueness(samples, uniqueness_field="uniqueness", roi_field=None):
    """Adds a uniqueness field to each sample scoring how unique it is with
    respect to the rest of the samples.

    This function only uses the pixel data and can therefore process labeled or
    unlabeled samples.

    Args:
        samples: an iterable of :class:`fiftyone.core.sample.Sample` instances
        uniqueness_field ("uniqueness"): the field name to use to store the
            uniqueness value for each sample
        roi_field (None): an optional :class:`fiftyone.core.labels.Detection`,
            :class:`fiftyone.core.labels.Detections`,
            :class:`fiftyone.core.labels.Polyline`, or
            :class:`fiftyone.core.labels.Polylines` field defining a region of
            interest within each image to use to compute uniqueness
    """
    import fiftyone.brain.internal.core.uniqueness as fbu

    fbu.compute_uniqueness(
        samples, uniqueness_field=uniqueness_field, roi_field=roi_field,
    )
