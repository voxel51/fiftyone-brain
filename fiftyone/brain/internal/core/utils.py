"""
Core utilities.

| Copyright 2017-2020, Voxel51, Inc.
| `voxel51.com <https://voxel51.com/>`_
|
"""
import fiftyone.core.collections as foc


def optimize_samples(samples, fields=None):
    """Optimizes processing of the given samples by only selecting the
    requested fields (if possible).

    If the input ``samples`` is not a
    :class:`fiftyone.core.collections.SampleCollection`, this method has no
    effect.

    Required sample fields will always be included.

    Iterating over the return :class:`fiftyone.core.view.DatasetView` may be
    significantly faster than iterating over the input collection if it
    contains many large, unnecessary fields.

    Args:
        samples: an iterable of :class:`fiftyone.core.sample.Sample` instances
        fields (None): an iterable of fields to select. If ``None``, only the
            required fields will be selected

    Returns:
        a :class:`fiftyone.core.view.DatasetView`, or the input ``samples``
    """
    if isinstance(samples, foc.SampleCollection):
        return samples.select_fields(fields)

    return samples
