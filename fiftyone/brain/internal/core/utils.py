"""
Core utilities.

| Copyright 2017-2020, Voxel51, Inc.
| `voxel51.com <https://voxel51.com/>`_
|
"""
import os

import fiftyone.core.collections as foc
import fiftyone.core.fields as fof
import fiftyone.core.labels as fol
import fiftyone.core.media as fom


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


def validate_image(sample):
    """Validates that the sample's media is an image that exists on disk.

    Args:
        sample: a :class:`fiftyone.core.sample.Sample`

    Raises:
        ValueError if the required conditions are not met
    """
    if not os.path.exists(sample.filepath):
        raise ValueError(
            "Sample '%s' source media '%s' does not exist on disk"
            % (sample.id, sample.filepath)
        )

    if sample.media_type != fom.IMAGE:
        raise ValueError(
            "Sample '%s' source media '%s' is not a recognized image format"
            % (sample.id, sample.filepath)
        )


def validate_video(sample):
    """Validates that the sample's media is a video that exists on disk.

    Args:
        sample: a :class:`fiftyone.core.sample.Sample`

    Raises:
        ValueError if the required conditions are not met
    """
    if not os.path.exists(sample.filepath):
        raise ValueError(
            "Sample '%s' source media '%s' does not exist on disk"
            % (sample.id, sample.filepath)
        )

    if sample.media_type != fom.VIDEO:
        raise ValueError(
            "Sample '%s' source media '%s' is not a recognized video format"
            % (sample.id, sample.filepath)
        )


def validate_collection_label_fields(
    sample_collection, field_names, allowed_label_types, same_type=False
):
    """Validates that the :class:`fiftyone.core.collections.SampleCollection`
    has fields with the specified :class:`fiftyone.core.labels.Label` types.

    Args:
        sample_collection: a
            :class:`fiftyone.core.collections.SampleCollection`
        field_names: an iterable of field names
        allowed_label_types: an iterable of allowed
            :class:`fiftyone.core.labels.Label` types
        same_type (False): whether to enforce that all fields have same type

    Raises:
        ValueError if the required conditions are not met
    """
    label_fields = sample_collection.get_field_schema(
        ftype=fof.EmbeddedDocumentField, embedded_doc_type=fol.Label
    )

    label_types = {}
    for field_name in field_names:
        if field_name not in label_fields:
            raise ValueError(
                "%s '%s' has no label field '%s'"
                % (
                    sample_collection.__class__.__name__,
                    sample_collection.name,
                    field_name,
                )
            )

        label_type = label_fields[field_name].document_type
        label_types[field_name] = label_type

        if label_type not in allowed_label_types:
            raise ValueError(
                "%s '%s' field '%s' is not a %s instance; found %s"
                % (
                    sample_collection.__class__.__name__,
                    sample_collection.name,
                    field_name,
                    allowed_label_types,
                    label_type,
                )
            )

    if same_type and len(set(label_types.values())) > 1:
        raise ValueError(
            "%s '%s' fields %s must have the same type; found %s"
            % (
                sample_collection.__class__.__name__,
                sample_collection.name,
                field_names,
                label_types,
            )
        )


def get_field(sample, field_name, allowed_types=None, allow_none=True):
    """Gets the given sample field and optionally validates their type and
    value.

    Args:
        sample: a :class:`fiftyone.core.sample.Sample`
        field_name: the name of the field to get
        allowed_types (None): an optional iterable of
            :class:`fiftyone.core.labels.Label` types to enforce that the field
            value has
        allow_none (True): whether to allow the field to be None

    Returns:
        the field value

    Raises:
        ValueError if the field does not exist or does not meet the specified
        criteria
    """
    try:
        value = sample[field_name]
    except KeyError:
        raise ValueError(
            "Sample '%s' has no field '%s'" % (sample.id, field_name)
        )

    if not allow_none and value is None:
        raise ValueError(
            "Sample '%s' field '%s' is None" % (sample.id, field_name)
        )

    if allowed_types is not None:
        field_type = type(value)
        if field_type not in allowed_types:
            raise ValueError(
                "Sample '%s' field '%s' is not a %s instance; found %s"
                % (sample.id, field_name, allowed_types, field_type)
            )

    return value


def get_fields(
    sample, field_names, allowed_types=None, same_type=False, allow_none=True
):
    """Gets the given sample fields and optionally validates their types and
    values.

    Args:
        sample: a :class:`fiftyone.core.sample.Sample`
        field_names: an iterable of field names to get
        allowed_types (None): an optional iterable of
            :class:`fiftyone.core.labels.Label` types to enforce that the
            field values have
        same_type (False): whether to enforce that all fields have same type
        allow_none (True): whether to allow the fields to be None

    Returns:
        a tuple of field values

    Raises:
        ValueError if a field does not exist or does not meet the specified
        criteria
    """
    label_types = {}
    values = []
    for field_name in field_names:
        value = get_field(
            sample,
            field_name,
            allowed_types=allowed_types,
            allow_none=allow_none,
        )

        if same_type:
            label_types[field_name] = type(value)
            values.append(value)

    if same_type and len(set(label_types.values())) > 1:
        raise ValueError(
            "Sample '%s' fields %s must have the same type; found %s"
            % (sample.id, field_names, label_types)
        )

    return tuple(values)
