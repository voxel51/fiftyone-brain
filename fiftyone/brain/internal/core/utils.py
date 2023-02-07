"""
Utilities.

| Copyright 2017-2023, Voxel51, Inc.
| `voxel51.com <https://voxel51.com/>`_
|
"""
import itertools
import logging

import numpy as np

import eta.core.utils as etau

import fiftyone.core.fields as fof
import fiftyone.core.labels as fol
import fiftyone.core.patches as fop
import fiftyone.zoo as foz
from fiftyone import ViewField as F


logger = logging.getLogger(__name__)


def get_ids(
    samples,
    patches_field=None,
    data=None,
    data_type="embeddings",
    handle_missing="skip",
    ref_sample_ids=None,
):
    if patches_field is None:
        if ref_sample_ids is not None:
            sample_ids = ref_sample_ids
        else:
            sample_ids = samples.values("id")

        if data is not None and len(sample_ids) != len(data):
            raise ValueError(
                "The number of %s (%d) in these results no longer matches the "
                "number of samples (%d) in the collection. You must "
                "regenerate the results"
                % (data_type, len(data), len(sample_ids))
            )

        return np.array(sample_ids), None

    sample_ids, label_ids = _get_patch_ids(
        samples,
        patches_field,
        handle_missing=handle_missing,
        ref_sample_ids=ref_sample_ids,
    )

    if data is not None and len(sample_ids) != len(data):
        raise ValueError(
            "The number of %s (%d) in these results no longer matches the "
            "number of labels (%d) in the '%s' field of the collection. You "
            "must regenerate the results"
            % (data_type, len(data), len(sample_ids), patches_field)
        )

    return np.array(sample_ids), np.array(label_ids)


def filter_ids(
    samples,
    index_sample_ids,
    index_label_ids,
    patches_field=None,
    allow_missing=True,
    warn_missing=False,
):
    _validate_args(samples, None, patches_field)

    if patches_field is None:
        if samples._is_patches:
            sample_ids = np.array(samples.values("sample_id"))
        else:
            sample_ids = np.array(samples.values("id"))

        keep_inds, good_inds, bad_ids = _parse_ids(
            sample_ids,
            index_sample_ids,
            "samples",
            allow_missing,
            warn_missing,
        )

        if bad_ids is not None:
            sample_ids = sample_ids[good_inds]

        return sample_ids, None, keep_inds, good_inds

    sample_ids, label_ids = _get_patch_ids(samples, patches_field)

    keep_inds, good_inds, bad_ids = _parse_ids(
        label_ids,
        index_label_ids,
        "labels",
        allow_missing,
        warn_missing,
    )

    if bad_ids is not None:
        sample_ids = sample_ids[good_inds]
        label_ids = label_ids[good_inds]

    return sample_ids, label_ids, keep_inds, good_inds


def _get_patch_ids(
    samples, patches_field, handle_missing="skip", ref_sample_ids=None
):
    if samples._is_patches:
        sample_id_path = "sample_id"
    else:
        sample_id_path = "id"

    label_type, label_id_path = samples._get_label_field_path(
        patches_field, "id"
    )
    is_list_field = issubclass(label_type, fol._LABEL_LIST_FIELDS)

    sample_ids, label_ids = samples.values([sample_id_path, label_id_path])

    if ref_sample_ids is not None:
        sample_ids, label_ids = _apply_ref_sample_ids(
            sample_ids, label_ids, ref_sample_ids
        )

    if is_list_field:
        sample_ids, label_ids = _flatten_list_ids(
            sample_ids, label_ids, handle_missing
        )

    return np.array(sample_ids), np.array(label_ids)


def _apply_ref_sample_ids(sample_ids, label_ids, ref_sample_ids):
    ref_label_ids = [None] * len(ref_sample_ids)
    inds_map = {_id: i for i, _id in enumerate(ref_sample_ids)}
    for _id, _lid in zip(sample_ids, label_ids):
        idx = inds_map.get(_id, None)
        if idx is not None:
            ref_label_ids[idx] = _lid

    return ref_sample_ids, ref_label_ids


def _flatten_list_ids(sample_ids, label_ids, handle_missing):
    _sample_ids = []
    _label_ids = []
    _add_missing = handle_missing == "image"

    for _id, _lids in zip(sample_ids, label_ids):
        if _lids:
            for _lid in _lids:
                _sample_ids.append(_id)
                _label_ids.append(_lid)
        elif _add_missing:
            _sample_ids.append(_id)
            _label_ids.append(None)

    return _sample_ids, _label_ids


def _parse_ids(ids, index_ids, ftype, allow_missing, warn_missing):
    if np.array_equal(ids, index_ids):
        return None, None, None

    inds_map = {_id: idx for idx, _id in enumerate(index_ids)}

    keep_inds = []
    bad_inds = []
    bad_ids = []
    for _idx, _id in enumerate(ids):
        ind = inds_map.get(_id, None)
        if ind is not None:
            keep_inds.append(ind)
        else:
            bad_inds.append(_idx)
            bad_ids.append(_id)

    num_missing_index = len(index_ids) - len(keep_inds)
    if num_missing_index > 0:
        if not allow_missing:
            raise ValueError(
                "The index contains %d %s that are not present in the "
                "provided collection" % (num_missing_index, ftype)
            )

        if warn_missing:
            logger.warning(
                "Ignoring %d %s from the index that are not present in the "
                "provided collection",
                num_missing_index,
                ftype,
            )

    num_missing_collection = len(bad_ids)
    if num_missing_collection > 0:
        if not allow_missing:
            raise ValueError(
                "The provided collection contains %d %s not present in the "
                "index" % (num_missing_collection, ftype)
            )

        if warn_missing:
            logger.warning(
                "Ignoring %d %s from the provided collection that are not "
                "present in the index",
                num_missing_collection,
                ftype,
            )

        bad_inds = np.array(bad_inds, dtype=np.int64)

        good_inds = np.full(ids.shape, True)
        good_inds[bad_inds] = False
    else:
        good_inds = None
        bad_ids = None

    keep_inds = np.array(keep_inds, dtype=np.int64)

    return keep_inds, good_inds, bad_ids


def filter_values(values, keep_inds, patches_field=None):
    if patches_field:
        _values = list(itertools.chain.from_iterable(values))
    else:
        _values = values

    _values = np.asarray(_values)

    if _values.size == keep_inds.size:
        _values = _values[keep_inds]
    else:
        num_expected = np.count_nonzero(keep_inds)
        if _values.size != num_expected:
            raise ValueError(
                "Expected %d raw values or %d pre-filtered values; found %d "
                "values" % (keep_inds.size, num_expected, values.size)
            )

    # @todo we might need to re-ravel patch values here in the future
    # We currently do not do this because all downstream users of this data
    # will gracefully handle either flat or nested list data

    return _values


def get_values(samples, path_or_expr, ids, patches_field=None):
    _validate_args(samples, path_or_expr, patches_field)
    return samples._get_values_by_id(
        path_or_expr, ids, link_field=patches_field
    )


def parse_embeddings_field(
    samples, embeddings_field, patches_field=None, allow_embedded=True
):
    if not etau.is_str(embeddings_field):
        raise ValueError(
            "Invalid embeddings_field=%s; expected a string field name"
            % embeddings_field
        )

    if patches_field is None:
        _embeddings_field, is_frame_field = samples._handle_frame_field(
            embeddings_field
        )

        if not allow_embedded and "." in _embeddings_field:
            ftype = "frame" if is_frame_field else "sample"
            raise ValueError(
                "Invalid embeddings_field=%s; expected a top-level %s field "
                "name that contains no '.'" % (_embeddings_field, ftype)
            )

        return embeddings_field

    if embeddings_field.startswith(patches_field + "."):
        _, root = samples._get_label_field_path(patches_field) + "."
        if not embeddings_field.startswith(root):
            raise ValueError(
                "Invalid embeddings_field=%s for patches_field=%s"
                % (embeddings_field, patches_field)
            )

        embeddings_field = embeddings_field[len(root) + 1]

    if not allow_embedded and "." in embeddings_field:
        raise ValueError(
            "Invalid embeddings_field=%s for patches_field=%s; expected a "
            "label attribute name that contains no '.'"
            % (embeddings_field, patches_field)
        )

    return embeddings_field


def get_embeddings(
    samples,
    model=None,
    patches_field=None,
    embeddings_field=None,
    embeddings=None,
    force_square=False,
    alpha=None,
    handle_missing="skip",
    agg_fcn=None,
    batch_size=None,
    num_workers=None,
    skip_failures=True,
):
    if model is None and embeddings_field is None and embeddings is None:
        raise ValueError(
            "One of `model`, `embeddings_field`, or `embeddings` must be "
            "provided"
        )

    if model is not None:
        if etau.is_str(model):
            model = foz.load_zoo_model(model)

        if patches_field is not None:
            logger.info("Computing patch embeddings...")
            embeddings = samples.compute_patch_embeddings(
                model,
                patches_field,
                embeddings_field=embeddings_field,
                force_square=force_square,
                alpha=alpha,
                handle_missing=handle_missing,
                batch_size=batch_size,
                num_workers=num_workers,
                skip_failures=skip_failures,
            )
        else:
            logger.info("Computing embeddings...")
            embeddings = samples.compute_embeddings(
                model,
                embeddings_field=embeddings_field,
                batch_size=batch_size,
                num_workers=num_workers,
                skip_failures=skip_failures,
            )

    if embeddings_field is not None:
        embeddings, samples = _load_embeddings(
            samples, embeddings_field, patches_field=patches_field
        )
        ref_sample_ids = None
    else:
        if isinstance(embeddings, dict):
            embeddings = [
                embeddings.get(_id, None) for _id in samples.values("id")
            ]

        embeddings, ref_sample_ids = _handle_missing_embeddings(
            embeddings, samples
        )

    if not isinstance(embeddings, np.ndarray) and not embeddings:
        embeddings = np.empty((0, 0), dtype=float)
        sample_ids = np.array([], dtype="<U24")
        if patches_field is not None:
            label_ids = np.array([], dtype="<U24")
        else:
            label_ids = None

        return embeddings, sample_ids, label_ids

    if patches_field is not None:
        if agg_fcn is not None:
            embeddings = np.stack([agg_fcn(e) for e in embeddings])
        else:
            embeddings = np.concatenate(embeddings, axis=0)
    elif not isinstance(embeddings, np.ndarray):
        embeddings = np.stack(embeddings)

    if agg_fcn is not None:
        patches_field = None

    sample_ids, label_ids = get_ids(
        samples,
        patches_field=patches_field,
        data=embeddings,
        data_type="embeddings",
        handle_missing=handle_missing,
        ref_sample_ids=ref_sample_ids,
    )

    return embeddings, sample_ids, label_ids


def _load_embeddings(samples, embeddings_field, patches_field=None):
    if patches_field is not None:
        label_type, embeddings_path = samples._get_label_field_path(
            patches_field, embeddings_field
        )
        is_list_field = issubclass(label_type, fol._LABEL_LIST_FIELDS)
    else:
        embeddings_path = embeddings_field
        is_list_field = False

    if is_list_field:
        samples = samples.filter_labels(
            patches_field, F(embeddings_field) != None
        )
    else:
        samples = samples.match(F(embeddings_path) != None)

    if samples.has_field(embeddings_path):
        _field = None
    else:
        _field = fof.VectorField()

    embeddings = samples.values(embeddings_path, _field=_field)

    if is_list_field:
        embeddings = [np.stack(e) for e in embeddings if e]

    return embeddings, samples


def _validate_args(samples, path_or_expr, patches_field):
    if patches_field is not None:
        _validate_patches_args(samples, path_or_expr, patches_field)
    else:
        _validate_samples_args(samples, path_or_expr)


def _validate_samples_args(samples, path_or_expr):
    if not etau.is_str(path_or_expr):
        return

    path, _, list_fields, _, _ = samples._parse_field_name(path_or_expr)

    if list_fields:
        raise ValueError(
            "Values path '%s' contains invalid list field '%s'"
            % (path, list_fields[0])
        )


def _validate_patches_args(samples, path_or_expr, patches_field):
    if etau.is_str(path_or_expr) and not path_or_expr.startswith(
        patches_field + "."
    ):
        raise ValueError(
            "Values path '%s' must start with patches field '%s'"
            % (path_or_expr, patches_field)
        )

    if (
        isinstance(samples, fop.PatchesView)
        and patches_field != samples.patches_field
    ):
        raise ValueError(
            "This patches view contains labels from field '%s', not "
            "'%s'" % (samples.patches_field, patches_field)
        )

    if isinstance(
        samples, fop.EvaluationPatchesView
    ) and patches_field not in (
        samples.gt_field,
        samples.pred_field,
    ):
        raise ValueError(
            "This evaluation patches view contains patches from "
            "fields '%s' and '%s', not '%s'"
            % (samples.gt_field, samples.pred_field, patches_field)
        )


def _handle_missing_embeddings(embeddings, samples):
    if isinstance(embeddings, np.ndarray):
        return embeddings, None

    missing_inds = []
    for idx, embedding in enumerate(embeddings):
        if embedding is None:
            missing_inds.append(idx)

    if not missing_inds:
        return embeddings, None

    embeddings = [e for e in embeddings if e is not None]
    ref_sample_ids = list(np.delete(samples.values("id"), missing_inds))

    return embeddings, ref_sample_ids
