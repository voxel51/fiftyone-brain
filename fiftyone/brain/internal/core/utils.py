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

import fiftyone.core.patches as fop
import fiftyone.zoo as foz
from fiftyone import ViewField as F


logger = logging.getLogger(__name__)


def parse_data(
    samples,
    data,
    sample_ids=None,
    label_ids=None,
    patches_field=None,
    data_type="embeddings",
    allow_missing=False,
    warn_missing=True,
):
    if sample_ids is None:
        sample_ids, label_ids = get_ids(
            samples,
            patches_field=patches_field,
            data=data,
            data_type=data_type,
        )

        return samples, data, sample_ids, label_ids

    samples, sample_ids, label_ids, keep_inds, good_inds = filter_ids(
        samples,
        sample_ids,
        label_ids,
        patches_field=patches_field,
        allow_missing=True,
        warn_missing=False,
    )

    if good_inds is not None:
        num_missing = good_inds.size - np.count_nonzero(good_inds)
        ftype = "samples" if patches_field is None else "labels"

        if not allow_missing:
            raise ValueError(
                "The provided collection contains %d %s whose %s are not "
                "present in the index" % (num_missing, ftype, data_type)
            )

        if warn_missing:
            logger.warning(
                "Ignoring %d %s from the provided collection whose %s are not "
                "present in the index",
                num_missing,
                ftype,
                data_type,
            )

    if keep_inds is not None:
        num_missing = len(data) - len(keep_inds)
        if num_missing > 0:
            ftype = "samples" if patches_field is None else "labels"

            if not allow_missing:
                raise ValueError(
                    "The index contains %d %s whose %s are no longer present "
                    "in the provided collection"
                    % (num_missing, data_type, ftype)
                )

            if warn_missing:
                logger.warning(
                    "Ignoring %d %s from the index whose %s are not present "
                    "in the provided collection",
                    num_missing,
                    data_type,
                    ftype,
                )

        data = data[keep_inds, :]

    return samples, data, sample_ids, label_ids


def get_ids(samples, patches_field=None, data=None, data_type="embeddings"):
    if patches_field is None:
        sample_ids = samples.values("id")

        if data is not None and len(sample_ids) != len(data):
            raise ValueError(
                "The number of %s (%d) does not match the number of samples "
                "(%d) in the collection "
                % (data_type, len(data), len(sample_ids))
            )

        return np.array(sample_ids), None

    sample_ids = []
    label_ids = []
    for l in samples._get_selected_labels(fields=patches_field):
        sample_ids.append(l["sample_id"])
        label_ids.append(l["label_id"])

    if data is not None and len(sample_ids) != len(data):
        raise ValueError(
            "The number of %s (%d) does not match the number of labels (%d) "
            "in the '%s' field of the collection"
            % (data_type, len(data), len(sample_ids), patches_field)
        )

    return np.array(sample_ids), np.array(label_ids)


def filter_ids(
    view,
    index_sample_ids,
    index_label_ids,
    index_samples=None,
    patches_field=None,
    allow_missing=False,
    warn_missing=True,
):
    # No filtering required
    if index_samples is not None and (
        view == index_samples or view.view() == index_samples.view()
    ):
        return view, index_sample_ids, index_label_ids, None, None

    if patches_field is None:
        if view._is_patches:
            _sample_ids = np.array(view.values("sample_id"))
        else:
            _sample_ids = np.array(view.values("id"))

        keep_inds, good_inds, bad_ids = _parse_ids(
            _sample_ids,
            index_sample_ids,
            "samples",
            allow_missing,
            warn_missing,
        )

        if bad_ids is not None:
            _sample_ids = _sample_ids[good_inds]

            if view._is_patches:
                view = view.exclude_by("sample_id", bad_ids)
            else:
                view = view.exclude(bad_ids)

        return view, _sample_ids, None, keep_inds, good_inds

    # Filter labels in patches view

    if (
        isinstance(view, fop.PatchesView)
        and patches_field != view.patches_field
    ):
        raise ValueError(
            "This patches view contains labels from field '%s', not "
            "'%s'" % (view.patches_field, patches_field)
        )

    if isinstance(view, fop.EvaluationPatchesView) and patches_field not in (
        view.gt_field,
        view.pred_field,
    ):
        raise ValueError(
            "This evaluation patches view contains patches from "
            "fields '%s' and '%s', not '%s'"
            % (view.gt_field, view.pred_field, patches_field)
        )

    labels = view._get_selected_labels(fields=patches_field)
    _sample_ids = np.array([l["sample_id"] for l in labels])
    _label_ids = np.array([l["label_id"] for l in labels])

    keep_inds, good_inds, bad_ids = _parse_ids(
        _label_ids,
        index_label_ids,
        "labels",
        allow_missing,
        warn_missing,
    )

    if bad_ids is not None:
        _sample_ids = _sample_ids[good_inds]
        _label_ids = _label_ids[good_inds]
        view = view.exclude_labels(ids=bad_ids, fields=patches_field)

    return view, _sample_ids, _label_ids, keep_inds, good_inds


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

    keep_inds = np.array(keep_inds, dtype=np.int64)

    if not bad_inds:
        return keep_inds, None, None

    if not allow_missing:
        raise ValueError(
            "The provided collection contains %d %s not present in the index"
            % (len(bad_ids), ftype)
        )

    if warn_missing:
        logger.warning(
            "Ignoring %d %s from the provided collection that are not present "
            "in the index",
            len(bad_ids),
            ftype,
        )

    bad_inds = np.array(bad_inds, dtype=np.int64)

    good_inds = np.full(ids.shape, True)
    good_inds[bad_inds] = False

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
    elif embeddings_field is not None:
        embeddings = samples.values(embeddings_field)

    if embeddings is None:
        raise ValueError(
            "One of `model`, `embeddings_field`, or `embeddings` must be "
            "provided"
        )

    if isinstance(embeddings, dict):
        embeddings = [
            embeddings.get(_id, None) for _id in samples.values("id")
        ]

    if patches_field is not None:
        _handle_missing_patch_embeddings(embeddings, samples, patches_field)

        if agg_fcn is not None:
            embeddings = [agg_fcn(e) for e in embeddings]
            embeddings = np.stack(embeddings)
        else:
            embeddings = np.concatenate(embeddings, axis=0)
    else:
        _handle_missing_embeddings(embeddings)

        if agg_fcn is not None:
            embeddings = [agg_fcn(e) for e in embeddings]

        embeddings = np.stack(embeddings)

    return embeddings


def _handle_missing_embeddings(embeddings):
    if isinstance(embeddings, np.ndarray):
        return

    missing_inds = []
    num_dims = None
    for idx, embedding in enumerate(embeddings):
        if embedding is None:
            missing_inds.append(idx)
        elif num_dims is None:
            num_dims = embedding.size

    if not missing_inds:
        return

    missing_embedding = np.zeros(num_dims or 16)
    for idx in missing_inds:
        embeddings[idx] = missing_embedding.copy()

    logger.warning("Using zeros for %d missing embeddings", len(missing_inds))


def _handle_missing_patch_embeddings(embeddings, samples, patches_field):
    missing_inds = []
    num_dims = None
    for idx, embedding in enumerate(embeddings):
        if embedding is None:
            missing_inds.append(idx)
        elif num_dims is None:
            num_dims = embedding.shape[1]

    if not missing_inds:
        return

    missing_embedding = np.zeros(num_dims or 16)

    _, labels_path = samples._get_label_field_path(patches_field)
    patch_counts = samples.values(F(labels_path).length())

    num_missing = 0
    for idx in missing_inds:
        count = patch_counts[idx]
        embeddings[idx] = np.tile(missing_embedding, (count, 1))
        num_missing += count

    if num_missing > 0:
        logger.warning(
            "Using zeros for %d missing patch embeddings", num_missing
        )
