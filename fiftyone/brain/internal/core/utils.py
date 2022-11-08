"""
Utilities.

| Copyright 2017-2022, Voxel51, Inc.
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


def get_ids(samples, patches_field=None):
    if patches_field is None:
        sample_ids = np.array(samples.values("id"))
        return sample_ids, None

    sample_ids = []
    label_ids = []
    for l in samples._get_selected_labels(fields=patches_field):
        sample_ids.append(l["sample_id"])
        label_ids.append(l["label_id"])

    return np.array(sample_ids), np.array(label_ids)


def filter_ids(
    view,
    samples,
    index_sample_ids,
    index_label_ids,
    patches_field=None,
    allow_missing=False,
):
    # No filtering required
    if view == samples or view.view() == samples.view():
        return view, index_sample_ids, index_label_ids, None, None

    if patches_field is None:
        _sample_ids = np.array(view.values("id"))
        keep_inds, good_inds, bad_ids = _parse_ids(
            _sample_ids, index_sample_ids, allow_missing
        )

        if bad_ids is not None:
            _sample_ids = _sample_ids[good_inds]
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
        _label_ids, index_label_ids, allow_missing
    )

    if bad_ids is not None:
        _sample_ids = _sample_ids[good_inds]
        _label_ids = _label_ids[good_inds]
        view = view.exclude_labels(ids=bad_ids, fields=patches_field)

    return view, _sample_ids, _label_ids, keep_inds, good_inds


def _parse_ids(ids, index_ids, allow_missing):
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
            "The provided collection contains %d IDs (eg '%s') not present in "
            "the index" % (len(bad_ids), bad_ids[0])
        )

    logger.warning(
        "Ignoring %d IDs from the provided collection that are not present in "
        "the index",
        len(bad_ids),
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
