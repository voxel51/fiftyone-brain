"""
Utilities.

| Copyright 2017-2022, Voxel51, Inc.
| `voxel51.com <https://voxel51.com/>`_
|
"""
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


def filter_ids(view, samples, sample_ids, label_ids, patches_field=None):
    # No filtering required
    if view == samples or view.view() == samples.view():
        return view, sample_ids, label_ids, None

    if patches_field is None:
        _sample_ids = view.values("id")
        keep_inds = _get_keep_inds(_sample_ids, sample_ids)
        return view, np.array(_sample_ids), None, keep_inds

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
    _sample_ids = [l["sample_id"] for l in labels]
    _label_ids = [l["label_id"] for l in labels]
    keep_inds = _get_keep_inds(_label_ids, label_ids)
    return view, np.array(_sample_ids), np.array(_label_ids), keep_inds


def _get_keep_inds(ids, ref_ids):
    if len(ids) == len(ref_ids) and list(ids) == list(ref_ids):
        return None

    inds_map = {_id: idx for idx, _id in enumerate(ref_ids)}

    keep_inds = []
    bad_ids = []
    for _id in ids:
        ind = inds_map.get(_id, None)
        if ind is not None:
            keep_inds.append(ind)
        else:
            bad_ids.append(_id)

    num_bad = len(bad_ids)

    if num_bad == 1:
        raise ValueError(
            "The provided view contains ID '%s' not present in the index"
            % bad_ids[0]
        )

    if num_bad > 1:
        raise ValueError(
            "The provided view contains %d IDs (eg '%s') not present in the "
            "index" % (num_bad, bad_ids[0])
        )

    return np.array(keep_inds, dtype=np.int64)


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
