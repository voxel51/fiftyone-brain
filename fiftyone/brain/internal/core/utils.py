"""
Utilities.

| Copyright 2017-2021, Voxel51, Inc.
| `voxel51.com <https://voxel51.com/>`_
|
"""
import numpy as np

import fiftyone.core.patches as fop


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
    if view is None or view == samples:
        return samples, sample_ids, label_ids, None

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

    return np.array(keep_inds)
