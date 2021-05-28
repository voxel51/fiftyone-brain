"""
Similarity methods.

| Copyright 2017-2021, Voxel51, Inc.
| `voxel51.com <https://voxel51.com/>`_
|
"""
from bson import ObjectId

import numpy as np
import sklearn.metrics as skm

import eta.core.utils as etau

from fiftyone import ViewField as F
import fiftyone.core.brain as fob
import fiftyone.core.stages as fos
import fiftyone.core.patches as fop
import fiftyone.core.validation as fov
import fiftyone.zoo as foz

from fiftyone.brain.similarity import SimilarityConfig, SimilarityResults

import logging


logger = logging.getLogger(__name__)

_AGGREGATIONS = {"mean": np.mean, "min": np.min, "max": np.max}

_DEFAULT_MODEL = "mobilenet-v2-imagenet-torch"
_DEFAULT_BATCH_SIZE = None


def compute_similarity(
    samples,
    patches_field,
    embeddings,
    brain_key,
    model,
    batch_size,
    force_square,
    alpha,
):
    """See ``fiftyone/brain/__init__.py``."""

    fov.validate_collection(samples)

    if model is None and embeddings is None:
        model = foz.load_zoo_model(_DEFAULT_MODEL)
        if batch_size is None:
            batch_size = _DEFAULT_BATCH_SIZE

    if etau.is_str(embeddings):
        embeddings_field = embeddings
        embeddings = None
    else:
        embeddings_field = None

    config = SimilarityConfig(
        embeddings_field=embeddings_field,
        model=model,
        patches_field=patches_field,
    )
    brain_method = config.build()
    if brain_key is not None:
        brain_method.register_run(samples, brain_key)

    if model is not None:
        if etau.is_str(model):
            model = foz.load_zoo_model(model)

        if patches_field is not None:
            logger.info("Computing patch embeddings...")
            embeddings = samples.compute_patch_embeddings(
                model,
                patches_field,
                embeddings_field=embeddings_field,
                batch_size=batch_size,
                force_square=force_square,
                alpha=alpha,
            )
        else:
            logger.info("Computing embeddings...")
            embeddings = samples.compute_embeddings(
                model,
                embeddings_field=embeddings_field,
                batch_size=batch_size,
            )

    if embeddings_field is not None:
        embeddings = samples.values(embeddings_field)
        embeddings = [e for e in embeddings if e is not None and e.size > 0]
        if patches_field is not None:
            embeddings = np.concatenate(embeddings, axis=0)
        else:
            embeddings = np.stack(embeddings)

    if isinstance(embeddings, dict):
        embeddings = [embeddings[_id] for _id in samples.values("id")]
        embeddings = [e for e in embeddings if e is not None and e.size > 0]
        embeddings = np.concatenate(embeddings, axis=0)

    results = SimilarityResults(samples, embeddings, config)
    brain_method.save_run_results(samples, brain_key, results)

    return results


def sort_by_similarity(
    samples,
    embeddings,
    query_ids,
    sample_ids,
    label_ids=None,
    patches_field=None,
    filter_ids=False,
    k=None,
    reverse=False,
    metric="euclidean",
    aggregation="mean",
    mongo=False,
):
    if etau.is_str(query_ids):
        query_ids = [query_ids]

    if not query_ids:
        raise ValueError("At least one query ID must be provided")

    if aggregation not in _AGGREGATIONS:
        raise ValueError(
            "Unsupported aggregation method '%s'. Supported values are %s"
            % (aggregation, tuple(_AGGREGATIONS.keys()))
        )

    #
    # Extract query embeddings
    #

    if patches_field is None:
        ids = sample_ids
    else:
        ids = label_ids

    bad_ids = []
    query_inds = []
    for query_id in query_ids:
        _inds = np.where(ids == query_id)[0]
        if _inds.size == 0:
            bad_ids.append(query_id)
        else:
            query_inds.append(_inds[0])

    if bad_ids:
        raise ValueError(
            "Query IDs %s were not included in this index" % bad_ids
        )

    query_embeddings = embeddings[query_inds]

    #
    # Filter possible results, if necessary
    #

    if filter_ids:
        if patches_field is None:
            possible_ids = set(samples.values("id"))
            keep = np.array([_id in possible_ids for _id in sample_ids])
            sample_ids = sample_ids[keep]
            embeddings = embeddings[keep]
            ids = sample_ids
        elif isinstance(samples, (fop.PatchesView, fop.EvaluationPatchesView)):
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
            ) and patches_field not in (samples.gt_field, samples.pred_field):
                raise ValueError(
                    "This evaluation patches view contains patches from "
                    "fields '%s' and '%s', not '%s'"
                    % (samples.gt_field, samples.pred_field, patches_field)
                )

            ids_map = samples._get_ids_map(patches_field)
            keep = np.array([label_id in ids_map for label_id in label_ids])

            label_ids = label_ids[keep]
            embeddings = embeddings[keep]
            sample_ids = np.array([ids_map[_id] for _id in label_ids])
            ids = label_ids
        else:
            possible_ids = set(
                l["label_id"]
                for l in samples._get_selected_labels(fields=patches_field)
            )
            keep = np.array([_id in possible_ids for _id in label_ids])

            sample_ids = sample_ids[keep]
            label_ids = label_ids[keep]
            embeddings = embeddings[keep]
            ids = label_ids

    #
    # Perform sorting
    #

    dists = skm.pairwise_distances(embeddings, query_embeddings, metric=metric)

    agg_fcn = _AGGREGATIONS[aggregation]
    dists = agg_fcn(dists, axis=1)

    inds = np.argsort(dists)
    if reverse:
        inds = np.flip(inds)

    if k is not None:
        inds = inds[:k]

    result_ids = list(ids[inds])

    #
    # Construct sorted view
    #

    view = samples
    pipeline = []

    if patches_field is None:
        stage = fos.Select(result_ids, ordered=True)
        if mongo:
            pipeline.extend(stage.to_mongo(view))

        view = view.add_stage(stage)
    else:
        result_sample_ids = _unique_no_sort(sample_ids[inds])
        stage = fos.Select(result_sample_ids, ordered=True)
        if mongo:
            pipeline.extend(stage.to_mongo(view))

        view = view.add_stage(stage)

        if k is not None:
            _ids = [ObjectId(_id) for _id in result_ids]
            stage = fos.FilterLabels(patches_field, F("_id").is_in(_ids))
            if mongo:
                pipeline.extend(stage.to_mongo(view))

            view = view.add_stage(stage)

    if mongo:
        return pipeline

    return view


def _unique_no_sort(values):
    seen = set()
    return [v for v in values if v not in seen and not seen.add(v)]


class Similarity(fob.BrainMethod):
    """Similarity method.

    Args:
        config: a :class:`fiftyone.brain.similarity.SimilarityConfig`
    """

    def get_fields(self, samples, brain_key):
        fields = []
        if self.config.patches_field is not None:
            fields.append(self.config.patches_field)

        return fields

    def cleanup(self, samples, brain_key):
        pass
