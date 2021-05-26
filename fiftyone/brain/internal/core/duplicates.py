"""
Methods that compute insights related to sample uniqueness.

| Copyright 2017-2021, Voxel51, Inc.
| `voxel51.com <https://voxel51.com/>`_
|
"""
import logging

import numpy as np
import sklearn.neighbors as skn
import sklearn.preprocessing as skp

import eta.core.utils as etau

import fiftyone.core.brain as fob
import fiftyone.core.labels as fol
import fiftyone.core.media as fom
import fiftyone.core.validation as fov
import fiftyone.zoo as foz

import fiftyone.brain.internal.models as fbm
from fiftyone.brain.duplicates import DuplicatesConfig, DuplicatesResults


logger = logging.getLogger(__name__)


_ALLOWED_PATCH_FIELD_TYPES = (
    fol.Detection,
    fol.Detections,
    fol.Polyline,
    fol.Polylines,
)
_DEFAULT_MODEL = "simple-resnet-cifar10"
_DEFAULT_BATCH_SIZE = 16


def compute_duplicates(
    samples,
    patches_field,
    embeddings,
    brain_key,
    model,
    metric,
    thresh,
    fraction,
    batch_size,
    force_square,
    alpha,
):
    """See ``fiftyone/brain/__init__.py``."""

    fov.validate_collection(samples)

    if patches_field is not None:
        fov.validate_collection_label_fields(
            samples, patches_field, _ALLOWED_PATCH_FIELD_TYPES
        )

    if samples.media_type == fom.VIDEO:
        raise ValueError("Duplicates does not yet support video collections")

    if etau.is_str(embeddings):
        embeddings_field = embeddings
        embeddings = None
    else:
        embeddings_field = None

    config = DuplicatesConfig(
        metric=metric,
        thresh=thresh,
        fraction=fraction,
        embeddings_field=embeddings_field,
        model=model,
        patches_field=patches_field,
    )
    brain_method = config.build()
    brain_method.register_run(samples, brain_key)

    if thresh is None and fraction is None:
        if metric == "cosine":
            default_degrees = 5  # @todo tune this?
            thresh = np.sqrt(
                2.0 - 2.0 * np.cos(default_degrees * np.pi / 180.0)
            )
        else:
            raise ValueError(
                "You must provide either `thresh` or `fraction` when "
                "`metric` != 'cosine'"
            )

    #
    # Get embeddings
    #

    if model is not None or (embeddings is None and embeddings_field is None):
        if etau.is_str(model):
            model = foz.load_zoo_model(model)
        elif model is None:
            model = fbm.load_model(_DEFAULT_MODEL)
            if batch_size is None:
                batch_size = _DEFAULT_BATCH_SIZE

        logger.info("Generating embeddings...")

        if patches_field is None:
            embeddings = samples.compute_embeddings(
                model, batch_size=batch_size
            )
        else:
            embeddings = samples.compute_patch_embeddings(
                model,
                patches_field,
                handle_missing="skip",
                batch_size=batch_size,
                force_square=force_square,
                alpha=alpha,
            )

    if embeddings_field is not None:
        # extracts a potentially huge number of embedding vectors/arrays
        embeddings = samples.values(embeddings_field)

    if isinstance(embeddings, dict):
        _embeddings = []
        for _id in samples.values("id"):
            e = embeddings.get(_id, None)
            if e is not None:
                _embeddings.append(e)

        embeddings = np.concatenate(_embeddings, axis=0)

    logger.info("Detecting duplicates...")

    num_embeddings = len(embeddings)
    neighbors = init_neighbors(embeddings, metric)

    if fraction is not None:
        keep, thresh = _remove_duplicates_fraction(
            neighbors, fraction, num_embeddings, init_thresh=thresh
        )
    else:
        keep = _remove_duplicates_thresh(neighbors, thresh, num_embeddings)

    if patches_field is not None:
        id_path = samples._get_label_field_path(patches_field, "id")
        ids = samples.values(id_path, unwind=True)
    else:
        ids = samples.values("id")

    keep_ids = np.array([_id for idx, _id in enumerate(ids) if idx in keep])

    logger.info("Duplicates computation complete")

    results = DuplicatesResults(
        samples, embeddings, keep_ids, thresh, config, neighbors=neighbors,
    )
    brain_method.save_run_results(samples, brain_key, results)

    return results


def init_neighbors(embeddings, metric):
    if metric == "cosine":
        embeddings = skp.normalize(embeddings, axis=1)
        metric = "euclidean"

    neighbors = skn.NearestNeighbors(
        n_neighbors=None, radius=None, algorithm="auto", metric=metric
    )
    neighbors.fit(embeddings)
    return neighbors


def _remove_duplicates_thresh(neighbors, thresh, num_embeddings):
    inds = neighbors.radius_neighbors(radius=thresh, return_distance=False)

    keep = set(range(num_embeddings))
    for ind in range(num_embeddings):
        if ind in keep:
            keep -= {i for i in inds[ind] if i > ind}

    return keep


def _remove_duplicates_fraction(
    neighbors, fraction, num_embeddings, init_thresh=None
):
    if init_thresh is not None:
        thresh = init_thresh
    else:
        thresh = 1

    thresh_lims = [0, None]
    num_target = int(round((1.0 - fraction) * num_embeddings))

    while True:
        keep = _remove_duplicates_thresh(neighbors, thresh, num_embeddings)
        num_keep = len(keep)

        logger.info(
            "threshold: %f, kept: %d, target: %d",
            thresh,
            num_keep,
            num_target,
        )

        if num_keep == num_target:
            break

        if num_keep < num_target:
            # Need to decrease threshold
            thresh_lims[1] = thresh
            thresh = 0.5 * (thresh_lims[0] + thresh)
        else:
            # Need to increase threshold
            thresh_lims[0] = thresh
            if thresh_lims[1] is not None:
                thresh = 0.5 * (thresh + thresh_lims[1])
            else:
                thresh *= 2

    return keep, thresh


class Duplicates(fob.BrainMethod):
    """Duplicates method.

    Args:
        config: a :class:`fiftyone.brain.duplicates.DuplicatesConfig`
    """

    def get_fields(self, samples, brain_key):
        fields = []
        if self.config.patches_field is not None:
            fields.append(self.config.patches_field)

        return fields

    def cleanup(self, samples, brain_key):
        pass
