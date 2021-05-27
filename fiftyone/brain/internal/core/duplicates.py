"""
Methods that compute insights related to sample uniqueness.

| Copyright 2017-2021, Voxel51, Inc.
| `voxel51.com <https://voxel51.com/>`_
|
"""
import itertools
import logging
import multiprocessing

import numpy as np
import sklearn.metrics as skm
import sklearn.neighbors as skn
import sklearn.preprocessing as skp

import eta.core.utils as etau

import fiftyone.core.aggregations as foa
import fiftyone.core.brain as fob
import fiftyone.core.labels as fol
import fiftyone.core.media as fom
import fiftyone.core.utils as fou
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

_MAX_PRECOMPUTE_DISTS = 15000  # ~1.7GB to store distance matrix in-memory


def compute_exact_duplicates(samples, num_workers, skip_failures):
    """See ``fiftyone/brain/__init__.py``."""

    fov.validate_collection(samples)

    if num_workers is None:
        if samples.media_type == fom.VIDEO:
            num_workers = multiprocessing.cpu_count()
        else:
            num_workers = 1

    logger.info("Computing filehashes...")

    method = "md5" if samples.media_type == fom.VIDEO else None

    if num_workers == 1:
        hashes = _compute_filehashes(samples, method)
    else:
        hashes = _compute_filehashes_multi(samples, method, num_workers)

    num_missing = sum(h is None for h in hashes)
    if num_missing > 0:
        msg = "Failed to compute %d filehashes" % num_missing
        if skip_failures:
            logger.warning(msg)
        else:
            raise ValueError(msg)

    dup_ids = []
    observed_hashes = set()
    for _id, _hash in hashes.items():
        if _hash is None:
            continue

        if _hash in observed_hashes:
            dup_ids.append(_id)
        else:
            observed_hashes.add(_hash)

    return dup_ids


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
            thresh = 1.0 - np.cos(default_degrees * np.pi / 180.0)
        else:
            raise ValueError(
                "You must provide either `thresh` or `fraction` when "
                "`metric` != 'cosine'"
            )

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
    neighbors, cosine_hack = init_neighbors(embeddings, metric)

    if fraction is not None:
        keep, thresh = _remove_duplicates_fraction(
            neighbors,
            fraction,
            num_embeddings,
            init_thresh=thresh,
            cosine_hack=cosine_hack,
        )
    else:
        keep = _remove_duplicates_thresh(
            neighbors, thresh, num_embeddings, cosine_hack=cosine_hack
        )

    if patches_field is not None:
        id_path = samples._get_label_field_path(patches_field, "id")
        ids = samples.values(id_path, unwind=True)
    else:
        ids = samples.values("id")

    dup_ids = np.array([_id for idx, _id in enumerate(ids) if idx not in keep])
    keep_ids = np.array([_id for idx, _id in enumerate(ids) if idx in keep])

    logger.info("Duplicates computation complete")

    results = DuplicatesResults(
        samples,
        embeddings,
        config,
        dup_ids=dup_ids,
        keep_ids=keep_ids,
        thresh=thresh,
        neighbors=neighbors,
    )
    brain_method.save_run_results(samples, brain_key, results)

    return results


def init_neighbors(embeddings, metric):
    # Center embeddings
    embeddings = np.asarray(embeddings)
    embeddings -= embeddings.mean(axis=0, keepdims=True)

    # For small datasets, compute entire distance matrix
    num_embeddings = len(embeddings)
    if num_embeddings <= _MAX_PRECOMPUTE_DISTS:
        embeddings = skm.pairwise_distances(embeddings, metric=metric)
        metric = "precomputed"
    else:
        logger.info(
            "Computing neighbors for %d embeddings; this may take awhile...",
            num_embeddings,
        )

    #
    # For large datasets, use ``NearestNeighbors``
    #
    # @todo upper-bound number of allowed dimensions here? For many samples
    # with many dimensions, this will be impractically slow...
    #

    # Nearest neighbors does not directly support cosine distance, so we
    # approximate via euclidean distance on unit-norm embeddings
    if metric == "cosine":
        cosine_hack = True
        embeddings = skp.normalize(embeddings, axis=1)
        metric = "euclidean"
    else:
        cosine_hack = False

    neighbors = skn.NearestNeighbors(algorithm="auto", metric=metric)
    neighbors.fit(embeddings)

    return neighbors, cosine_hack


def _remove_duplicates_thresh(
    neighbors, thresh, num_embeddings, cosine_hack=False
):
    # When not using brute force, we approximate cosine distance by computing
    # Euclidean distance on unit-norm embeddings. ED = sqrt(2 * CD), so we need
    # to scale the threshold appropriately
    if cosine_hack:
        thresh = np.sqrt(2.0 * thresh)

    inds = neighbors.radius_neighbors(radius=thresh, return_distance=False)

    keep = set(range(num_embeddings))
    for ind in range(num_embeddings):
        if ind in keep:
            keep -= {i for i in inds[ind] if i > ind}

    return keep


def _remove_duplicates_fraction(
    neighbors, fraction, num_embeddings, init_thresh=None, cosine_hack=False
):
    if init_thresh is not None:
        thresh = init_thresh
    else:
        thresh = 1

    thresh_lims = [0, None]
    num_target = int(round((1.0 - fraction) * num_embeddings))
    num_keep = -1

    while True:
        keep = _remove_duplicates_thresh(
            neighbors, thresh, num_embeddings, cosine_hack=cosine_hack
        )
        num_keep_last = num_keep
        num_keep = len(keep)

        logger.info(
            "threshold: %f, kept: %d, target: %d",
            thresh,
            num_keep,
            num_target,
        )

        if num_keep == num_target or (
            num_keep == num_keep_last
            and thresh_lims[1] is not None
            and thresh_lims[1] - thresh_lims[0] < 1e-6
        ):
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


def _compute_filehashes(samples, method):
    # ids, filepaths = samples.values(["id", "filepath"])
    ids, filepaths = samples.aggregate(
        [foa.Values("id"), foa.Values("filepath")]
    )

    with fou.ProgressBar(total=len(ids)) as pb:
        return {
            _id: _compute_filehash(filepath, method)
            for _id, filepath in pb(zip(ids, filepaths))
        }


def _compute_filehashes_multi(samples, method, num_workers):
    # ids, filepaths = samples.values(["id", "filepath"])
    ids, filepaths = samples.aggregate(
        [foa.Values("id"), foa.Values("filepath")]
    )

    methods = itertools.repeat(method)

    inputs = list(zip(ids, filepaths, methods))

    with fou.ProgressBar(total=len(inputs)) as pb:
        with multiprocessing.Pool(processes=num_workers) as pool:
            return {
                k: v
                for k, v in pb(
                    pool.imap_unordered(_do_compute_filehash, inputs)
                )
            }


def _compute_filehash(filepath, method):
    try:
        filehash = fou.compute_filehash(filepath)
        # filehash = fou.compute_filehash(filepath, method=method)
    except:
        filehash = None

    return filehash


def _do_compute_filehash(args):
    _id, filepath, method = args
    try:
        filehash = fou.compute_filehash(filepath)
        # filehash = fou.compute_filehash(filepath, method=method)
    except:
        filehash = None

    return _id, filehash


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
