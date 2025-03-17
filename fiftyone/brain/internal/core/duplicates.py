"""
Duplicates methods.

| Copyright 2017-2025, Voxel51, Inc.
| `voxel51.com <https://voxel51.com/>`_
|
"""
from collections import defaultdict
import itertools
import logging
import multiprocessing

import eta.core.utils as etau

import fiftyone.core.media as fom
import fiftyone.core.utils as fou
import fiftyone.core.validation as fov

import fiftyone.brain as fb
import fiftyone.brain.similarity as fbs
import fiftyone.brain.internal.core.utils as fbu


logger = logging.getLogger(__name__)

_DEFAULT_MODEL = "resnet18-imagenet-torch"


def compute_near_duplicates(
    samples,
    threshold=None,
    roi_field=None,
    embeddings=None,
    similarity_index=None,
    model=None,
    model_kwargs=None,
    force_square=False,
    alpha=None,
    batch_size=None,
    num_workers=None,
    skip_failures=True,
    progress=None,
):
    """See ``fiftyone/brain/__init__.py``."""

    fov.validate_collection(samples)

    if etau.is_str(embeddings):
        embeddings_field, embeddings_exist = fbu.parse_data_field(
            samples,
            embeddings,
            data_type="embeddings",
        )
        embeddings = None
    else:
        embeddings_field = None
        embeddings_exist = None

    if etau.is_str(similarity_index):
        similarity_index = samples.load_brain_results(similarity_index)

    if (
        model is None
        and embeddings is None
        and similarity_index is None
        and not embeddings_exist
    ):
        model = _DEFAULT_MODEL

    if similarity_index is None:
        similarity_index = fb.compute_similarity(
            samples,
            backend="sklearn",
            roi_field=roi_field,
            embeddings=embeddings_field or embeddings,
            model=model,
            model_kwargs=model_kwargs,
            force_square=force_square,
            alpha=alpha,
            batch_size=batch_size,
            num_workers=num_workers,
            skip_failures=skip_failures,
            progress=progress,
        )
    elif not isinstance(similarity_index, fbs.DuplicatesMixin):
        raise ValueError(
            "This method only supports similarity indexes that implement the "
            "%s mixin" % fbs.DuplicatesMixin
        )

    similarity_index.find_duplicates(thresh=threshold)

    return similarity_index


def compute_exact_duplicates(samples, num_workers, skip_failures, progress):
    """See ``fiftyone/brain/__init__.py``."""

    fov.validate_collection(samples)

    if num_workers is None:
        if samples.media_type == fom.VIDEO:
            num_workers = multiprocessing.cpu_count()
        else:
            num_workers = 1

    logger.info("Computing filehashes...")

    method = "md5" if samples.media_type == fom.VIDEO else None

    if num_workers <= 1:
        hashes = _compute_filehashes(samples, method, progress)
    else:
        hashes = _compute_filehashes_multi(
            samples, method, num_workers, progress
        )

    num_missing = sum(h is None for h in hashes)
    if num_missing > 0:
        msg = "Failed to compute %d filehashes" % num_missing
        if skip_failures:
            logger.warning(msg)
        else:
            raise ValueError(msg)

    neighbors_map = defaultdict(list)

    observed_hashes = {}
    for _id, _hash in hashes.items():
        if _hash is None:
            continue

        if _hash in observed_hashes:
            neighbors_map[observed_hashes[_hash]].append(_id)
        else:
            observed_hashes[_hash] = _id

    return dict(neighbors_map)


def _compute_filehashes(samples, method, progress):
    ids, filepaths = samples.values(["id", "filepath"])

    with fou.ProgressBar(total=len(ids), progress=progress) as pb:
        return {
            _id: _compute_filehash(filepath, method)
            for _id, filepath in pb(zip(ids, filepaths))
        }


def _compute_filehashes_multi(samples, method, num_workers, progress):
    ids, filepaths = samples.values(["id", "filepath"])

    methods = itertools.repeat(method)

    inputs = list(zip(ids, filepaths, methods))

    with fou.ProgressBar(total=len(inputs), progress=progress) as pb:
        with multiprocessing.Pool(processes=num_workers) as pool:
            return {
                k: v
                for k, v in pb(
                    pool.imap_unordered(_do_compute_filehash, inputs)
                )
            }


def _compute_filehash(filepath, method):
    try:
        filehash = fou.compute_filehash(filepath, method=method)
    except:
        filehash = None

    return filehash


def _do_compute_filehash(args):
    _id, filepath, method = args
    try:
        filehash = fou.compute_filehash(filepath, method=method)
    except:
        filehash = None

    return _id, filehash
