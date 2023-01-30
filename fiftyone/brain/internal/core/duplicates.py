"""
Duplicates methods.

| Copyright 2017-2023, Voxel51, Inc.
| `voxel51.com <https://voxel51.com/>`_
|
"""
from collections import defaultdict
import itertools
import logging
import multiprocessing

import fiftyone.core.media as fom
import fiftyone.core.utils as fou
import fiftyone.core.validation as fov


logger = logging.getLogger(__name__)


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

    if num_workers <= 1:
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


def _compute_filehashes(samples, method):
    ids, filepaths = samples.values(["id", "filepath"])

    with fou.ProgressBar(total=len(ids)) as pb:
        return {
            _id: _compute_filehash(filepath, method)
            for _id, filepath in pb(zip(ids, filepaths))
        }


def _compute_filehashes_multi(samples, method, num_workers):
    ids, filepaths = samples.values(["id", "filepath"])

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
