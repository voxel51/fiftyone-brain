"""
Internal FiftyOne Brain package.

Contains all non-public code powering the ``fiftyone.brain`` public namespace.

| Copyright 2017-2020, Voxel51, Inc.
| `voxel51.com <https://voxel51.com/>`_
|
"""
import os

import eta


__models_cache__ = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "internal", "models", "cache"
)

eta.config.models_dirs.insert(0, __models_cache__)
