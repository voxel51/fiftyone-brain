"""
The brains behind FiftyOne: a powerful package for dataset curation, analysis,
and visualization.

See https://github.com/voxel51/fiftyone for more information.

| Copyright 2017-2020, Voxel51, Inc.
| `voxel51.com <https://voxel51.com/>`_
|
"""
import os

import eta

from .hardness import compute_hardness
from .mistakenness import compute_mistakenness
from .uniqueness import compute_uniqueness


__models_cache__ = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "internal", "models", "cache"
)

eta.config.models_dirs.insert(0, __models_cache__)
