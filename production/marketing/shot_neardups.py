"""
Script to generate a screen-shot of near-duplicates using the uniqueness and
ranking.

run: python shot_neardups.py

Outputs are placed in the `outputs/` directory.

Copyright 2017-2020, Voxel51, Inc.
voxel51.com
"""
# pragma pylint: disable=redefined-builtin
# pragma pylint: disable=unused-wildcard-import
# pragma pylint: disable=wildcard-import
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from builtins import *

# pragma pylint: enable=redefined-builtin
# pragma pylint: enable=unused-wildcard-import
# pragma pylint: enable=wildcard-import

import eta.core.utils as etau


__output_root__ = "outputs"

etau.ensure_dir(__output_root__)
