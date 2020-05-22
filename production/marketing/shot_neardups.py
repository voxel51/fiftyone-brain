"""
Script to generate a screen-shot of near-duplicates using the uniqueness and
ranking.

You need to run this inside of an interactive shell like ipython because the
system will launch the gui and we need to have that process independent of the
python code.
```
ipython [1]: run shot_neardups.py
```

Outputs are placed in the `outputs/` directory.

WARNING: Current implementation executes a drop dataset on start.

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

import os
import time

import eta.core.utils as etau

import fiftyone as fo
import fiftyone.zoo as foz
import fiftyone.brain as fob
import fiftyone.core.odm as foo

foo.drop_database()

localtime = lambda: time.strftime("%Y%m%d-%H%M%S", time.localtime())

__output_root__ = "outputs"

etau.ensure_dir(__output_root__)

print("Working on near-duplicates in CIFAR-10 Test")

dataset = foz.load_zoo_dataset("cifar10", split="test")

fob.compute_uniqueness(dataset)

view = dataset.view().sort_by("uniqueness")

print("Launching dashboard...")

session = fo.launch_dashboard(view=view)

input("Press enter when you are ready for the screen shot.")

snap_name = "neardups_cifar10_test" + localtime() + ".png"
etau.save_window_snapshot(
    window_name="FiftyOne", file_path=os.path.join(__output_root__, snap_name)
)
