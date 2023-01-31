"""
Script to generate a screenshot of near-duplicates using the uniqueness and
ranking.

You need to run this inside of an interactive shell like IPython because the
system will launch the dashboard and we need to have that process independent
of the python code.

Outputs are placed in the `outputs/` directory.

Usage::

    # From inside IPython
    run shot_neardups.py

Copyright 2017-2023, Voxel51, Inc.
voxel51.com
"""
import os
import time

import eta.core.utils as etau

import fiftyone as fo
import fiftyone.zoo as foz

import fiftyone.brain as fob


OUTPUT_DIR = "outputs"


print("Working on near-duplicates in CIFAR-10 Test")

dataset = foz.load_zoo_dataset("cifar10", split="test")

fob.compute_uniqueness(dataset)

view = dataset.view().sort_by("uniqueness")

print("Launching dashboard...")
session = fo.launch_dashboard(view=view)

input("Press enter when you are ready for the screenshot:")

localtime = lambda: time.strftime("%Y%m%d-%H%M%S", time.localtime())
outname = "neardups_cifar10_test" + localtime() + ".png"
outpath = os.path.join(OUTPUT_DIR, outname)

etau.save_window_snapshot("FiftyOne", outpath)
