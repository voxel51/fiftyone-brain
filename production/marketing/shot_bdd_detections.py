"""
Script to generate a screenshot of some object detections from the BDD dataset.

You need to run this inside of an interactive shell like IPython because the
system will launch the dashboard and we need to have that process independent
of the python code.

Outputs are placed in the `outputs/` directory.

Usage::

    # From inside IPython
    run shot_bdd_detections.py

Copyright 2017-2023, Voxel51, Inc.
voxel51.com
"""
import os
import time

import eta.core.utils as etau
import eta.core.storage as etas

import fiftyone as fo

import fiftyone.brain as fob


FILE_ID = "1uioZHQ8VVUYb2OIrdD5M5gSw4XmGIOi_"

DATASET_DIR = "data/bdd_subset"
OUTPUT_DIR = "outputs"


print("Working on BDD object detections")

# Download dataset if necessary
if not os.path.isdir(DATASET_DIR):
    print("Downloading dataset to '%s'" % DATASET_DIR)
    client = etas.GoogleDriveStorageClient()
    tmp_path = client.get_file_metadata(FILE_ID)["name"]
    client.download(FILE_ID, tmp_path)
    etau.extract_archive(tmp_path, outdir="data", delete_archive=True)

dataset = fo.Dataset.from_image_detection_dataset(DATASET_DIR, "BDD")

fob.compute_uniqueness(dataset)

view = dataset.view().sort_by("uniqueness", reverse=True)

print("Launching dashboard...")
session = fo.launch_dashboard(view=view)

input("Press enter when you are ready for the screenshot:")

localtime = lambda: time.strftime("%Y%m%d-%H%M%S", time.localtime())
outname = "bdd_detections_" + localtime() + ".png"
outpath = os.path.join(OUTPUT_DIR, outname)

etau.save_window_snapshot("FiftyOne", outpath)
