"""
le_test1 --> initial tests for using fiftyone with this code
This test just loads the dataset into fiftyone and exists.
It times the loading, etc...


| Copyright 2017-2020, Voxel51, Inc.
| `voxel51.com <https://voxel51.com/>`_
|
"""
import argparse
from functools import partial
import json
import os
import random
import sys
import time

from scipy.misc import imsave
from scipy.stats import entropy

import fiftyone as fo
from fiftyone.core.odm import drop_database

from config import *
from utils import Timer

TEMP_TRAIN_DIR="/tmp/le_test1/train"
TEMP_VALID_DIR="/tmp/le_test1/valid"

localtime = lambda: time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())

def write_images(root, images, overwrite=False):
    paths = []
    for i, s in enumerate(images):
        path = os.path.join(root, f"{i:05d}.jpg")
        paths.append(path)

        if overwrite or not os.path.exists(path):
            img = s.copy()
            img = transpose(img, source='CHW', target='HWC')
            imsave(path, img)

    return paths

def main():

    ## Initial Data Input
    # Produces train_set and valid_set that are lists of tuples: (image, label)
    timer = Timer()
    whole_dataset = cifar10(root=DATA_DIR)
    print("Preprocessing training data")
    transforms = [
        partial(normalise, mean=np.array(cifar10_mean, dtype=np.float32), std=np.array(cifar10_std, dtype=np.float32)),
        partial(transpose, source='NHWC', target='NCHW'),
    ]
    whole_train_set = list(zip(*preprocess(whole_dataset['train'], [partial(pad, border=4)] + transforms).values()))
    valid_set = list(zip(*preprocess(whole_dataset['valid'], transforms).values()))
    print(f"Finished loading and preprocessing in {timer():.2f} seconds")

    print(f"train set: {len(whole_train_set)} samples")
    print(f"valid set: {len(valid_set)} samples")

    # TEMPORARY
    # Not production usage of fiftyone, but gets the point across to actually
    # get the Data into the system, from the format of this experiment
    #
    # The system does not support a good way to get data already loaded in
    # memory into it.  I think this is a significant limitation for certain
    # types of usage.  Furthermore, there are challenges here in this actual
    # usage of fiftyone because it is a bad idea to need to perform the
    # preprocessing on my data everytime I load an image in from disk, if I can
    # avoid that.
    #
    os.makedirs(TEMP_TRAIN_DIR, exist_ok=True)
    os.makedirs(TEMP_VALID_DIR, exist_ok=True)
    train_image_paths = write_images(TEMP_TRAIN_DIR, list(zip(*whole_train_set))[0])
    valid_image_paths = write_images(TEMP_VALID_DIR, list(zip(*valid_set))[0])

    # make the actual labels for the cifar-10 world
    labels = []
    for i, s in enumerate(cifar10_classes):
        labels.append(fo.ClassificationLabel.create(label=s))

    timer = Timer()
    drop_database()
    dataset = fo.Dataset("le_cifar10")

    samples = []
    for i, s in enumerate(whole_train_set):
        sample = fo.Sample.create(train_image_paths[i], tags=["train"])
        sample.add_label("ground_truth", labels[s[1]])
        samples.append(sample)
    train_ids = dataset.add_samples(samples)

    samples = []
    for i, s in enumerate(valid_set):
        sample = fo.Sample.create(valid_image_paths[i], tags=["valid"])
        sample.add_label("ground_truth", labels[s[1]])
        samples.append(sample)
    valid_ids = dataset.add_samples(samples)

    print(f"Finished getting data into fiftyone in {timer():.2f} seconds")


if __name__ == "__main__":
    main()
