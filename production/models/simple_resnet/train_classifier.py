"""
Trains a clean model using the brain.model simple_resnet class; it stores the
weights in a file for publishing the model.

Supports these config parameters:

    - config.take
    - config.epochs
    - config.batch_size
    - config.n_max
    - config.p_initial
    - config.n_rounds
    - config.cold_start
    - config.stats_path
    - config.model_path

A simple (minimal) set of these to run on a small machine is::

    run train_classifier.py -t 2000 -e 12 -b 64 --n_rounds 1 --p_initial 1.0 -m /tmp/foo.pth

| Copyright 2017-2023, Voxel51, Inc.
| `voxel51.com <https://voxel51.com/>`_
|
"""
from functools import partial
import json
import os
import random
import sys
import time

from fiftyone.brain.internal.models.simple_resnet import *

from config import *
from datasets import *
from preprocess import *
from training import *
from utils import Timer


TEMP_TRAIN_DIR = "/tmp/le_test/train"
TEMP_VALID_DIR = "/tmp/le_test/valid"

localtime = lambda: time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())


def main(config):

    ## Initial Data Input
    # Produces train_set and valid_set that are lists of tuples: (image, label)
    timer = Timer()
    whole_dataset = cifar10(root=DATA_DIR)
    print("Preprocessing training data")
    transforms = [
        partial(
            normalise,
            mean=np.array(cifar10_mean, dtype=np.float32),
            std=np.array(cifar10_std, dtype=np.float32),
        ),
        partial(transpose, source="NHWC", target="NCHW"),
    ]
    whole_train_set = list(
        zip(
            *preprocess(
                whole_dataset["train"], [partial(pad, border=4)] + transforms
            ).values()
        )
    )
    valid_set = list(
        zip(*preprocess(whole_dataset["valid"], transforms).values())
    )
    print(f"Finished loading and preprocessing in {timer():.2f} seconds")

    print(f"train set: {len(whole_train_set)} samples")
    print(f"valid set: {len(valid_set)} samples")

    if config.take:
        whole_train_set = whole_train_set[: config.take]
        valid_set = whole_train_set[: config.take]
        print(f"using a subset of the data")
        print(f"train set: {len(whole_train_set)} samples")
        print(f"valid set: {len(valid_set)} samples")

    # function of dataset
    N_labels = 10

    # set up the variables for training the model in each increment of the
    # dataset size
    lr_schedule = PiecewiseLinear([0, 5, config.epochs], [0, 0.4, 0])
    train_transforms = [Crop(32, 32), FlipLR(), Cutout(8, 8)]

    # compute the derived parameters for the trial based on the dataset and the
    # provided config.
    total_N = len(whole_train_set)

    # should be cleaner
    if config.n_max < 0:
        config.n_max = total_N

    start_N = round(config.p_initial * total_N)

    incr_N = (
        0
        if config.n_rounds == 1
        else round((config.n_max - start_N) / (config.n_rounds - 1))
    )

    print(f"Setting up the experiment: {total_N} training samples.")
    print(f"- starting with {start_N}")
    print(f"- incrementing by {incr_N} for each of {config.n_rounds-1} rounds")
    print(f"- total rounds: {config.n_rounds}")

    print(f"Starting the model training at {localtime()}")

    inuse_N = start_N

    model = simple_resnet().to(device).half()
    logs, state = Table(), {MODEL: model, LOSS: x_ent_loss}

    valid_batches = DataLoader(
        valid_set, config.batch_size, shuffle=False, drop_last=False
    )

    whole_train_set_use = whole_train_set[0:inuse_N]
    whole_train_set_avail = whole_train_set[inuse_N:]
    print(
        f"Split training set into two; "
        + f"using {len(whole_train_set_use)}, "
        + f"available {len(whole_train_set_avail)}"
    )

    stats = {}

    for iteration in range(config.n_rounds):
        print(f"beginning next round of training, using {inuse_N} samples")

        if config.cold_start:
            model = simple_resnet().to(device).half()
            logs, state = Table(), {MODEL: model, LOSS: x_ent_loss}

        train_batches = DataLoader(
            Transform(whole_train_set_use, train_transforms),
            config.batch_size,
            shuffle=True,
            set_random_choices=True,
            drop_last=True,
        )
        lr = lambda step: (
            lr_schedule(step / len(train_batches)) / config.batch_size
        )
        opts = [
            SGD(
                trainable_params(model).values(),
                {
                    "lr": lr,
                    "weight_decay": Const(5e-4 * config.batch_size),
                    "momentum": Const(0.9),
                },
            )
        ]
        state[OPTS] = opts

        for epoch in range(config.epochs):
            logs.append(
                union(
                    {"epoch": epoch + 1},
                    train_epoch(
                        state,
                        Timer(torch.cuda.synchronize),
                        train_batches,
                        valid_batches,
                    ),
                )
            )
        (
            logs.df()
            .query(f"epoch=={config.epochs}")[["train_acc", "valid_acc"]]
            .describe()
        )

        model.train(False)

        # record scores for this iteration
        iteration_stats = {}
        iteration_stats["in_use"] = inuse_N

        correct = 0
        total = 0
        class_correct = list(0.0 for i in range(10))
        class_total = list(0.0 for i in range(10))
        with torch.no_grad():
            for data in valid_batches.dataloader:
                images, labels = data
                y = model(images.cuda().half())
                _, predicted = torch.max(y, 1)
                total += labels.size(0)
                labels_gpu = labels.cuda().half()
                correct += (predicted == labels_gpu).sum().item()
                c = (predicted == labels_gpu).squeeze()
                for i in range(min(config.batch_size, len(labels))):
                    label = labels[i]
                    class_correct[label] += c[i].item()
                    class_total[label] += 1

        iteration_stats["validation_accuracy"] = correct / total

        model.train(True)

        # extend the corr_train_set_use with that from avail
        whole_train_set_use.extend(whole_train_set_avail[0:incr_N])
        whole_train_set_avail = whole_train_set_avail[incr_N:]
        inuse_N += incr_N
        assert inuse_N == len(whole_train_set_use)

        stats[inuse_N] = iteration_stats

    print(f"finished the full training; stats to follow")
    print(stats)

    if config.stats_path:
        with open(config.stats_path, "w") as fp:
            json.dump(stats, fp)

    if config.model_path:
        torch.save(model.state_dict(), config.model_path)


if __name__ == "__main__":

    config = commandline()

    print(f"running with config: {config}")

    main(config)
