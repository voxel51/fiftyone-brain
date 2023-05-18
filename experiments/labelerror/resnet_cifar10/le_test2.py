"""
le_test2 --> initial tests for using fiftyone with this code
Loads the data, assigns ground-truth labels, then trains a clean model

| Copyright 2017-2023, Voxel51, Inc.
| `voxel51.com <https://voxel51.com/>`_
|
"""
from functools import partial
import json
import os
import time

from imageio import imsave

import fiftyone as fo

from config import *
from datasets import *
from simple_resnet import *
from utils import Timer


TEMP_TRAIN_DIR = "/tmp/le_test/train"
TEMP_VALID_DIR = "/tmp/le_test/valid"

localtime = lambda: time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())


def write_images(root, images, overwrite=False):
    paths = []
    for i, s in enumerate(images):
        path = os.path.join(root, f"{i:05d}.jpg")
        paths.append(path)

        if overwrite or not os.path.exists(path):
            img = s.copy()
            img = transpose(img, source="CHW", target="HWC")
            imsave(path, img)

    return paths


def main(config):
    #
    # Load dataset
    #
    # `train_set` and `valid_set` that are lists of `(image, label)` tuples
    #

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

    # Write raw dataset to disk
    os.makedirs(TEMP_TRAIN_DIR, exist_ok=True)
    os.makedirs(TEMP_VALID_DIR, exist_ok=True)
    train_image_paths = write_images(
        TEMP_TRAIN_DIR, list(zip(*whole_train_set))[0]
    )
    valid_image_paths = write_images(TEMP_VALID_DIR, list(zip(*valid_set))[0])

    #
    # Load data into FiftyOne
    #

    timer = Timer()
    dataset = fo.Dataset("le_cifar10")

    # Train split
    train_samples = []
    for i, s in enumerate(whole_train_set):
        train_samples.append(
            fo.Sample(
                train_image_paths[i],
                tags=["train"],
                ground_truth=fo.Classification(label=cifar10_classes[s[1]]),
            )
        )

    train_ids = dataset.add_samples(train_samples)

    # Valid split
    valid_samples = []
    for i, s in enumerate(valid_set):
        valid_samples.append(
            fo.Sample(
                valid_image_paths[i],
                tags=["valid"],
                ground_truth=fo.Classification(label=cifar10_classes[s[1]]),
            )
        )

    valid_ids = dataset.add_samples(valid_samples)

    print(f"Finished getting data into fiftyone in {timer():.2f} seconds")
    print(dataset.summary())

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

    corrupt_N = round(config.p_corrupt * total_N)

    print(f"Setting up the experiment: {total_N} training samples.")
    print(f"- starting with {start_N}")
    print(f"- incrementing by {incr_N} for each of {config.n_rounds-1} rounds")
    print(f"- total rounds: {config.n_rounds}")

    print(f"Starting the model training at {localtime()}")

    inuse_N = start_N

    model = Network(simple_resnet()).to(device).half()
    logs, state = Table(), {MODEL: model, LOSS: x_ent_loss}

    valid_batches = DataLoader(
        valid_set, config.batch_size, shuffle=False, drop_last=False
    )

    # initially randomly shuffle the dataset and take the initial number of
    # samples
    whole_train_set_use = whole_train_set[0:inuse_N]
    whole_train_set_avail = whole_train_set[inuse_N:]
    print(
        f"Split training set into two; using {len(whole_train_set_use)}, "
        + f"available {len(whole_train_set_avail)}"
    )

    sm = torch.nn.Softmax(dim=1)

    stats = {}

    for iteration in range(config.n_rounds):
        print(f"beginning next round of training, using {inuse_N} samples")

        if config.cold_start:
            model = Network(simple_resnet()).to(device).half()
            logs, state = Table(), {MODEL: model, LOSS: x_ent_loss}

        train_batches = DataLoader(
            Transform(whole_train_set_use, train_transforms),
            config.batch_size,
            shuffle=True,
            set_random_choices=True,
            drop_last=True,
        )
        lr = (
            lambda step: lr_schedule(step / len(train_batches))
            / config.batch_size
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
        logs.df().query(f"epoch=={config.epochs}")[
            ["train_acc", "valid_acc"]
        ].describe()

        model.train(False)  # == model.eval()

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
                inputs = dict(input=images.cuda().half())
                outputs = model(inputs)
                y = outputs["logits"]
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
