"""
Tests a model by loading the cifar10 dataset and then running predictions
against it.

@todo Update this to use the actual class functionality of the model rather
than the low-level model setup.

Run with at least the following command line:
    -m model_path.pth

Uses
    -t number

| Copyright 2017-2023, Voxel51, Inc.
| `voxel51.com <https://voxel51.com/>`_
|
"""
from functools import partial
import os
import time

from imageio import imsave

import fiftyone as fo

from fiftyone.brain.internal.models.simple_resnet import *

from preprocess import *
from config import *
from datasets import *
from utils import Timer


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

    ## Initial Data Input
    # Produces train_set and valid_set that are lists of tuples: (image, label)
    # (only the valid_set will be used in this script)
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
    valid_set = list(
        zip(*preprocess(whole_dataset["valid"], transforms).values())
    )
    print(f"Finished loading and preprocessing in {timer():.2f} seconds")

    print(f"valid set: {len(valid_set)} samples")

    if config.take:
        valid_set = valid_set[: config.take]
        print(f"using a subset of the data")
        print(f"valid set: {len(valid_set)} samples")

    # Write raw dataset to disk
    os.makedirs(TEMP_VALID_DIR, exist_ok=True)
    valid_image_paths = write_images(TEMP_VALID_DIR, list(zip(*valid_set))[0])

    # make the actual labels for the cifar-10 world
    labels = []
    for i, s in enumerate(cifar10_classes):
        labels.append(fo.Classification(label=s))

    #
    # Load data into FiftyOne
    #

    timer = Timer()
    dataset = fo.Dataset("le_cifar10")

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

    # load the model using the model path config
    assert config.model_path
    model = simple_resnet().to(device).half()
    model.load_state_dict(torch.load(config.model_path))

    print("Model loaded.")

    model.train(False)

    # I need to get my datasets into a format where I'll have the ids available
    # as well during the data loading

    valid_imgs, valid_labels = zip(*valid_set)
    fo_valid_set = list(zip(valid_imgs, valid_labels, valid_ids))

    valid_batches = DataLoader(
        fo_valid_set, config.batch_size, shuffle=False, drop_last=False
    )

    print("running predictions on the data")

    correct = 0
    total = 0
    class_correct = list(0.0 for i in range(10))
    class_total = list(0.0 for i in range(10))
    with torch.no_grad():
        for images, labels, ids in valid_batches.dataloader:
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

            for prediction, theid, thelogit in zip(predicted, ids, y):
                label = fo.Classification(
                    label=cifar10_classes[prediction], logits=thelogit
                )
                dataset[theid]["prediction"] = label
                dataset[theid]["max-logit"] = float(
                    np.max(thelogit.cpu().numpy())
                )

    print(
        "Accuracy of the network on the 10K test images: %.2f%%"
        % (100 * correct / total)
    )

    for i in range(10):
        print(
            "Accuracy of %9s : %.2f%%"
            % (cifar10_classes[i], 100 * class_correct[i] / class_total[i])
        )

    print("done")

    return dataset


if __name__ == "__main__":

    config = commandline()

    print(f"running with config: {config}")

    dataset = main(config)
