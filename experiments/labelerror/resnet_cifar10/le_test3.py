"""
le_test3 --> initial tests for using fiftyone with this code
Loads the data, assigns ground-truth labels, loads a model
Runs prediction and associates predictions with the fiftyone dataset

| Copyright 2017-2023, Voxel51, Inc.
| `voxel51.com <https://voxel51.com/>`_
|
"""
from functools import partial
import os
import time

import ipdb
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

    if config.take:
        whole_train_set = whole_train_set[: config.take]
        valid_set = whole_train_set[: config.take]
        print(f"using a subset of the data")
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

    # load the model using the model path config
    assert config.model_path
    model = Network(simple_resnet()).to(device).half()
    model.load_state_dict(torch.load(config.model_path))

    model.train(False)  # == model.eval()

    # I need to get my datasets into a format where I'll have the ids available
    # as well during the data loading
    train_imgs, train_labels = zip(*whole_train_set)
    fo_train_set = list(zip(train_imgs, train_labels, train_ids))

    valid_imgs, valid_labels = zip(*valid_set)
    fo_valid_set = list(zip(valid_imgs, valid_labels, valid_ids))

    train_batches = DataLoader(
        fo_train_set, config.batch_size, shuffle=False, drop_last=False
    )
    valid_batches = DataLoader(
        fo_valid_set, config.batch_size, shuffle=False, drop_last=False
    )

    correct = 0
    total = 0
    class_correct = list(0.0 for i in range(10))
    class_total = list(0.0 for i in range(10))
    with torch.no_grad():
        for images, labels, sample_ids in valid_batches.dataloader:
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

            # Add predictions to FiftyOne dataset
            for prediction, sample_id, logits in zip(predicted, sample_ids, y):
                sample = dataset[sample_id]
                sample["prediction"] = fo.Classification(
                    label=cifar10_classes[prediction], logits=logits,
                )
                sample["max-logit"] = np.max(logits.cpu().numpy())
                sample.save()

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

    if config.start_ipython:
        ipdb.set_trace()

    return dataset


if __name__ == "__main__":

    config = commandline()

    print(f"running with config: {config}")

    dataset = main(config)
