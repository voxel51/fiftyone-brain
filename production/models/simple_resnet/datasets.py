"""
Implementation of datasets for the experiments

| Copyright 2017-2023, Voxel51, Inc.
| `voxel51.com <https://voxel51.com/>`_
|
"""
from functools import lru_cache as cache

import torch

from utils import *


class DataLoader:
    def __init__(
        self,
        dataset,
        batch_size,
        shuffle,
        set_random_choices=False,
        num_workers=0,
        drop_last=False,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.set_random_choices = set_random_choices
        self.dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=True,
            shuffle=shuffle,
            drop_last=drop_last,
        )

    def __iter__(self):
        if self.set_random_choices:
            self.dataset.set_random_choices()
        return (
            {"input": x.to(device).half(), "target": y.to(device).long()}
            for (x, y) in self.dataloader
        )

    def __len__(self):
        return len(self.dataloader)


@cache(None)
def cifar10(root="./data"):
    try:
        import torchvision

        download = lambda train: torchvision.datasets.CIFAR10(
            root=root, train=train, download=True
        )
        return {
            k: {"data": v.data, "targets": v.targets}
            for k, v in [
                ("train", download(train=True)),
                ("valid", download(train=False)),
            ]
        }
    except ImportError:
        from tensorflow.keras import datasets

        (
            (train_images, train_labels),
            (valid_images, valid_labels),
        ) = datasets.cifar10.load_data()
        return {
            "train": {"data": train_images, "targets": train_labels.squeeze()},
            "valid": {"data": valid_images, "targets": valid_labels.squeeze()},
        }


cifar10_mean, cifar10_std = [
    (
        125.31,
        122.95,
        113.87,
    ),  # equals np.mean(cifar10()['train']['data'], axis=(0,1,2))
    (
        62.99,
        62.09,
        66.70,
    ),  # equals np.std(cifar10()['train']['data'], axis=(0,1,2))
]
cifar10_classes = "airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck".split(
    ", "
)
