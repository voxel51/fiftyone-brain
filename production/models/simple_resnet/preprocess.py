"""
Preprocessing functions

| Copyright 2017-2023, Voxel51, Inc.
| `voxel51.com <https://voxel51.com/>`_
|
"""
import copy
from functools import singledispatch

import numpy as np
import torch
from torch import nn


## Data Preprocessing and Handling
def preprocess(dataset, transforms):
    dataset = copy.copy(dataset)  # shallow copy
    for transform in transforms:
        dataset["data"] = transform(dataset["data"])
    return dataset


@singledispatch
def normalise(x, mean, std):
    return (x - mean) / std


@normalise.register(np.ndarray)
def _(x, mean, std):
    # faster inplace for numpy arrays
    x = np.array(x, np.float32)
    x -= mean
    x *= 1.0 / std
    return x


unnormalise = lambda x, mean, std: x * std + mean


@singledispatch
def pad(x, border):
    raise NotImplementedError


@pad.register(np.ndarray)
def _(x, border):
    return np.pad(
        x, [(0, 0), (border, border), (border, border), (0, 0)], mode="reflect"
    )


@pad.register(torch.Tensor)
def _(x, border):
    return nn.ReflectionPad2d(border)(x)


@singledispatch
def transpose(x, source, target):
    raise NotImplementedError


@transpose.register(np.ndarray)
def _(x, source, target):
    return x.transpose([source.index(d) for d in target])


@transpose.register(torch.Tensor)
def _(x, source, target):
    return x.permute([source.index(d) for d in target])


class Transform:
    def __init__(self, dataset, transforms):
        self.dataset, self.transforms = dataset, transforms
        self.choices = None

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        data, labels = self.dataset[index]
        data = data.copy()
        for choices, f in zip(self.choices, self.transforms):
            data = f(data, **choices[index])
        return data, labels

    def set_random_choices(self):
        self.choices = []
        x_shape = self.dataset[0][0].shape
        N = len(self)
        for t in self.transforms:
            self.choices.append(np.random.choice(t.options(x_shape), N))
            x_shape = (
                t.output_shape(x_shape)
                if hasattr(t, "output_shape")
                else x_shape
            )
