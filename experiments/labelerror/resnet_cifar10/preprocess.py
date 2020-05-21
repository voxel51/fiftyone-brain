"""
Preprocessing functions

| Copyright 2017-2020, Voxel51, Inc.
| `voxel51.com <https://voxel51.com/>`_
|
"""
import copy
from functools import singledispatch

import numpy as np
import torch

## Data Preprocessing and Handling
def preprocess(dataset, transforms):
    dataset = copy.copy(dataset) #shallow copy
    for transform in transforms:
        dataset['data'] = transform(dataset['data'])
    return dataset

@singledispatch
def normalise(x, mean, std):
    return (x - mean) / std

@normalise.register(np.ndarray)
def _(x, mean, std):
    #faster inplace for numpy arrays
    x = np.array(x, np.float32)
    x -= mean
    x *= 1.0/std
    return x

unnormalise = lambda x, mean, std: x*std + mean

@singledispatch
def pad(x, border):
    raise NotImplementedError

@pad.register(np.ndarray)
def _(x, border):
    return np.pad(x, [(0, 0), (border, border), (border, border), (0, 0)], mode='reflect')

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

