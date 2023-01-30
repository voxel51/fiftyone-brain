"""
Implementation of a simple ResNet that is suitable only for smallish data.

The original implementation of this is from David Page's work on fast model
training with resnets at https://github.com/davidcpage/cifar10-fast.

| Copyright 2017-2023, Voxel51, Inc.
| `voxel51.com <https://voxel51.com/>`_
|
"""
from collections import namedtuple
import os

import numpy as np
import torch
from torch import nn


def simple_resnet(
    channels=None,
    weight=0.125,
    pool=nn.MaxPool2d(2),
    extra_layers=(),
    res_layers=("layer1", "layer3"),
):
    channels = channels or {
        "prep": 64,
        "layer1": 128,
        "layer2": 256,
        "layer3": 512,
    }
    net = {
        "input": (None, []),
        "prep": conv_bn(3, channels["prep"]),
        "layer1": dict(
            conv_bn(channels["prep"], channels["layer1"]), pool=pool
        ),
        "layer2": dict(
            conv_bn(channels["layer1"], channels["layer2"]), pool=pool
        ),
        "layer3": dict(
            conv_bn(channels["layer2"], channels["layer3"]), pool=pool
        ),
        "pool": nn.MaxPool2d(4),
        "flatten": Flatten(),
        "linear": nn.Linear(channels["layer3"], 10, bias=False),
        "logits": Mul(weight),
    }
    for layer in res_layers:
        net[layer]["residual"] = residual(channels[layer])

    for layer in extra_layers:
        net[layer]["extra"] = conv_bn(channels[layer], channels[layer])

    return Network(net, input_layer="input", output_layer="logits")


class Network(nn.Module):
    def __init__(self, net, input_layer=None, output_layer=None):
        super().__init__()
        self.input_layer = input_layer
        self.output_layer = output_layer
        self.graph = build_graph(net)
        for path, (val, _) in self.graph.items():
            setattr(self, path.replace("/", "_"), val)

    def nodes(self):
        return (node for node, _ in self.graph.values())

    def forward(self, inputs):
        if self.input_layer:
            outputs = {self.input_layer: inputs}
        else:
            outputs = dict(inputs)

        for k, (node, ins) in self.graph.items():
            # only compute nodes that are not supplied as inputs.
            if k not in outputs:
                outputs[k] = node(*[outputs[x] for x in ins])

        if self.output_layer:
            return outputs[self.output_layer]

        return outputs

    def half(self):
        for node in self.nodes():
            if isinstance(node, nn.Module) and not isinstance(
                node, nn.BatchNorm2d
            ):
                node.half()

        return self


def has_inputs(node):
    return type(node) is tuple


def build_graph(net):
    flattened = pipeline(net)
    resolve_input = lambda rel_path, path, idx: (
        os.path.normpath(os.path.sep.join((path, "..", rel_path)))
        if isinstance(rel_path, str)
        else flattened[idx + rel_path][0]
    )
    return {
        path: (
            node[0],
            [resolve_input(rel_path, path, idx) for rel_path in node[1]],
        )
        for idx, (path, node) in enumerate(flattened)
    }


def pipeline(net):
    return [
        (os.path.sep.join(path), (node if has_inputs(node) else (node, [-1])))
        for (path, node) in path_iter(net)
    ]


class Crop(namedtuple("Crop", ("h", "w"))):
    def __call__(self, x, x0, y0):
        return x[..., y0 : y0 + self.h, x0 : x0 + self.w]

    def options(self, shape):
        *_, H, W = shape
        return [
            {"x0": x0, "y0": y0}
            for x0 in range(W + 1 - self.w)
            for y0 in range(H + 1 - self.h)
        ]

    def output_shape(self, shape):
        *_, H, W = shape
        return (*_, self.h, self.w)


class FlipLR(namedtuple("FlipLR", ())):
    def __call__(self, x, choice):
        if isinstance(x, np.ndarray):
            return x[..., ::-1].copy()

        return torch.flip(x, [-1]) if choice else x

    def options(self, shape):
        return [{"choice": b} for b in [True, False]]


class Cutout(namedtuple("Cutout", ("h", "w"))):
    def __call__(self, x, x0, y0):
        x[..., y0 : y0 + self.h, x0 : x0 + self.w] = 0.0
        return x

    def options(self, shape):
        *_, H, W = shape
        return [
            {"x0": x0, "y0": y0}
            for x0 in range(W + 1 - self.w)
            for y0 in range(H + 1 - self.h)
        ]


class PiecewiseLinear(namedtuple("PiecewiseLinear", ("knots", "vals"))):
    def __call__(self, t):
        return np.interp([t], self.knots, self.vals)[0]


class Const(namedtuple("Const", ["val"])):
    def __call__(self, x):
        return self.val


class Identity(namedtuple("Identity", [])):
    def __call__(self, x):
        return x


class Add(namedtuple("Add", [])):
    def __call__(self, x, y):
        return x + y


class AddWeighted(namedtuple("AddWeighted", ["wx", "wy"])):
    def __call__(self, x, y):
        return self.wx * x + self.wy * y


class Mul(nn.Module):
    def __init__(self, weight):
        super().__init__()
        self.weight = weight

    def __call__(self, x):
        return x * self.weight


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), x.size(1))


class Concat(nn.Module):
    def forward(self, *xs):
        return torch.cat(xs, 1)


class BatchNorm(nn.BatchNorm2d):
    def __init__(
        self,
        num_features,
        eps=1e-05,
        momentum=0.1,
        weight_freeze=False,
        bias_freeze=False,
        weight_init=1.0,
        bias_init=0.0,
    ):
        super().__init__(num_features, eps=eps, momentum=momentum)
        if weight_init is not None:
            self.weight.data.fill_(weight_init)

        if bias_init is not None:
            self.bias.data.fill_(bias_init)

        self.weight.requires_grad = not weight_freeze
        self.bias.requires_grad = not bias_freeze


def conv_bn(c_in, c_out):
    return {
        "conv": nn.Conv2d(
            c_in, c_out, kernel_size=3, stride=1, padding=1, bias=False
        ),
        "bn": BatchNorm(c_out),
        "relu": nn.ReLU(True),
    }


def residual(c):
    return {
        "in": Identity(),
        "res1": conv_bn(c, c),
        "res2": conv_bn(c, c),
        "add": (Add(), ["in", "res2/relu"]),
    }


def path_iter(nested_dict, pfx=()):
    for name, val in nested_dict.items():
        if isinstance(val, dict):
            yield from path_iter(val, (*pfx, name))
        else:
            yield ((*pfx, name), val)


MODEL = "model"
VALID_MODEL = "valid_model"
OUTPUT = "output"
