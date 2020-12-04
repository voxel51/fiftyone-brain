"""
Implementation of a simple ResNet that is suitable only for smallish data.

The original implementation of this is from David Page's work on fast model
training with resnets at https://github.com/davidcpage/cifar10-fast.

| Copyright 2017-2020, Voxel51, Inc.
| `voxel51.com <https://voxel51.com/>`_
|
"""
from collections import namedtuple
import os

import numpy as np
import torch
from torch import nn
import torchvision

from eta.core.config import Config, ConfigError
import eta.core.data as etad
import eta.core.learning as etal


# This is a small model with a fixed size, so let cudnn optimize
torch.backends.cudnn.benchmark = True

__use_gpu__ = torch.cuda.is_available()
__device__ = torch.device("cuda:0" if __use_gpu__ else "cpu")


class SimpleResnetImageClassifierConfig(Config, etal.HasPublishedModel):
    """:class:`SimpleResnetImageClassifier` configuration settings.

    Attributes:
        model_name: the name of the published model to load. If this value is
            provided, ``model_path`` does not need to be
        model_path: the path to the model ``.pth`` weights to use. If this
            value is provided, ``model_name`` does not need to be
        labels_string: a comma-separated list of the class-names in the
            classifier, ordered in accordance with the trained model
        image_mean: a 3-array of mean values in ``[0, 1]`` for preprocessing
            the input
        image_std: a 3-array of std values in ``[0, 1]`` for preprocessing the
            input
    """

    def __init__(self, d):
        d = self.init(d)

        self.image_mean = self.parse_array(d, "image_mean")
        self.image_std = self.parse_array(d, "image_std")

        self.labels_string = self.parse_string(
            d, "labels_string", default=None
        )


# @todo Can we make a more generic bridge to ETA for Torch models that
# would specify some location or function that defines the graph but is
# otherwise generic?


class SimpleResnetImageClassifier(
    etal.ImageClassifier, etal.ExposesFeatures, etal.ExposesProbabilities
):
    """A simple ResNet implementation.

    This model exposes embeddings and probabilities for its predictions.

    The model requires preprocessing defined by its :meth:`transforms`
    property. By default, you are responsible for applying this preprocessing
    prior to performing prediction (usually via a Torch DataLoader).

    Alternatively, you can call :meth:`toggle_preprocess` to set the
    :meth:`preprocess` property to ``True``. Then any images you supply will
    have preprocessing applied to them before performing prediction.

    Args:
        config: a :class:`SimpleResnetImageClassifierConfig` defining the
            model to load
    """

    def __init__(self, config):
        self.config = config

        self.config.download_model_if_necessary()
        self.weights_path = self.config.model_path

        self._class_labels = self.config.labels_string.split(",")
        self._num_classes = len(self._class_labels)

        self._transforms = None
        self._model = None
        self._last_features = None
        self._last_probs = None
        self._preprocess = False

        self._setup_model()

    @property
    def preprocess(self):
        """Whether preprocessing will be applied during prediction by the
        model.
        """
        return self._preprocess

    @property
    def transforms(self):
        """The ``torchvision.transforms`` that must be applied to each image
        before prediction.
        """
        return self._transforms

    @property
    def is_multilabel(self):
        return False

    @property
    def exposes_features(self):
        return True

    @property
    def exposes_probabilities(self):
        return True

    @property
    def num_classes(self):
        return self._num_classes

    @property
    def class_labels(self):
        return self._class_labels

    @property
    def features_dim(self):
        # @todo support getting this information?
        return None

    def get_features(self):
        return self._last_features

    def get_probabilities(self):
        return self._last_probs

    def toggle_preprocess(self, set_to=None):
        """Toggles the preprocessing state of the model.

        Args:
            set_to (None): explicitly set preprocessing state to this value
                rather than toggling
        """
        if set_to is not None:
            self._preprocess = set_to
        else:
            self._preprocess = not self._preprocess

    def predict(self, img):
        """Computes the prediction on a single image.

        If `self.preprocess == True`, the input must be a PIL image.

        Args:
            img: a PIL image (HWC), a numpy array (HWC) or Torch tensor (CHW)

        Returns:
            an ``eta.core.data.AttributeContainer``
        """
        if isinstance(img, torch.Tensor):
            imgs = img.unsqueeze(0)
        else:
            imgs = [img]

        return self.predict_all(imgs)[0]

    def predict_all(self, imgs):
        """Computes predictions for the tensor of images.

        If `self.preprocess == True`, the input must be a list of PIL images.

        Args:
            imgs: a list of PIL or numpy images (HWC), a numpy array of images
                (NHWC), or a Torch tensor (NCHW)

        Returns:
            a list of ``eta.core.data.AttributeContainer`` instances
        """
        logits = self._predict_all(imgs)

        predictions = np.argmax(logits, axis=1)
        odds = np.exp(logits)
        odds /= np.sum(odds, axis=1, keepdims=True)
        confidences = np.max(odds, axis=1)

        self._last_probs = np.expand_dims(odds, axis=1)

        return self._make_predictions(predictions, confidences)

    def _predict_all(self, imgs):
        imgs = self._preprocess_batch(imgs)

        if __use_gpu__:
            imgs = imgs.cuda().half()

        inputs = dict(input=imgs)
        outputs = self._model(inputs)

        self._last_features = (
            outputs["flatten"].detach().cpu().numpy()
        ).astype(np.float32, copy=False)

        return np.float32(outputs["logits"].detach().cpu().numpy())

    def _preprocess_batch(self, imgs):
        if self._preprocess:
            return torch.stack([self._transforms(img) for img in imgs])

        if isinstance(imgs, torch.Tensor):
            return imgs

        # Converts PIL/numpy (HWC) to Torch tensor (CHW)
        t = torchvision.transforms.ToTensor()
        return torch.stack([t(img) for img in imgs])

    def _make_predictions(self, predictions, confidences):
        attributes = []
        for prediction, confidence in zip(predictions, confidences):
            attr = etad.CategoricalAttribute(
                "label", self._class_labels[prediction], confidence=confidence
            )
            container = etad.AttributeContainer.from_iterable([attr])
            attributes.append(container)

        return attributes

    def _setup_model(self):
        # Instantiates the model and sets up any preprocessing, etc.
        self._transforms = torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize([32, 32]),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(
                    self.config.image_mean, self.config.image_std
                ),
            ]
        )

        # Load the model
        model = Network(simple_resnet()).to(__device__)
        if __use_gpu__:
            model = model.half()

        model.load_state_dict(
            torch.load(self.weights_path, map_location=__device__)
        )
        model.train(False)

        self._model = model


#
# Utils
#


def path_iter(nested_dict, pfx=()):
    for name, val in nested_dict.items():
        if isinstance(val, dict):
            yield from path_iter(val, (*pfx, name))
        else:
            yield ((*pfx, name), val)


#
# Network definition
#


has_inputs = lambda node: type(node) is tuple


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


class Network(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.graph = build_graph(net)
        for path, (val, _) in self.graph.items():
            setattr(self, path.replace("/", "_"), val)

    def nodes(self):
        return (node for node, _ in self.graph.values())

    def forward(self, inputs):
        outputs = dict(inputs)
        for k, (node, ins) in self.graph.items():
            # only compute nodes that are not supplied as inputs.
            if k not in outputs:
                outputs[k] = node(*[outputs[x] for x in ins])

        return outputs

    def half(self):
        for node in self.nodes():
            if isinstance(node, nn.Module) and not isinstance(
                node, nn.BatchNorm2d
            ):
                node.half()

        return self


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
    n = {
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
        n[layer]["residual"] = residual(channels[layer])

    for layer in extra_layers:
        n[layer]["extra"] = conv_bn(channels[layer], channels[layer])

    return n


MODEL = "model"
VALID_MODEL = "valid_model"
OUTPUT = "output"
