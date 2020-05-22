"""
Implementation of a simple Resnet that is suitable only for smallish data.

The original implementation of this is from David Page's work on fast model
training with resnets at https://github.com/davidcpage/cifar10-fast.

@todo This code needs to be significantly clean and tightened.  It is here now
just to get something in the codebase with a model that we can train for use in
a variety of tests and other things like developing uniqueness.

| Copyright 2017-2020, Voxel51, Inc.
| `voxel51.com <https://voxel51.com/>`_
|
"""
from os.path import normpath, sep
from collections import namedtuple

import numpy as np
from PIL import Image as PILImage
import torch
from torch import nn
import torchvision

from eta.core.config import Config, ConfigError
import eta.core.data as etad
import eta.core.learning as etal
import eta.core.models as etam


# This is a small model with a fixed size, so let cudnn optimize
torch.backends.cudnn.benchmark = True

__use_gpu__ = torch.cuda.is_available()
__device__ = torch.device("cuda:0" if __use_gpu__ else "cpu")


class SimpleResnetImageClassifierConfig(
    Config, etal.HasDefaultDeploymentConfig
):
    """SimpleResnetImageClassifier configuration settings.

    Attributes:
        model_name: the name of the published model to load.  If this value is
            provided, ``model_path`` does not need to be
        model_path: the path to the model ``.pth`` weights to use.  If this
            value is provided, ``model_name`` does not need to be
        labels_string: a comma-separated list of the class-names in the
            classifier, ordered in accordance with the trained model
        image_mean: a 3-array of mean values in ``[0, 1]`` for preprocessing
            the input
        image_std: a 3-array of std values in ``[0, 1]`` for preprocessing the
            input
    """

    def __init__(self, d):
        self.model_name = self.parse_string(d, "model_name", default=None)
        self.model_path = self.parse_string(d, "model_path", default=None)

        if self.model_name:
            d = self.load_default_deployment_params(d, self.model_name)

        self.image_mean = self.parse_array(d, "image_mean", default=None)
        self.image_std = self.parse_array(d, "image_std", default=None)

        self.labels_string = self.parse_string(
            d, "labels_string", default=None
        )

        self._validate()

    def _validate(self):
        if not self.model_name and not self.model_path:
            raise ConfigError(
                "Either `model_name` or `model_path` must be provided"
            )

        if not self.image_mean or not self.image_std:
            raise ConfigError(
                "Both `image_mean` and `image_std` must be provided"
            )


class SimpleResnetImageClassifier(etal.ImageClassifier):
    """A simple Resnet implementation.

    This implementation assumes that the preprocessing transformations have
    already been applied to the images before they are sent to the prediction
    and related functions.  If you want these functions to perform the
    preprocessing for you, then you need to call `toggle_preprocess()` first.
    Typically, this is not needed when you are working with Torch DataLoaders,
    for example.

    Current implementation supports single processing of Numpy Arrays and PIL
    Images (with or without preprocessing) and batch processing of Torch
    Tensors (already preprocessed).

    Attributes:
        config: a :class:`SimpleResnetImageClassifierConfig`
    """

    """
    @todo Can we make a more generic bridge to ETA for Torch models that would
    specify some location or function that defines the graph but is otherwise
    generic?
    """

    def __init__(self, config):
        self.config = config

        if self.config.model_path:
            self.weights_path = self.config.model_path
        else:
            self.weights_path = etam.download_model(self.config.model_name)

        self.labels_map = self.config.labels_string.split(", ")
        self.labels_rev = {v: i for i, v in enumerate(self.labels_map)}

        self._transforms = None
        self._model = None

        # This is toggled via toggle_preprocess()
        self._preprocess = False

        self._setup_model()

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

    @property
    def transforms(self):
        return self._transforms

    @property
    def model(self):
        return self._model

    def predict(self, img):
        """Computes the prediction on a single image.

        Args:
            img: a PIL image, numpy array or Torch tensor (CHW)

        Returns:
            an ``eta.core.data.AttributeContainer``
        """
        if isinstance(img, torch.Tensor):
            raise NotImplementedError("predict cannot accept torch.Tensor")

        img = self._preprocess_if_needed(img)

        # need to check the shape of img to ensure that it meets the contract,
        # since the user may have passed in the numpy array directly and it was
        # not preprocessed
        if len(img.shape) != 4:
            if isinstance(img, np.ndarray):
                img = img[np.newaxis, :]
            elif isinstance(img, torch.Tensor):
                img = img.unsqueeze(0)

        return self.predict_all(img)[0]

    def predict_all(self, imgs):
        """Computes predictions for the tensor of images.

        This method assumes that the images are already preprocessed.

        Args:
            imgs: a Torch tensor or numpy array of images (NCHW)

        Returns:
            a list of ``eta.core.data.AttributeContainer`` instances
        """
        if isinstance(imgs, np.ndarray):
            imgs = torch.from_numpy(imgs)

        if __use_gpu__:
            imgs = imgs.cuda().half()

        inputs = dict(input=imgs)
        outputs = self._model(inputs)
        logits = np.float32(outputs["logits"].detach().cpu().numpy())
        predictions = np.argmax(logits, axis=1)
        odds = np.exp(logits)
        confidences = np.max(odds, axis=1) / np.sum(odds, axis=1)

        attributes = []
        for prediction, confidence in zip(predictions, confidences):
            attr = etad.CategoricalAttribute(
                "label", self.labels_map[prediction], confidence=confidence
            )
            container = etad.AttributeContainer.from_iterable([attr])
            attributes.append(container)

        return attributes

    def embed_all(self, imgs):
        """Embeds the imgs into the model's space.

        Args:
            imgs: a Torch tensor or numpy array of images (NCHW)

        Returns:
            a ``num_ims x dim`` array of embeddings
        """

        #
        # @todo Should this be an implementation of the get_features?
        #
        # unclear if the layer should be flatten or linear
        #

        imgs = self._preprocess_if_needed(imgs)

        if __use_gpu__:
            imgs = imgs.cuda().half()

        inputs = dict(input=imgs)
        outputs = self._model(inputs)
        return np.float32(outputs["flatten"].detach().cpu().numpy())

    def toggle_preprocess(self, set_to=None):
        """Toggles the preprocessing boolean.

        Args:
            set_to (None): force preprocessing to True/False rather than
                toggling
        """
        if set_to:
            assert isinstance(set_to, bool)
            self._preprocess = set_to
        else:
            self._preprocess = not self._preprocess

    def _preprocess_if_needed(self, img):
        """Preprocesses the single image through the transforms if needed."""
        if self._preprocess:
            print("preprocessing")
            if isinstance(img, torch.Tensor):
                return NotImplementedError(
                    "Cannot preprocess Tensors at this time."
                )
            if isinstance(img, np.ndarray):
                print("preprocessing from ndarray")
                # CONVERT TO PIL
                # need to separately process each image
                if np.max(img) <= 1.00001:
                    img = img * 255
                img = PILImage.fromarray(np.uint8(img))

            img = self._transforms(img)

        return img


#
## Utils; should they be moved elsewhere?
#
def path_iter(nested_dict, pfx=()):
    for name, val in nested_dict.items():
        if isinstance(val, dict):
            yield from path_iter(val, (*pfx, name))
        else:
            yield ((*pfx, name), val)


#
## Define the network
#
has_inputs = lambda node: type(node) is tuple


def build_graph(net):
    flattened = pipeline(net)
    resolve_input = lambda rel_path, path, idx: (
        normpath(sep.join((path, "..", rel_path)))
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
        (sep.join(path), (node if has_inputs(node) else (node, [-1])))
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
