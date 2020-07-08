"""
Test drivers for the simple_resnet code.

| Copyright 2017-2020, Voxel51, Inc.
| `voxel51.com <https://voxel51.com/>`_
|
"""
# pragma pylint: disable=redefined-builtin
# pragma pylint: disable=unused-wildcard-import
# pragma pylint: disable=wildcard-import
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from builtins import *

# pragma pylint: enable=redefined-builtin
# pragma pylint: enable=unused-wildcard-import
# pragma pylint: enable=wildcard-import

import pytest

import imageio
from PIL import Image
import torch
import torchvision

import eta.core.data as etad
import eta.core.image as etai
import eta.core.learning as etal

import fiftyone.brain as fob
import fiftyone.zoo as foz


def transpose(x, source, target):
    """Simple transpose function for Tensors."""
    return x.permute([source.index(d) for d in target])


def check_prediction(actual, expected):
    """Compare two eta.core.data.CategoricalAttribute instances."""
    assert isinstance(actual, etad.CategoricalAttribute)
    assert isinstance(expected, etad.CategoricalAttribute)
    actual = actual.serialize()
    expected = expected.serialize()
    for k in actual:
        if k == "confidence":
            continue
        assert actual[k] == expected[k], k


def test_simple_resnet():
    dataset = foz.load_zoo_dataset("cifar10", split="test")
    view = dataset.view().take(100)
    sample = next(iter(view))
    filepath = sample.filepath
    print("Working on image at %s" % filepath)

    im_pil = Image.open(filepath)
    print("im_pil is type %s" % type(im_pil))

    im_numpy = imageio.imread(filepath)
    print("im_numpy is type %s" % type(im_numpy))
    print(im_numpy.shape)

    im_torch = torch.from_numpy(im_numpy)
    im_torch = transpose(im_torch, "HWC", "CHW")
    print("im_torch is type %s" % type(im_torch))
    print(im_torch.shape)
    assert tuple(reversed(im_torch.shape)) == im_numpy.shape

    im_eta = etai.read(filepath)
    print("im_eta is type %s" % type(im_eta))
    print(im_eta.shape)
    assert tuple(im_eta.shape) == im_numpy.shape

    im_et2 = im_eta / 255
    print("im_et2 is type %s" % type(im_et2))
    print(im_et2.shape)
    assert tuple(im_et2.shape) == im_numpy.shape

    model = etal.load_default_deployment_model("simple_resnet_cifar10")

    model.toggle_preprocess()

    print("PIL")
    p = model.predict(im_pil)
    print(p[0])
    p_pil = p[0]

    print("IMAGEIO")
    p = model.predict(im_numpy)
    print(p[0])
    check_prediction(p[0], p_pil)

    print("TORCH")
    with pytest.raises(NotImplementedError):
        p = model.predict(im_torch)
        print(p[0])
    print("successfully raised error on torch image")

    print("ETA")
    p = model.predict(im_eta)
    print(p[0])
    check_prediction(p[0], p_pil)

    print("ET2")
    p = model.predict(im_et2)
    print(p[0])
    check_prediction(p[0], p_pil)

    # @todo logging a need to understand why the predictions on the ETA images are
    # different than those on the other formats. -- just by a confidence level


if __name__ == "__main__":
    test_simple_resnet()
