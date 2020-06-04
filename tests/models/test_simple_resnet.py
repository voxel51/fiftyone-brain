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

import imageio
from PIL import Image
import torch
import torchvision

import eta.core.image as etai
import eta.core.learning as etal

import fiftyone.brain as fob
import fiftyone.zoo as foz
import fiftyone.core.odm as foo


def transpose(x, source, target):
    """Simple transpose function for Tensors."""
    return x.permute([source.index(d) for d in target])


def test_simple_resnet():
    foo.drop_database()
    dataset = foz.load_zoo_dataset("cifar10", split="test")
    view = dataset.view().take(100)
    sample = next(iter(view))
    filepath = sample.filepath
    print("Working on image at %s" % filepath)

    im_pil = Image.open(filepath)
    print(f"im_pil is type {type(im_pil)}")

    im_numpy = imageio.imread(filepath)
    print(f"im_numpy is type {type(im_numpy)}")
    print(im_numpy.shape)

    im_torch = torch.from_numpy(im_numpy)
    im_torch = transpose(im_torch, "HWC", "CHW")
    print(f"im_torch is type {type(im_torch)}")
    print(im_torch.shape)

    im_eta = etai.read(filepath)
    print(f"im_eta is type {type(im_eta)}")
    print(im_eta.shape)

    im_et2 = im_eta / 255
    print(f"im_et2 is type {type(im_et2)}")
    print(im_et2.shape)

    model = etal.load_default_deployment_model("simple_resnet_cifar10")

    model.toggle_preprocess()

    print("PIL")
    p = model.predict(im_pil)
    print(p[0])

    print("IMAGEIO")
    p = model.predict(im_numpy)
    print(p[0])

    print("TORCH")
    try:
        p = model.predict(im_torch)
        print(p[0])
    except NotImplementedError:
        print("successfully raised error on torch image")

    print("ETA")
    p = model.predict(im_eta)
    print(p[0])

    print("ET2")
    p = model.predict(im_et2)
    print(p[0])

    # @todo logging a need to understand why the predictions on the ETA images are
    # different than those on the other formats. -- just by a confidence level
