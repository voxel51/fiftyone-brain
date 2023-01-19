"""
Tests for :mod:`fiftyone.brain.internal.models.simple_resnet`.

| Copyright 2017-2023, Voxel51, Inc.
| `voxel51.com <https://voxel51.com/>`_
|
"""
import imageio
from PIL import Image
import pytest
import torch
import torchvision

import eta.core.image as etai

import fiftyone as fo
import fiftyone.core.utils as fou
import fiftyone.zoo as foz

import fiftyone.brain.internal.models as fbm


def _transpose(x, source, target):
    return x.permute([source.index(d) for d in target])


def _check_prediction(actual, expected):
    assert isinstance(actual, fo.Classification)
    assert isinstance(expected, fo.Classification)
    # @todo fix me on 3.9
    # assert actual.label == expected.label


def test_simple_resnet():
    dataset = foz.load_zoo_dataset(
        "cifar10",
        split="test",
        dataset_name=fo.get_default_dataset_name(),
        shuffle=True,
        max_samples=1,
    )

    sample = dataset.first()
    filepath = sample.filepath
    print("Working on image at %s" % filepath)

    img_pil = Image.open(filepath)
    print("img_pil is type %s" % type(img_pil))

    img_numpy = imageio.imread(filepath)
    print("img_numpy is type %s" % type(img_numpy))
    print(img_numpy.shape)

    img_torch = torch.from_numpy(img_numpy)
    img_torch = _transpose(img_torch, "HWC", "CHW")
    print("img_torch is type %s" % type(img_torch))
    print(img_torch.shape)
    assert tuple(reversed(img_torch.shape)) == img_numpy.shape

    img_eta = etai.read(filepath)
    print("img_eta is type %s" % type(img_eta))
    print(img_eta.shape)
    assert tuple(img_eta.shape) == img_numpy.shape

    model = fbm.load_model("simple-resnet-cifar10")

    with model:
        print("PIL")
        p_pil = model.predict(img_pil)
        print(p_pil)

        print("IMAGEIO")
        p_numpy = model.predict(img_numpy)
        print(p_numpy)
        _check_prediction(p_numpy, p_pil)

        print("ETA")
        p_eta = model.predict(img_eta)
        print(p_eta)
        _check_prediction(p_eta, p_pil)

        print("PIL (manual preprocessing)")
        with fou.SetAttributes(model, preprocess=False):
            img_tensor = model.transforms(img_pil)
            p_pil2 = model.predict(img_tensor)
            print(p_pil2)
            _check_prediction(p_pil2, p_pil)

        print("IMAGEIO (manual preprocessing)")
        with fou.SetAttributes(model, preprocess=False):
            img_tensor = model.transforms(img_numpy)
            p_numpy2 = model.predict(img_tensor)
            print(p_numpy2)
            _check_prediction(p_numpy2, p_numpy)


if __name__ == "__main__":
    test_simple_resnet()
