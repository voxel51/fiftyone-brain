"""
Test drivers for uniqueness

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

import fiftyone.brain as fob
import fiftyone.zoo as foz
import fiftyone.core.odm as foo
from fiftyone.core.dataset import Dataset
import fiftyone.types as fot


def test_uniqueness():
    foo.drop_database()
    dataset = foz.load_zoo_dataset("cifar10", split="test")
    assert "uniqueness" not in dataset.get_field_schema()

    view = dataset.view().take(100)
    fob.compute_uniqueness(view)

    print(dataset.summary())
    assert "uniqueness" in dataset.get_field_schema()


def test_gray():
    """Test default support for handling grayscale images."""
    foo.drop_database()

    dataset = Dataset.from_dir(
        "/home/jason/Downloads/data/test", fot.ImageDirectory
    )

    print(dataset.summary())


if __name__ == "__main__":
    # test_uniqueness()
    test_gray()
