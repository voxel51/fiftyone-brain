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

import os.path

import eta.core.storage as etas
import eta.core.utils as etau

import fiftyone.brain as fob
import fiftyone.zoo as foz
from fiftyone.core.dataset import Dataset
import fiftyone.types as fot


def test_uniqueness():
    dataset = foz.load_zoo_dataset("cifar10", split="test")
    assert "uniqueness" not in dataset.get_field_schema()

    view = dataset.view().take(100)
    fob.compute_uniqueness(view)

    print(dataset.summary())
    assert "uniqueness" in dataset.get_field_schema()


def test_gray():
    """Test default support for handling grayscale images.

    Uses a test data zip at this location:
    https://drive.google.com/file/d/1ECeNnLmKQCHxlVdRqGefV5eXOD_OkmWx/view?usp=sharing

    Requires Google Drive Voxel51 credentials to work; see the
    eta/docs/storage_dev_guide.md.
    """
    with etau.TempDir() as tempdir:
        tmp_zip = os.path.join(tempdir, "data.zip")
        tmp_data = os.path.join(tempdir, "brain_grayscale_test_data")
        client = etas.GoogleDriveStorageClient()
        client.download("1ECeNnLmKQCHxlVdRqGefV5eXOD_OkmWx", tmp_zip)
        etau.extract_zip(tmp_zip, delete_zip=True)

        dataset = Dataset.from_dir(tmp_data, fot.ImageDirectory)

        fob.compute_uniqueness(dataset)

        print(dataset.summary())


if __name__ == "__main__":
    test_uniqueness()
    test_gray()
