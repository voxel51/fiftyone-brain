"""
Test drivers for uniqueness

| Copyright 2017-2020, Voxel51, Inc.
| `voxel51.com <https://voxel51.com/>`_
|
"""
import os

import eta.core.storage as etas
import eta.core.utils as etau

import fiftyone as fo
import fiftyone.zoo as foz

import fiftyone.brain as fob


def test_uniqueness():
    dataset = foz.load_zoo_dataset("cifar10", split="test")
    assert "uniqueness" not in dataset.get_field_schema()

    view = dataset.view().take(100)
    fob.compute_uniqueness(view)

    print(dataset)
    assert "uniqueness" in dataset.get_field_schema()


def test_gray():
    """Test default support for handling grayscale images.

    Requires Voxel51 Google Drive credentials to download the test data.
    """
    with etau.TempDir() as tempdir:
        tmp_zip = os.path.join(tempdir, "data.zip")
        tmp_data = os.path.join(tempdir, "brain_grayscale_test_data")
        client = etas.GoogleDriveStorageClient()
        client.download("1ECeNnLmKQCHxlVdRqGefV5eXOD_OkmWx", tmp_zip)
        etau.extract_zip(tmp_zip, delete_zip=True)

        dataset = fo.Dataset.from_dir(tmp_data, fo.types.ImageDirectory)

        fob.compute_uniqueness(dataset)

        print(dataset)


if __name__ == "__main__":
    test_uniqueness()
    test_gray()
