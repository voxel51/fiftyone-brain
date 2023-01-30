"""
Uniqueness tests.

| Copyright 2017-2023, Voxel51, Inc.
| `voxel51.com <https://voxel51.com/>`_
|
"""
import os
import unittest

import eta.core.storage as etas
import eta.core.utils as etau

import fiftyone as fo
import fiftyone.brain as fob
import fiftyone.zoo as foz


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
    with etau.TempDir() as tmpdir:
        tmp_zip = os.path.join(tmpdir, "data.zip")
        tmp_data = os.path.join(tmpdir, "brain_grayscale_test_data")
        client = etas.GoogleDriveStorageClient()
        client.download("1ECeNnLmKQCHxlVdRqGefV5eXOD_OkmWx", tmp_zip)
        etau.extract_zip(tmp_zip, delete_zip=True)

        dataset = fo.Dataset.from_dir(tmp_data, fo.types.ImageDirectory)

        fob.compute_uniqueness(dataset)
        print(dataset)


if __name__ == "__main__":
    fo.config.show_progress_bars = True
    unittest.main(verbosity=2)
