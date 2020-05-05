#!/usr/bin/env python
"""
Installs `fiftyone-brain`.

| Copyright 2017-2020, Voxel51, Inc.
| `voxel51.com <https://voxel51.com/>`_
|
"""
from setuptools import setup, find_packages


setup(
    name="fiftyone-brain",
    version="0.1.0",
    description="FiftyOne Brain",
    author="Voxel51, Inc.",
    author_email="info@voxel51.com",
    url="https://github.com/voxel51/fiftyone-brain",
    license="",
    packages=find_packages(),
    include_package_data=True,
    classifiers=[
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: POSIX :: Linux",
        "Programming Language :: Python :: 2.7",
        "Programming Language :: Python :: 3",
    ],
    scripts=[],
    python_requires=">=2.7",
)
