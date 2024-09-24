#!/usr/bin/env python
"""
Installs `fiftyone-brain`.

| Copyright 2017-2024, Voxel51, Inc.
| `voxel51.com <https://voxel51.com/>`_
|
"""
import os
from setuptools import setup


VERSION = "0.18.0"


def get_version():
    if "RELEASE_VERSION" in os.environ:
        version = os.environ["RELEASE_VERSION"]
        if not version.startswith(VERSION):
            raise ValueError(
                "Release version doest not match version: %s and %s"
                % (version, VERSION)
            )
        return version

    return VERSION


with open("README.md", "r") as fh:
    long_description = fh.read()


setup(
    name="fiftyone-brain",
    version=get_version(),
    description="FiftyOne Brain",
    author="Voxel51, Inc.",
    author_email="info@voxel51.com",
    url="https://github.com/voxel51/fiftyone-brain",
    license="Apache",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=["fiftyone.brain"],
    include_package_data=True,
    install_requires=["numpy", "scipy>=1.2.0", "scikit-learn"],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Processing",
        "Topic :: Scientific/Engineering :: Image Recognition",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Topic :: Scientific/Engineering :: Visualization",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: POSIX :: Linux",
        "Operating System :: Microsoft :: Windows",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    scripts=[],
    python_requires=">=3.9",
)
