#!/usr/bin/env python
"""
Installs `fiftyone-brain`.

All static package metadata lives in pyproject.toml; this shim exists to
support the RELEASE_VERSION environment variable, which the build workflow
uses to build release candidate versions.

| Copyright 2017-2026, Voxel51, Inc.
| `voxel51.com <https://voxel51.com/>`_
|
"""
import os

from setuptools import setup

VERSION = "0.24.0"


def get_version():
    if "RELEASE_VERSION" in os.environ:
        version = os.environ["RELEASE_VERSION"]
        if not version.startswith(VERSION):
            raise ValueError(
                "Release version does not match version: %s and %s"
                % (version, VERSION)
            )
        return version

    return VERSION


setup(version=get_version())
