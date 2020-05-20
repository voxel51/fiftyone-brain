#!/usr/bin/env python
"""
Installs `fiftyone-brain`.

| Copyright 2017-2020, Voxel51, Inc.
| `voxel51.com <https://voxel51.com/>`_
|
"""
import os
import shutil
from distutils.command.build import build
from setuptools import setup
from wheel.bdist_wheel import bdist_wheel


class CustomBuild(build):
    def run(self):
        from pyarmor.pyarmor import main as call_pyarmor

        build.run(self)
        # remove the source and bytecode (.pyc) files, and replace them with
        # obfuscated files
        brain_dir = os.path.join(self.build_lib, "fiftyone", "brain")
        shutil.rmtree(brain_dir)
        call_pyarmor(
            [
                "obfuscate",
                "--recursive",
                "--output",
                brain_dir,
                os.path.join("fiftyone", "brain", "__init__.py"),
            ]
        )


class CustomBdistWheel(bdist_wheel):
    def finalize_options(self):
        bdist_wheel.finalize_options(self)
        # not pure Python - pytransform shared lib from pyarmor is OS-dependent
        self.root_is_pure = False


cmdclass = {
    "build": CustomBuild,
    "bdist_wheel": CustomBdistWheel,
}

setup(
    name="fiftyone-brain",
    version="0.1.0",
    description="FiftyOne Brain",
    author="Voxel51, Inc.",
    author_email="info@voxel51.com",
    url="https://github.com/voxel51/fiftyone-brain",
    license="",
    packages=["fiftyone.brain"],
    include_package_data=True,
    install_requires=[
        "numpy",
        "scipy",
    ],
    classifiers=[
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: POSIX :: Linux",
        "Programming Language :: Python :: 2.7",
        "Programming Language :: Python :: 3",
    ],
    scripts=[],
    python_requires=">=2.7",
    cmdclass=cmdclass,
)
