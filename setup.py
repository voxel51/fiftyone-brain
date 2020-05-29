#!/usr/bin/env python
"""
Installs `fiftyone-brain`.

| Copyright 2017-2020, Voxel51, Inc.
| `voxel51.com <https://voxel51.com/>`_
|
"""
import os

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
        for root, dirs, files in os.walk(brain_dir):
            for filename in files:
                if (
                    os.path.splitext(filename)[-1].lower().startswith(".py")
                    or "pytransform" in root
                ):
                    full_path = os.path.join(root, filename)
                    print("Removing", full_path)
                    os.remove(full_path)

        call_pyarmor(
            [
                "obfuscate",
                "--recursive",
                "--output",
                brain_dir,
                "--platform",
                self.pyarmor_platform,
                os.path.join("fiftyone", "brain", "__init__.py"),
            ]
        )


class CustomBdistWheel(bdist_wheel):
    def finalize_options(self):
        bdist_wheel.finalize_options(self)
        # not pure Python - pytransform shared lib from pyarmor is OS-dependent
        self.root_is_pure = False

        # rewrite platform names - we currently only support 64-bit targets
        if self.plat_name.startswith("linux"):
            self.plat_name = "linux_x86_64"
            pyarmor_platform = "linux.x86_64"
        elif self.plat_name.startswith("mac"):
            # unclear what minimum version PyArmor requires, but the .dylib was
            # built on 10.11 per https://pyarmor.readthedocs.io/en/latest/platforms.html#platform-tables
            self.plat_name = "macosx_10_11_x86_64"
            pyarmor_platform = "darwin.x86_64"
        elif self.plat_name.startswith("win"):
            self.plat_name = "win_amd64"
            pyarmor_platform = "windows.x86_64"
        else:
            raise ValueError(
                "Unsupported target platform: %r" % self.plat_name
            )

        # pass to "build" command instance
        build = self.reinitialize_command("build")
        build.pyarmor_platform = pyarmor_platform


setup(
    name="fiftyone-brain",
    version="0.1.2",
    description="FiftyOne Brain",
    author="Voxel51, Inc.",
    author_email="info@voxel51.com",
    url="https://github.com/voxel51/fiftyone-brain",
    license="",
    packages=["fiftyone.brain"],
    include_package_data=True,
    install_requires=["numpy", "scipy", "scikit-learn"],
    classifiers=[
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: POSIX :: Linux",
        "Programming Language :: Python :: 2.7",
        "Programming Language :: Python :: 3",
    ],
    scripts=[],
    python_requires=">=2.7",
    cmdclass={"build": CustomBuild, "bdist_wheel": CustomBdistWheel,},
)
