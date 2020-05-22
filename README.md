# FiftyOne Brain

The proprietary brains behind [FiftyOne](https://github.com/voxel51/fiftyone).

<img src="https://user-images.githubusercontent.com/3719547/74191434-8fe4f500-4c21-11ea-8d73-555edfce0854.png" alt="voxel51-logo.png" width="40%"/>

## Repository Layout

-   `./docs` documentation about the repository and project
-   `./experiments` contains internal-only examples in the form of specific
    experiments about FiftyOne's value.
-   `./fiftyone` is the actual `fiftyone.brain` python code
-   `./production` contains the work needed to generate production models,
    code, etc., associated with the brain. It is contained scripts and such for
    repeatable, versioned model training for uniqueness, for example.
-   `./requirements` is the standard requirements folder for the python project

## Installation

Clone the repository:

```shell
git clone https://github.com/voxel51/fiftyone-brain
cd fiftyone-brain
```

and install it:

```shell
bash install.bash
```

We strongly recommend that you install in a
[virtual environment](https://virtualenv.pypa.io/en/stable) to maintain a clean
workspace.

### Developer installation

If you are a developer contributing to this repository, you should perform a
developer installation using the `-d` flag of the install script:

```shell
bash install.bash -d
```

You should also checkout the
[Developer's Guide](https://github.com/voxel51/fiftyone-brain/blob/develop/docs/dev_guide.md)
to get started.

## Uninstallation

```shell
pip uninstall fiftyone-brain
```

## Copyright

Copyright 2017-2020, Voxel51, Inc.<br> voxel51.com
