# FiftyOne Brain

The proprietary brains behind [FiftyOne](https://github.com/voxel51/fiftyone).

<img src="https://user-images.githubusercontent.com/3719547/74191434-8fe4f500-4c21-11ea-8d73-555edfce0854.png" alt="voxel51-logo.png" width="40%"/>

## Repository Layout

-   `docs/` documentation about the repository and project

-   `experiments/` internal-only examples that demonstrate concrete value-add
    of the FiftyOne Brain

-   `fiftyone/brain/` definition of the `fiftyone.brain` package

    -   `fiftyone/brain/internal/` all propreitary internal code powering the
        public namespace of the Brain

-   `production/` work needed to generate production models, code, etc.,
    associated with the Brain. It contains scripts and such for repeatable,
    versioned model training for uniqueness, for example

-   `requirements/` Python requirements for the project

-   `tests/` tests for the various components of the Brain

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
