# FiftyOne Brain

The brains behind [FiftyOne](https://github.com/voxel51/fiftyone). For
documentation, [see here](https://docs.voxel51.com/brain.html).

## Repository layout

-   `fiftyone/brain/`: definition of the `fiftyone.brain` namespace
-   `requirements/`: Python requirements for the project
-   `tests/`: tests for the various components of the Brain

## Installation

The FiftyOne Brain is distributed via the `fiftyone-brain` package, and a
suitable version is automatically included with every `fiftyone` install!

```shell
pip install fiftyone
pip show fiftyone-brain
```

### Installing from source

If you wish to do a source install of the latest FiftyOne Brain version, simply
clone this repository:

```shell
git clone https://github.com/voxel51/fiftyone-brain
cd fiftyone-brain
```

and run the install script:

```shell
bash install.bash
```

### Developer installation

If you are a developer contributing to this repository, you should perform a
developer installation using the `-d` flag of the install script:

```shell
bash install.bash -d
```

Check out the [contribution guide](CONTRIBUTING.md) to get started.

## Uninstallation

```shell
pip uninstall fiftyone-brain
```

## Citation

If you use FiftyOne in your research, feel free to cite the project (but only
if you love it 😊):

```bibtex
@article{moore2020fiftyone,
  title={FiftyOne},
  author={Moore, B. E. and Corso, J. J.},
  journal={GitHub. Note: https://github.com/voxel51/fiftyone},
  year={2020}
}
```
