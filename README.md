<div align="center">
<p align="center">

**FiftyOne Brain**

**Open Source AI from @Voxel51**

<!-- prettier-ignore -->
<a href="https://voxel51.com/fiftyone">FiftyOne Website</a> •
<a href="https://voxel51.com/docs/fiftyone">FiftyOne Docs</a> •
<a href="https://docs.voxel51.com/user_guide_brain.html">FiftyOne Brain Docs</a> •
<a href="https://voxel51.com/blog/">Blog</a> •
<a href="https://slack.voxel51.com">Community</a>

[![PyPI python](https://img.shields.io/pypi/pyversions/fiftyone-brain)](https://pypi.org/project/fiftyone-brain)
[![PyPI version](https://badge.fury.io/py/fiftyone.svg)](https://pypi.org/project/fiftyone-brain)
[![Downloads](https://static.pepy.tech/badge/fiftyone-brain)](https://pepy.tech/project/fiftyone-brain)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Slack](https://img.shields.io/badge/Slack-4A154B?logo=slack&logoColor=white)](https://slack.voxel51.com)
[![Medium](https://img.shields.io/badge/Medium-12100E?logo=medium&logoColor=white)](https://medium.com/voxel51)
[![Mailing list](http://bit.ly/2Md9rxM)](https://share.hsforms.com/1zpJ60ggaQtOoVeBqIZdaaA2ykyk)
[![Twitter](https://img.shields.io/twitter/follow/Voxel51?style=social)](https://twitter.com/voxel51)

</p>
</div>

---

The open source AI/ML capabilities for the
[FiftyOne](https://github.com/voxel51/fiftyone) ecosystem, implementing the
AIML capabilities enabling users to automatically analyze and manipulate
datasets and models in smart and intuitive ways. The FiftyOne Brain includes
features like visual similarity search, query by text, finding unique and
representative samples, finding media quality problems and annotation mistakes,
and more.

## Repository Layout

-   `fiftyone/brain/` definition of the `fiftyone.brain` namespace

-   `requirements/` Python requirements for the project

-   `tests/` tests for the various components of the Brain

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

## Documentation

The main documentation for the FiftyOne Brain is available at
[fiftyone.ai](https://docs.voxel51.com/user_guide_brain.html). This includes
various types of AI/ML functionality like

-   Working with embeddings
-   Computing and working with visual similarity
-   Dataset and sample insights like uniqueness and hardness
-   Configuring and managing brain runs

## Citation

If you use the FiftyOne Brain in your research, please cite the project.

```bibtex
@article{moore2020fiftyone,
  title={FiftyOne},
  author={Moore, B. E. and Corso, J. J.},
  journal={GitHub. Note: https://github.com/voxel51/fiftyone-brain},
  year={2020}
}
```
