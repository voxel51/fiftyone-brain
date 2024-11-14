<div align="center">
<p align="center">

<img src="https://github.com/user-attachments/assets/17afdf93-289c-40f1-805c-06344f095cf6" height="55px">

**Open Source AI from [Voxel51](https://voxel51.com)**

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

FiftyOne Brain contains the open source AI/ML capabilities for the
[FiftyOne ecosystem](https://github.com/voxel51/fiftyone), enabling users to
automatically analyze and manipulate their datasets and models. FiftyOne Brain
includes features like visual similarity search, query by text, finding unique
and representative samples, finding media quality problems and annotation
mistakes, and more 🚀

## Documentation

Public documentation for the FiftyOne Brain is
[available here](https://docs.voxel51.com/user_guide/brain.html).

## Installation

The FiftyOne Brain is distributed via the `fiftyone-brain` package, and a
suitable version is automatically included with every `fiftyone` install:

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
# Mac or Linux
bash install.bash

# Windows
.\install.bat
```

### Developer installation

If you are a developer contributing to this repository, you should perform a
developer installation using the `-d` flag of the install script:

```shell
# Mac or Linux
bash install.bash -d

# Windows
.\install.bat -d
```

Check out the [contribution guide](CONTRIBUTING.md) to get started.

## Uninstallation

```shell
pip uninstall fiftyone-brain
```

## Repository layout

-   `fiftyone/brain/` definition of the `fiftyone.brain` namespace
-   `requirements/` Python requirements for the project
-   `tests/` tests for the various components of the Brain

## Citation

If you use the FiftyOne Brain in your research, please cite the project:

```bibtex
@article{moore2020fiftyone,
  title={FiftyOne},
  author={Moore, B. E. and Corso, J. J.},
  journal={GitHub. Note: https://github.com/voxel51/fiftyone-brain},
  year={2020}
}
```
