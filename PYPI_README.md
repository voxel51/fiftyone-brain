<div align="center">

<h1>
    FiftyOne Brain
</h1>

**The brains behind
[FiftyOne](https://github.com/voxel51/fiftyone).**

<p align="center">
  <a href="https://voxel51.com/docs/fiftyone/user_guide/brain.html">Docs</a> •
  <a href="https://colab.research.google.com/github/voxel51/fiftyone-examples/blob/master/examples/quickstart.ipynb">Try it Now</a> •
  <a href="https://voxel51.com/docs/fiftyone/tutorials/index.html">Tutorials</a> •
  <a href="https://github.com/voxel51/fiftyone-examples">Examples</a>
</p>

[![PyPI python](https://img.shields.io/pypi/pyversions/fiftyone-brain)](https://pypi.org/project/fiftyone-brain)
[![PyPI version](https://badge.fury.io/py/fiftyone-brain.svg)](https://pypi.org/project/fiftyone-brain)
[![Slack](https://img.shields.io/badge/Slack-4A154B?logo=slack&logoColor=white)](https://join.slack.com/t/fiftyone-users/shared_invite/zt-gtpmm76o-9AjvzNPBOzevBySKzt02gg)
[![Medium](https://img.shields.io/badge/Medium-12100E?logo=medium&logoColor=white)](https://medium.com/voxel51)
[![Twitter](https://img.shields.io/badge/Twitter-1DA1F2?logo=twitter&logoColor=white)](https://twitter.com/voxel51)

<img src="https://user-images.githubusercontent.com/25985824/104953482-7ea97a00-5994-11eb-8cc3-f648c15502b1.png" alt="fiftyone-brain.png">

</div>

---

The [FiftyOne Brain](https://voxel51.com/docs/fiftyone/user_guide/brain.html)
provides powerful machine learning techniques that are designed to transform
how you curate your data from an art into a measurable science.

The FiftyOne Brain is a separate Python package that is bundled with
[FiftyOne](https://voxel51.com/docs/fiftyone). Although it is closed-source, it
is licensed as freeware, and you have permission to use it for commercial or
non-commercial purposes. See the license below for more details.

The FiftyOne Brain methods are useful across the stages of the machine learning
workflow:

-   **[Uniqueness](https://voxel51.com/docs/fiftyone/user_guide/brain.html#image-uniqueness):**
    During the training loop for a model, the best results will be seen when
    training on unique data. The FiftyOne Brain provides a uniqueness measure
    for images that compare the content of every image in a FiftyOne Dataset
    with all other images. Uniqueness operates on raw images and does not
    require any prior annotation on the data. It is hence very useful in the
    early stages of the machine learning workflow when you are likely asking
    "What data should I select to annotate?"

-   **[Mistakenness](https://voxel51.com/docs/fiftyone/user_guide/brain.html#label-mistakes):**
    Annotations mistakes create an artificial ceiling on the performance of
    your models. However, finding these mistakes by hand is at least as arduous
    as the original annotation was, especially in cases of larger datasets. The
    FiftyOne Brain provides a quantitative mistakenness measure to identify
    possible label mistakes. Mistakenness operates on labeled images and
    requires the logit-output of your model predictions in order to provide
    maximum efficacy. It also works on detection datasets to find missed
    objects, incorrect annotations, and localization issues.

-   **[Hardness](https://voxel51.com/docs/fiftyone/user_guide/brain.html#sample-hardness):**
    While a model is training, it will learn to understand attributes of
    certain samples faster than others. The FiftyOne Brain provides a hardness
    measure that calculates how easy or difficult it is for your model to
    understand any given sample. Mining hard samples is a tried and true
    measure of mature machine learning processes. Use your current model
    instance to compute predictions on unlabeled samples to determine which are
    the most valuable to have annotated and fed back into the system as
    training samples, for example.
