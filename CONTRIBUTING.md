# Contributing to FiftyOne Brain

All Brain contributions should follow the practices established in
[FiftyOne](https://github.com/voxel51/fiftyone/blob/develop/CONTRIBUTING.md).

## Adding new public methods to the Brain package

The `fiftyone.brain` package should expose all core user-functionality at the
base level. For example, for hardness, the user should be able to execute calls
in the following way:

```py
# Users should be able to do this
import fiftyone.brain as fob

fob.compute_hardness(...)

# And NOT have to do this
import fiftyone.brain.hardness as fobh

fobh.compute_hardness(...)
```

To achieve this, follow the existing pattern of declaring new public methods in
[`fiftyone/brain/__init__.py`](https://github.com/voxel51/fiftyone-brain/blob/develop/fiftyone/brain/__init__.py).

Be sure to include a detailed docstring for all methods in this file, as they
are pulled in by FiftyOne documentation builds and are made available in the
[public docs](https://docs.voxel51.com/api/fiftyone.brain.html).
