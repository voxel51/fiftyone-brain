# Contributing to FiftyOne Brain
For all contributions we currently follow all of the practices established in
[FiftyOne](https://github.com/voxel51/fiftyone/blob/develop/CONTRIBUTING.md). 

## Exposure of methods in top-level brain package

The `fiftyone.brain` package should expose all core user-functionality at the
base level. For example, for hardness, the user should be able to execute calls
in the following way:

```py
# Users should be able to do this
import fiftyone.brain as fob
...
fob.compute_hardness(...)

# And not have to do this
import fiftyone.brain.hardness as fobh
...
fobh.compute_hardness(...)
```

So, in `fiftyone/brain/__init__.py`, you should import core, public-facing
methods:

```py
from .hardness import compute_hardness
```

## Copyright

Copyright 2017-2024, Voxel51, Inc.<br> voxel51.com
