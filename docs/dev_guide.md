# FiftyOne Brain Developer's Guide

This document describes best practices for contributing to the `fiftyone-brain`
codebase.

We currently follow all of the practices established in
[ETA](https://github.com/voxel51/eta) for the Python code in this repository,
so this guide is mostly pointers to the relevant resources there!

## Coding style

-   [Python style guide](https://github.com/voxel51/eta/blob/develop/docs/python_style_guide.md)
-   [Linting guide](https://github.com/voxel51/eta/blob/develop/docs/linting_guide.md)
-   [Logging guide](https://github.com/voxel51/eta/blob/develop/docs/logging_guide.md)
-   [Markdown style guide](https://github.com/voxel51/eta/blob/develop/docs/markdown_style_guide.md)

## Documentation

All private or internal-facing documentation should be kept out of docstrings,
as these are visible to end-users of the package. Instead, this information can
be placed in `#`-style comments or additional triple-quoted string literals:

```python
def func():
    """Public-facing docs."""
    # a short private note
    ...


def func2():
    """Public-facing docs."""
    """
    Longer, even more private information.
    """
    ...
```

Be sure to keep a line break between the end of a docstring and the beginning
of another string literal to prevent them from being
[concatenated](https://docs.python.org/3/reference/lexical_analysis.html#string-literal-concatenation).

## Exposure of methods in top-level brain package

The `fiftyone.brain` package should expose all core user-functionality at the base level.  For example, for hardness, the user should be able to execute calls in the following way:
```
# this is good
import fiftyone.brain as fob
...
fob.compute_hardness(...)

# this is less good
import fiftyone.brain.hardness as fobh
...
fobh.compute_hardness(...)
```

So, in the `fiftyone.brain` package `__init__.py`, you should import core, public-facing methods:
```
from .hardness




## Copyright

Copyright 2017-2020, Voxel51, Inc.<br> voxel51.com
