# FiftyOne Brain Developer's Guide

This document describes best practices for contributing to the `fiftyone-brain`
codebase.

We currently follow all of the practices established in
[ETA](https://github.com/voxel51/eta) for the Python code in this repository,
so this guide is mostly pointers to the relevant resources there!

> Happy exception: FiftyOne Brain is strictly Python 3 code, so we do not
> follow the Python 2 compability instructions in ETA!

## Coding style

-   [Python style guide](https://github.com/voxel51/eta/blob/develop/docs/python_style_guide.md)
-   [Linting guide](https://github.com/voxel51/eta/blob/develop/docs/linting_guide.md)
-   [Logging guide](https://github.com/voxel51/eta/blob/develop/docs/logging_guide.md)
-   [Markdown style guide](https://github.com/voxel51/eta/blob/develop/docs/markdown_style_guide.md)

## Documentation

All private or internal-facing documentation in the **public namespace** of the
`fiftyone.brain` package (i.e., `fiftyone.brain.*`) should be kept out of
docstrings, as these are visible to end-users of the package. Instead, this
information can be placed in `#`-style comments or additional triple-quoted
string literals:

```python
def func():
    """Public-facing docs."""
    # A short private note
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

Copyright 2017-2023, Voxel51, Inc.<br> voxel51.com
