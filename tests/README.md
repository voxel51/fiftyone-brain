# FiftyOne-Brain Tests

The brain currently uses both
[unittest](https://docs.python.org/3/library/unittest.html) and
[pytest](https://docs.pytest.org/en/stable) to implement its tests.

## Contents

| File                 | Description                                              |
| -------------------- | -------------------------------------------------------- |
| `test_uniqueness.py` | Tests of the uniqueness capability                       |
| `models/*.py`        | Tests of the various models used by the brain            |
| `intensive/*.py`     | Intensive tests that are not included in automated tests |

## Running tests

The tests require the core `fiftyone` package, which is not installed by
`install.sh`. See the
[developer installation](../README.md#developer-installation) instructions for
how to set it up alongside your source install of the Brain.

Some tests also need optional packages. Without them, those tests either skip
or fail, so to run the same set of tests as CI, install them too:

```shell
pip install imageio pytest torch torchvision lancedb
```

To run all automated tests in this directory, as CI does, execute:

```shell
pytest . -s --ignore intensive/
```

The tests create datasets in the database that FiftyOne is configured to use.
If other environments with a different `fiftyone` version share that database,
tests may fail with migration errors, so use a separate database instead:

```shell
FIFTYONE_DATABASE_NAME=fiftyone-brain-tests pytest . -s --ignore intensive/
```

The intensive tests must be run separately, as some of their module names clash
with those of the automated tests:

```shell
pytest intensive/ -s
```

To run a specific set of tests, execute:

```shell
pytest <file>.py -s
```

To run a specific test case, execute:

```shell
pytest <file>.py -s -k <test_function_name>
```

## Copyright

Copyright 2017-2026, Voxel51, Inc.<br> voxel51.com
