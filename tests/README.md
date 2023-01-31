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

To run all tests in this directory, execute:

```shell
pytest . -s
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

Copyright 2017-2023, Voxel51, Inc.<br> voxel51.com
