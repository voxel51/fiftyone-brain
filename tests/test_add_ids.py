"""
fiftyone.brain.internal.core.utils.add_ids() tests.

| Copyright 2017-2026, Voxel51, Inc.
| `voxel51.com <https://voxel51.com/>`_
|
"""

import logging

import numpy as np
import pytest

import fiftyone.brain.internal.core.utils as fbu

INDEX_IDS = np.array(["a", "b", "c"])


def _add(ids, **kwargs):
    return fbu.add_ids(np.array(ids), None, INDEX_IDS.copy(), None, **kwargs)


def test_overwrite_replaces_existing_ids():
    index_ids, _, ii, jj = _add(["b", "d"], overwrite=True)

    assert list(index_ids) == ["a", "b", "c", "d"]
    assert list(ii) == [0, 1]
    assert list(jj) == [1, 3]


@pytest.mark.parametrize("warn_existing", [False, True])
def test_no_overwrite_skips_existing_ids(warn_existing):
    # Whether existing IDs are skipped must not depend on whether a warning
    # is requested
    index_ids, _, ii, jj = _add(
        ["b", "d"], overwrite=False, warn_existing=warn_existing
    )

    assert list(index_ids) == ["a", "b", "c", "d"]
    assert list(ii) == [1]
    assert list(jj) == [3]


def test_no_overwrite_only_existing_ids_is_a_no_op():
    index_ids, _, ii, jj = _add(["a", "c"], overwrite=False)

    assert list(index_ids) == ["a", "b", "c"]
    assert ii.size == 0
    assert jj.size == 0


def test_no_overwrite_warning_names_an_existing_id(caplog):
    with caplog.at_level(logging.WARNING):
        _add(["d", "b"], overwrite=False, warn_existing=True)

    assert "Ignoring 1 IDs (eg 'b')" in caplog.text


def test_disallow_existing_error_names_an_existing_id():
    with pytest.raises(ValueError, match="eg 'b'"):
        _add(["d", "b"], overwrite=False, allow_existing=False)
