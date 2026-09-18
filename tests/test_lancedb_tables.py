"""
Tests for how the LanceDB backend finds its tables.

Against a real database in a temporary directory rather than a stand-in,
because what is under test is how LanceDB pages a listing and what it
raises for a table that is not there -- neither of which a fake would
establish.

| Copyright 2017-2026, Voxel51, Inc.
| `voxel51.com <https://voxel51.com/>`_
|
"""

import os
import shutil

import pyarrow as pa
import pytest

import fiftyone.brain.internal.core.lancedb as foblancedb

lancedb = pytest.importorskip("lancedb")

#: More tables than a single default page holds, so a listing that stops at
#: the first page is visibly short.
TABLE_COUNT = 26


@pytest.fixture(name="database")
def fixture_database(tmp_path):
    """Yields a database holding :data:`TABLE_COUNT` single-row tables."""
    database = lancedb.connect(os.path.join(str(tmp_path), "db"))
    for index in range(TABLE_COUNT):
        database.create_table(f"table{index:02d}", pa.table({"x": [1]}))

    yield database


@pytest.fixture(name="page_size")
def fixture_page_size():
    """Sets the module's page size, and restores it afterwards."""
    original = foblancedb._DB_TABLE_PG_LIMIT

    yield lambda size: setattr(foblancedb, "_DB_TABLE_PG_LIMIT", size)

    foblancedb._DB_TABLE_PG_LIMIT = original


class TestTableNames:
    """Reading the whole table listing."""

    @pytest.mark.parametrize("size", [1, 2, 3, 7, 25, 26, 100])
    def test_every_table_is_listed_once(self, database, page_size, size):
        # `list_tables` caps a page, so a listing that stops at the first
        # page misses tables. Paged on the wrong cursor it also repeats and
        # drops them, and only a page size covering every table hides that.
        page_size(size)
        expected = sorted(database.list_tables(limit=TABLE_COUNT * 2).tables)

        listed = foblancedb._table_names(database)

        assert sorted(listed) == expected
        assert len(listed) == len(set(listed))

    def test_the_default_listing_is_short(self, database):
        # The reason any of this exists: the unpaged call answers with a
        # page rather than with every table.
        assert len(database.table_names()) < TABLE_COUNT
        assert len(foblancedb._table_names(database)) == TABLE_COUNT

    def test_an_empty_database_lists_nothing(self, tmp_path):
        empty = lancedb.connect(os.path.join(str(tmp_path), "empty"))

        assert foblancedb._table_names(empty) == []


class TestOpenTable:
    """Asking for one table rather than listing every one to find it."""

    def test_a_table_that_is_there_comes_back(self, database):
        opened = foblancedb._open_table(database, "table00")

        assert opened is not None
        assert len(opened) == 1

    def test_a_table_that_is_not_there_is_not_an_error(self, database):
        # A run whose table has yet to be written is the ordinary case, and
        # reads as an index holding nothing.
        assert foblancedb._open_table(database, "table99") is None

    def test_a_table_that_cannot_be_read_is_raised(self, database, tmp_path):
        # Absence and unreadability are not distinguished by type here, so
        # swallowing everything would let a table this process cannot read
        # pass for an index holding nothing.
        broken = os.path.join(str(tmp_path), "db", "broken.lance")
        shutil.copytree(
            os.path.join(str(tmp_path), "db", "table00.lance"), broken
        )
        versions = os.path.join(broken, "_versions")
        for name in os.listdir(versions):
            with open(os.path.join(versions, name), "wb") as handle:
                handle.write(b"not a manifest")

        with pytest.raises(Exception) as raised:
            foblancedb._open_table(database, "broken")

        assert "was not found" not in str(raised.value)
