"""
Tests that run LanceDB itself rather than a stand-in.

Real connections, and a database in a temporary directory where a test
needs one, because what is under test is what LanceDB does: how it pages a
listing, what it raises for a table that is not there, and what a
connection is opened with. A fake would establish none of it, so the whole
module skips where the optional extra is absent. Configuration, which
needs neither, is covered in ``test_lancedb.py``.

| Copyright 2017-2026, Voxel51, Inc.
| `voxel51.com <https://voxel51.com/>`_
|
"""

import os
import types

import pyarrow as pa
import pytest

import fiftyone.brain.internal.core.lancedb as foblancedb

lancedb = pytest.importorskip("lancedb")

#: More tables than a single default page holds, so a listing that stops at
#: the first page is visibly short. Kept as low as that allows: writing a
#: table costs a directory and four files, which is slow enough on a CI disk
#: to dominate the suite if this is built per test.
TABLE_COUNT = 12


@pytest.fixture(name="database", scope="module")
def fixture_database(tmp_path_factory):
    """Yields a database holding :data:`TABLE_COUNT` single-row tables.

    Built once for the module: every test that takes it only reads, and a
    test that needs to write builds its own.
    """
    path = tmp_path_factory.mktemp("listing")
    database = lancedb.connect(os.path.join(str(path), "db"))
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

    @pytest.mark.parametrize("size", [1, 2, 3, 5, 11, 12, 100])
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

    def test_a_table_that_cannot_be_read_is_raised(self, tmp_path):
        # Absence and unreadability are not distinguished by type here, so
        # swallowing everything would let a table this process cannot read
        # pass for an index holding nothing.
        #
        # Its own database, since corrupting a table writes to one.
        root = os.path.join(str(tmp_path), "db")
        database = lancedb.connect(root)
        database.create_table("broken", pa.table({"x": [1]}))
        versions = os.path.join(root, "broken.lance", "_versions")
        for name in os.listdir(versions):
            with open(os.path.join(versions, name), "wb") as handle:
                handle.write(b"not a manifest")

        with pytest.raises(Exception) as raised:
            foblancedb._open_table(database, "broken")

        assert "was not found" not in str(raised.value)


class TestConnect:
    """What reaches the LanceDB connection."""

    def _connect(self, config):
        # `_initialize` reads only the config when the table name is set, so
        # a full index -- which needs a dataset -- is not required to observe
        # what the connection was opened with
        index = types.SimpleNamespace(config=config)
        foblancedb.LanceDBSimilarityIndex._initialize(index)

        return index._db

    def test_storage_options_reach_the_connection(self, tmp_path):
        options = {"timeout": "30s"}
        config = foblancedb.LanceDBSimilarityConfig(
            table_name="a-table",
            uri=str(tmp_path),
            storage_options=options,
        )

        connection = self._connect(config)

        assert connection.storage_options == options

    @pytest.mark.parametrize(
        "options",
        [pytest.param(None, id="none"), pytest.param({}, id="empty")],
    )
    def test_the_argument_is_omitted_when_unset(self, tmp_path, options):
        # No minimum lancedb version is declared, so a release without the
        # parameter must still work for callers that set no options, which
        # means passing none rather than an empty dict
        config = foblancedb.LanceDBSimilarityConfig(
            table_name="a-table", uri=str(tmp_path), storage_options=options
        )

        connection = self._connect(config)

        assert connection.storage_options is None

    def test_the_run_s_own_store_is_the_one_opened(self, tmp_path):
        config = foblancedb.LanceDBSimilarityConfig(
            table_name="a-table", uri=str(tmp_path)
        )

        assert self._connect(config).uri == str(tmp_path)
