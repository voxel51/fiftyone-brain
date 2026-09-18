"""
Tests for how the LanceDB backend finds its tables.

Both functions under test take the database handle as an argument, so a
stub stands in for it and no LanceDB install is needed -- which is what
lets these run in CI, where the package is absent.

| Copyright 2017-2026, Voxel51, Inc.
| `voxel51.com <https://voxel51.com/>`_
|
"""

import types

import pytest

import fiftyone.brain.internal.core.lancedb as foblancedb

NAMES = [f"table{i:02d}" for i in range(26)]


class StubDatabase:
    """A database that pages its table listing the way LanceDB does.

    The cursor it hands back is a storage key rather than a table name, and
    it is exclusive: a page begins after the name the cursor was taken from.
    The last page carries no cursor.
    """

    def __init__(self, names):
        self.names = list(names)
        self.pages = 0

    def list_tables(self, page_token=None, limit=None):
        self.pages += 1

        if page_token is None:
            start = 0
        else:
            start = self.names.index(page_token.removesuffix(".lance/")) + 1

        tables = self.names[start : start + limit]
        exhausted = start + len(tables) >= len(self.names)

        return types.SimpleNamespace(
            tables=tables,
            page_token=None if exhausted else f"{tables[-1]}.lance/",
        )


class RefusingDatabase:
    """A database whose ``open_table`` raises."""

    def __init__(self, error):
        self.error = error

    def open_table(self, table_name):
        raise self.error


@pytest.fixture(name="page_size")
def fixture_page_size():
    """Restores the module's page size after a test changes it."""
    original = foblancedb._DB_TABLE_PG_LIMIT

    yield lambda size: setattr(foblancedb, "_DB_TABLE_PG_LIMIT", size)

    foblancedb._DB_TABLE_PG_LIMIT = original


class TestTableNames:
    """Reading the whole table listing."""

    @pytest.mark.parametrize("size", [1, 2, 3, 7, 25, 26, 100])
    def test_every_table_is_listed_once(self, page_size, size):
        # `list_tables` caps a page, so a listing that stops at the first
        # page misses tables. Paged wrongly it can also repeat or drop them,
        # and only a page size that covers everything hides that.
        page_size(size)
        database = StubDatabase(NAMES)

        listed = foblancedb._table_names(database)

        assert listed == NAMES
        assert len(listed) == len(set(listed))

    def test_a_page_is_asked_for_at_the_configured_size(self, page_size):
        page_size(10)
        database = StubDatabase(NAMES)

        foblancedb._table_names(database)

        assert database.pages == 3

    @pytest.mark.usefixtures("page_size")
    def test_an_empty_database_lists_nothing(self):
        assert foblancedb._table_names(StubDatabase([])) == []


class TestOpenTable:
    """Asking for one table rather than listing every one to find it."""

    def test_a_table_that_is_there_comes_back(self):
        database = types.SimpleNamespace(open_table=lambda name: "the table")

        assert foblancedb._open_table(database, "present") == "the table"

    def test_a_table_that_is_not_there_is_not_an_error(self):
        # A run whose table has yet to be written is the ordinary case, and
        # reads as an index holding nothing.
        database = RefusingDatabase(ValueError("Table 'gone' was not found"))

        assert foblancedb._open_table(database, "gone") is None

    @pytest.mark.parametrize(
        "error",
        [
            pytest.param(
                ValueError("Invalid input, Failed to connect to namespace"),
                id="unreachable_store",
            ),
            pytest.param(
                RuntimeError("lance error: LanceError(IO)"), id="corrupted"
            ),
        ],
    )
    def test_any_other_refusal_is_raised(self, error):
        # Absence and unreachability share a type here, so swallowing
        # everything would let a store this process cannot read read as an
        # index holding nothing.
        database = RefusingDatabase(error)

        with pytest.raises(type(error)):
            foblancedb._open_table(database, "present")
