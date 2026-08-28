"""
Unit tests for the pgvector similarity backend.

These tests do not require a running PostgreSQL instance, so unlike the
pgvector tests in ``tests/intensive/``, they run in CI. The integration tests
live in ``tests/intensive/test_similarity.py``.

| Copyright 2017-2026, Voxel51, Inc.
| `voxel51.com <https://voxel51.com/>`_
|
"""
import pytest

from fiftyone.brain.internal.core.pgvector import (
    PgVectorSimilarityConfig,
    _default_ivfflat_lists,
    _default_ivfflat_probes,
    _parse_reloption_lists,
)


class TestIvfflatListDefaults:
    """Deriving the IVFFlat list count from a row count."""

    def test_small_tables_resolve_to_one_list(self):
        # A list count approaching the row count leaves lists empty or
        # singleton, which destroys recall. One list means exact search
        for num_rows in (0, 1, 78, 500, 999):
            assert _default_ivfflat_lists(num_rows) == 1

    def test_rows_over_1000_up_to_1m(self):
        assert _default_ivfflat_lists(1000) == 1
        assert _default_ivfflat_lists(10000) == 10
        assert _default_ivfflat_lists(50000) == 50
        assert _default_ivfflat_lists(1000000) == 1000

    def test_sqrt_rows_above_1m(self):
        assert _default_ivfflat_lists(4000000) == 2000
        assert _default_ivfflat_lists(9000000) == 3000

    def test_lists_never_exceed_rows(self):
        for num_rows in (1, 10, 100, 1000, 10000, 10**6, 10**7):
            assert _default_ivfflat_lists(num_rows) <= num_rows

    def test_degenerate_inputs(self):
        assert _default_ivfflat_lists(None) == 1
        assert _default_ivfflat_lists(-5) == 1


class TestIvfflatProbeDefaults:
    """Deriving the IVFFlat probe count from a list count."""

    def test_probes_follow_sqrt_lists(self):
        assert _default_ivfflat_probes(1) == 1
        assert _default_ivfflat_probes(100) == 10
        assert _default_ivfflat_probes(1000) == 31

    def test_probes_never_exceed_lists(self):
        for lists in (1, 2, 3, 10, 100, 1000, 32768):
            assert _default_ivfflat_probes(lists) <= lists

    def test_degenerate_inputs(self):
        assert _default_ivfflat_probes(None) == 1
        assert _default_ivfflat_probes(0) == 1


class TestIvfflatConfigDefaults:
    """Config-level defaults for the IVFFlat parameters."""

    def test_ivfflat_params_default_to_auto(self):
        config = PgVectorSimilarityConfig(
            index_type="ivfflat", metric="euclidean"
        )

        # None means "derive from the data at index build time"
        assert config.ivfflat_lists is None
        assert config.ivfflat_probes is None

    def test_explicit_ivfflat_params_are_preserved(self):
        config = PgVectorSimilarityConfig(
            index_type="ivfflat",
            metric="euclidean",
            ivfflat_lists=10,
            ivfflat_probes=5,
        )

        assert config.ivfflat_lists == 10
        assert config.ivfflat_probes == 5


class TestParseReloptionLists:
    """Reading the built list count out of ``pg_class.reloptions``."""

    def test_extracts_lists(self):
        assert _parse_reloption_lists(["lists=100"]) == 100
        assert _parse_reloption_lists(["lists=7"]) == 7

    def test_ignores_other_options(self):
        assert _parse_reloption_lists(["fillfactor=90", "lists=42"]) == 42

    def test_absent_or_empty(self):
        assert _parse_reloption_lists(None) is None
        assert _parse_reloption_lists([]) is None
        assert _parse_reloption_lists(["fillfactor=90"]) is None

    def test_unparseable_value(self):
        assert _parse_reloption_lists(["lists=abc"]) is None

    def test_does_not_match_prefixed_keys(self):
        assert _parse_reloption_lists(["nolists=5"]) is None
