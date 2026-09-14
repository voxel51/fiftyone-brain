"""
Unit tests for the LanceDB similarity backend.

These tests do not require a LanceDB store or object storage, so unlike the
LanceDB tests in ``tests/intensive/``, they run in CI.

| Copyright 2017-2026, Voxel51, Inc.
| `voxel51.com <https://voxel51.com/>`_
|
"""
from unittest import mock

import pytest

import fiftyone.brain as fb
from fiftyone.brain.internal.core.lancedb import LanceDBSimilarityConfig


@pytest.fixture(name="empty_backend_config")
def fixture_empty_backend_config():
    """Isolates tests from whatever the ambient brain config carries."""
    with mock.patch.object(fb.brain_config, "similarity_backends", {}):
        yield


class TestStorageOptions:
    """The storage options that carry a credential to the object store."""

    def test_defaults_to_none(self):
        config = LanceDBSimilarityConfig()
        assert config.storage_options is None

    def test_accepted_by_the_constructor(self):
        options = {"aws_access_key_id": "key", "aws_region": "us-east-1"}
        config = LanceDBSimilarityConfig(storage_options=options)
        assert config.storage_options == options

    def test_settable_after_construction(self):
        config = LanceDBSimilarityConfig()
        config.storage_options = {"bearer_token": "token"}
        assert config.storage_options == {"bearer_token": "token"}

    @pytest.mark.usefixtures("empty_backend_config")
    def test_load_credentials_supplies_them(self):
        config = LanceDBSimilarityConfig()
        config.load_credentials(storage_options={"aws_session_token": "token"})
        assert config.storage_options == {"aws_session_token": "token"}

    @pytest.mark.usefixtures("empty_backend_config")
    def test_load_credentials_leaves_existing_options_alone(self):
        # ``None`` means "not supplied", so a caller that only refreshes the
        # URI must not blank out the credential it is still using
        config = LanceDBSimilarityConfig(storage_options={"key": "value"})
        config.load_credentials(uri="/tmp/lancedb")
        assert config.storage_options == {"key": "value"}

    def test_load_credentials_falls_back_to_the_brain_config(self):
        # How a deployment injects a credential it never wants written to a
        # dataset: configure the backend, leave the call site bare
        options = {"google_service_account": "/var/run/sa.json"}
        with mock.patch.object(
            fb.brain_config,
            "similarity_backends",
            {"lancedb": {"storage_options": options}},
        ):
            config = LanceDBSimilarityConfig()
            config.load_credentials()

        assert config.storage_options == options


class TestSerialization:
    """What reaches the database when a brain run is saved."""

    @pytest.mark.parametrize(
        "field,value",
        [
            pytest.param(
                "storage_options",
                {"aws_secret_access_key": "SENSITIVE"},
                id="storage_options",
            ),
            pytest.param("uri", "s3://bucket/org-abc/dataset-1", id="uri"),
        ],
    )
    def test_credentials_are_not_serialized(self, field, value):
        # These are stored privately so a vended credential never lands in
        # the dataset's brain document
        config = LanceDBSimilarityConfig(**{field: value})
        serialized = config.serialize()

        assert field not in serialized
        assert "SENSITIVE" not in str(serialized)

    def test_the_rest_of_the_config_still_serializes(self):
        config = LanceDBSimilarityConfig(
            table_name="a-table",
            metric="euclidean",
            storage_options={"aws_access_key_id": "key"},
        )
        serialized = config.serialize()

        assert serialized["table_name"] == "a-table"
        assert serialized["metric"] == "euclidean"
