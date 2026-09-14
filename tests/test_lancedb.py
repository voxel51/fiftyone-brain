"""
Unit tests for the LanceDB similarity backend.

These tests do not require a LanceDB store or object storage, so unlike the
LanceDB tests in ``tests/intensive/``, they run in CI. The integration tests
live in ``tests/intensive/test_similarity.py``.

| Copyright 2017-2026, Voxel51, Inc.
| `voxel51.com <https://voxel51.com/>`_
|
"""
import types
from unittest import mock

import pytest

import fiftyone.brain as fob
from fiftyone.brain.internal.core import lancedb as foblancedb
from fiftyone.brain.internal.core.lancedb import (
    LanceDBSimilarityConfig,
    LanceDBSimilarityIndex,
)


@pytest.fixture(name="empty_backend_config")
def fixture_empty_backend_config():
    """Isolates tests from whatever the ambient brain config carries."""
    with mock.patch.object(fob.brain_config, "similarity_backends", {}):
        yield


class TestStorageOptions:
    """The storage options that carry a credential to the object store."""

    def test_defaults_to_none(self):
        config = LanceDBSimilarityConfig()
        assert config.storage_options is None

    @pytest.mark.parametrize(
        "supply",
        [
            pytest.param(
                lambda options: LanceDBSimilarityConfig(
                    storage_options=options
                ),
                id="constructor",
            ),
            pytest.param(
                lambda options: _assign(
                    LanceDBSimilarityConfig(), "storage_options", options
                ),
                id="setter",
            ),
        ],
    )
    def test_lands_on_the_private_attribute(self, supply):
        # The base config setattrs any unrecognized keyword, so reading the
        # value back proves nothing. What the field adds is that it is stored
        # privately, which is what keeps it out of `serialize()`
        options = {"aws_access_key_id": "key", "aws_region": "us-east-1"}
        config = supply(options)

        assert config._storage_options == options
        assert "storage_options" not in vars(config)

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
        # The backend entry in ``~/.fiftyone/brain_config.json`` supplies the
        # options when the call site passes none. Note that file is plaintext
        # and `fiftyone brain config` prints it, so it is not a secret store
        options = {"google_service_account": "/var/run/sa.json"}
        with mock.patch.object(
            fob.brain_config,
            "similarity_backends",
            {"lancedb": {"storage_options": options}},
        ):
            config = LanceDBSimilarityConfig()
            config.load_credentials()

        assert config.storage_options == options

    def test_the_backend_config_dict_is_not_shared(self):
        # `_load_parameters` hands over the global dict itself. Aliasing it
        # would let one index's credential refresh rewrite every other
        # index's credential in the process
        options = {"aws_session_token": "first"}
        with mock.patch.object(
            fob.brain_config,
            "similarity_backends",
            {"lancedb": {"storage_options": options}},
        ):
            config = LanceDBSimilarityConfig()
            config.load_credentials()
            other = LanceDBSimilarityConfig()
            other.load_credentials()

            config.storage_options["aws_session_token"] = "refreshed"

            assert other.storage_options == {"aws_session_token": "first"}
            assert options == {"aws_session_token": "first"}


class TestConnect:
    """What reaches ``lancedb.connect()``."""

    def _initialize(self, config):
        # `_initialize` reads only the config when the table name is set and
        # the store reports no tables, so a full index -- which needs a
        # dataset -- is not required to observe the connect call
        index = types.SimpleNamespace(config=config)
        LanceDBSimilarityIndex._initialize(index)
        return index

    def test_storage_options_are_passed(self):
        options = {"aws_access_key_id": "key"}
        config = LanceDBSimilarityConfig(
            table_name="a-table",
            uri="s3://bucket/org-abc",
            storage_options=options,
        )

        with mock.patch.object(foblancedb, "lancedb") as lancedb:
            lancedb.connect.return_value.table_names.return_value = []
            self._initialize(config)

        lancedb.connect.assert_called_once_with(
            "s3://bucket/org-abc", storage_options=options
        )

    @pytest.mark.parametrize(
        "options",
        [pytest.param(None, id="none"), pytest.param({}, id="empty")],
    )
    def test_the_argument_is_omitted_when_unset(self, options):
        # No minimum lancedb version is declared, so a release without the
        # parameter must still work for callers that set no options
        config = LanceDBSimilarityConfig(
            table_name="a-table", storage_options=options
        )

        with mock.patch.object(foblancedb, "lancedb") as lancedb:
            lancedb.connect.return_value.table_names.return_value = []
            self._initialize(config)

        lancedb.connect.assert_called_once_with("/tmp/lancedb")


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
            pytest.param("uri", "s3://bucket/SENSITIVE/dataset-1", id="uri"),
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
        assert "storage_options" not in serialized


def _assign(obj, name, value):
    setattr(obj, name, value)
    return obj
