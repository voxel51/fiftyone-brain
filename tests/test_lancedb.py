"""
Unit tests for the LanceDB similarity backend's configuration.

Credentials, serialization, and which store a run resolves to -- none of
which needs a database, so these run wherever the suite does. What the
backend does against a real store is covered in ``test_lancedb_store.py``,
and the integration tests live in ``tests/intensive/test_similarity.py``.

| Copyright 2017-2026, Voxel51, Inc.
| `voxel51.com <https://voxel51.com/>`_
|
"""

from unittest import mock

import pytest

import fiftyone.brain as fob
from fiftyone.brain.internal.core import lancedb as foblancedb
from fiftyone.brain.internal.core.lancedb import LanceDBSimilarityConfig


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


class TestSerialization:
    """What reaches the database when a brain run is saved."""

    def test_credentials_are_not_serialized(self):
        # Stored privately so a vended credential never lands in the
        # dataset's brain document
        config = LanceDBSimilarityConfig(
            storage_options={"aws_secret_access_key": "SENSITIVE"}
        )
        serialized = config.serialize()

        assert "storage_options" not in serialized
        assert "SENSITIVE" not in str(serialized)

    def test_the_uri_is_serialized(self):
        # A location rather than a credential, and no more revealing than
        # the sample filepaths in the same database. Recorded so two runs
        # may keep their tables in different stores, and so a run keeps the
        # one it was built in whatever a later reader is configured with
        config = LanceDBSimilarityConfig(uri="s3://bucket/vectors")

        assert config.serialize()["uri"] == "s3://bucket/vectors"

    def test_a_run_naming_no_store_serializes_none(self):
        # The fallback is resolved when the store is opened rather than
        # assigned here, so an unnamed run cannot acquire whichever store
        # happened to be configured the first time something read it
        config = LanceDBSimilarityConfig(table_name="a-table")

        assert config.serialize()["uri"] is None


class TestResolveUri:
    """Which store a run opens."""

    @pytest.fixture(name="backend")
    def fixture_backend(self):
        backends = fob.brain_config.similarity_backends
        original = backends.get("lancedb", {}).copy()
        entry = backends.setdefault("lancedb", {})
        entry.pop("uri", None)

        yield entry

        backends["lancedb"] = original

    def test_the_run_s_own_uri_wins(self, backend):
        backend["uri"] = "s3://configured/vectors"
        config = LanceDBSimilarityConfig(uri="s3://recorded/vectors")

        config.load_credentials()

        assert config.resolve_uri() == "s3://recorded/vectors"

    def test_an_unnamed_run_takes_the_configured_store(self, backend):
        backend["uri"] = "s3://configured/vectors"
        config = LanceDBSimilarityConfig()

        config.load_credentials()

        assert config.resolve_uri() == "s3://configured/vectors"
        # Resolved, not assigned: a later save must not record it
        assert config.uri is None

    @pytest.mark.usefixtures("backend")
    def test_neither_falls_back_to_a_local_directory(self):
        config = LanceDBSimilarityConfig()

        config.load_credentials()

        assert config.resolve_uri() == foblancedb.DEFAULT_URI

    def test_an_explicit_argument_wins_over_both(self, backend):
        backend["uri"] = "s3://configured/vectors"
        config = LanceDBSimilarityConfig(uri="s3://recorded/vectors")

        config.load_credentials(uri="s3://explicit/vectors")

        assert config.resolve_uri() == "s3://explicit/vectors"

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
