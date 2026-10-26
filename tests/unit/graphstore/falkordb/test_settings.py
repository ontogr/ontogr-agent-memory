"""Tests for FalkorDB settings."""

import pytest
from pydantic import ValidationError

from ontogr_mem.graphstore.falkordb.settings import FalkorDBSettings


@pytest.fixture(autouse=True)
def clean_environment(monkeypatch):
    """Remove FalkorDB variables a developer machine may have set."""
    for name in (
        "MEMORY_FALKORDB_HOST",
        "MEMORY_FALKORDB_PORT",
        "MEMORY_FALKORDB_RESILIENCE__MAX_ATTEMPTS",
    ):
        monkeypatch.delenv(name, raising=False)


class TestFalkorDBSettings:
    """Behavior of FalkorDBSettings."""

    def test_defaults(self):
        """Defaults point at a local server with the standard resilience values."""
        settings = FalkorDBSettings(_env_file=None)

        assert (settings.host, settings.port, settings.tls) == (
            "localhost",
            6379,
            False,
        )
        assert settings.max_connections == 16
        assert settings.resilience.max_attempts == 3
        assert settings.resilience.query_timeout_ms == 10_000
        assert settings.resilience.batch_size == 500

    def test_reads_prefixed_and_nested_environment(self, monkeypatch):
        """The prefix and the double-underscore delimiter select fields."""
        monkeypatch.setenv("MEMORY_FALKORDB_HOST", "db.internal")
        monkeypatch.setenv("MEMORY_FALKORDB_RESILIENCE__MAX_ATTEMPTS", "7")

        settings = FalkorDBSettings(_env_file=None)

        assert settings.host == "db.internal"
        assert settings.resilience.max_attempts == 7

    def test_rejects_zero_attempts(self, monkeypatch):
        """At least one attempt is required."""
        monkeypatch.setenv("MEMORY_FALKORDB_RESILIENCE__MAX_ATTEMPTS", "0")

        with pytest.raises(ValidationError):
            FalkorDBSettings(_env_file=None)
