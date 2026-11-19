"""Tests for Neo4j settings."""

import pytest

from ontogr_mem.graphstore.neo4j.settings import Neo4jSettings


@pytest.fixture(autouse=True)
def clean_environment(monkeypatch):
    """Remove Neo4j variables a developer machine may have set."""
    for name in (
        "MEMORY_NEO4J_MAX_CONNECTIONS",
        "MEMORY_NEO4J_POOL_TIMEOUT",
        "MEMORY_NEO4J_RESILIENCE__MAX_ATTEMPTS",
    ):
        monkeypatch.delenv(name, raising=False)


class TestNeo4jSettings:
    """Behavior of Neo4jSettings."""

    def test_defaults_include_pool_and_resilience_values(self):
        """Defaults include pool tuning and standard resilience values."""
        settings = Neo4jSettings(_env_file=None)

        assert settings.max_connections == 16
        assert settings.pool_timeout == 30
        assert settings.resilience.max_attempts == 3

    def test_reads_prefixed_and_nested_environment(self, monkeypatch):
        """The prefix and the double-underscore delimiter select fields."""
        monkeypatch.setenv("MEMORY_NEO4J_MAX_CONNECTIONS", "32")
        monkeypatch.setenv("MEMORY_NEO4J_POOL_TIMEOUT", "12")
        monkeypatch.setenv("MEMORY_NEO4J_RESILIENCE__MAX_ATTEMPTS", "7")

        settings = Neo4jSettings(_env_file=None)

        assert settings.max_connections == 32
        assert settings.pool_timeout == 12
        assert settings.resilience.max_attempts == 7
