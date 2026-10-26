"""Tests for Neo4j settings."""

import pytest

from ontogr_mem.graphstore.neo4j.settings import Neo4jSettings


@pytest.fixture(autouse=True)
def clean_environment(monkeypatch):
    """Remove Neo4j variables a developer machine may have set."""
    for name in (
        "MEMORY_NEO4J_URI",
        "MEMORY_NEO4J_PASSWORD",
        "MEMORY_NEO4J_RESILIENCE__MAX_ATTEMPTS",
    ):
        monkeypatch.delenv(name, raising=False)


class TestNeo4jSettings:
    """Behavior of Neo4jSettings."""

    def test_defaults_point_at_a_local_server_without_a_password(self):
        """With no environment the settings name a local bolt server."""
        settings = Neo4jSettings(_env_file=None)

        assert settings.uri == "bolt://localhost:7687"
        assert settings.password is None
        assert settings.resilience.max_attempts == 3

    def test_reads_prefixed_and_nested_environment(self, monkeypatch):
        """The prefix and the double-underscore delimiter select fields."""
        monkeypatch.setenv("MEMORY_NEO4J_URI", "bolt://db.internal:7687")
        monkeypatch.setenv("MEMORY_NEO4J_PASSWORD", "secret")
        monkeypatch.setenv("MEMORY_NEO4J_RESILIENCE__MAX_ATTEMPTS", "7")

        settings = Neo4jSettings(_env_file=None)

        assert settings.uri == "bolt://db.internal:7687"
        assert settings.password.get_secret_value() == "secret"
        assert settings.resilience.max_attempts == 7
