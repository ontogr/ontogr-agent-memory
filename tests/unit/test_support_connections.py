"""Behavior tests for the integration test connection helpers."""

import pytest

from tests.support.falkordb import drop_unique_constraint, falkordb_test_url
from tests.support.neo4j import neo4j_test_url


class TestNeo4jTestUrl:
    """The Neo4j fallback URL follows the configured test password."""

    def test_uses_configured_password_with_encoding(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The compose password is percent-encoded into the fallback URL."""
        monkeypatch.delenv("INTEGRATION_NEO4J_URL", raising=False)
        monkeypatch.setenv("MEMORY_NEO4J_PASSWORD", "p@ss word")

        assert neo4j_test_url() == (
            "bolt://neo4j:p%40ss%20word@localhost:17687?database=neo4j"
        )

    def test_integration_override_wins(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """An explicit integration URL takes precedence over the fallback."""
        monkeypatch.setenv("INTEGRATION_NEO4J_URL", "bolt://other:7687")

        assert neo4j_test_url() == "bolt://other:7687"

    def test_direct_uri_wins(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A caller-provided URI is returned unchanged."""
        monkeypatch.setenv("INTEGRATION_NEO4J_URL", "bolt://other:7687")

        assert neo4j_test_url(uri="bolt://direct:7687") == "bolt://direct:7687"


class TestFalkordbTestUrl:
    """The FalkorDB URL helpers preserve configured credentials and TLS."""

    def test_port_override_keeps_password_only_userinfo(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A port override keeps the default user's password and TLS scheme."""
        monkeypatch.setenv("INTEGRATION_FALKORDB_URL", "rediss://:secret@host:16379")

        assert falkordb_test_url(port=6380) == "rediss://:secret@host:6380"

    async def test_constraint_helper_uses_configured_credentials(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The raw Redis client connects with the URL's credentials and TLS."""
        captured: dict[str, object] = {}

        class FakeRedis:
            """Capture the connection settings used by the raw client."""

            def __init__(self, **kwargs: object) -> None:
                captured.update(kwargs)

            async def execute_command(self, *args: object) -> None:
                """Accept the constraint command without a server."""

            async def aclose(self) -> None:
                """Accept the client close without a server."""

        monkeypatch.setenv(
            "INTEGRATION_FALKORDB_URL", "rediss://user:secret@host:16379"
        )
        monkeypatch.setattr("tests.support.falkordb.Redis", FakeRedis)

        await drop_unique_constraint("graph", "Label", "prop")

        assert captured == {
            "host": "host",
            "port": 16379,
            "username": "user",
            "password": "secret",
            "ssl": True,
            "decode_responses": True,
        }
