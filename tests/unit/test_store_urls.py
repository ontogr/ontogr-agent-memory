"""Behavior tests for store URL parsing."""

from pathlib import Path

import pytest
from pydantic import SecretStr

from ontogr_mem.store_urls import (
    DEFAULT_GRAPH_URL,
    ExplicitTenantError,
    FalkorTarget,
    Neo4jTarget,
    PostgresTarget,
    SqliteTarget,
    StoreUrlError,
    parse_graph_url,
    parse_state_url,
    require_explicit_tenant,
    sqlite_url,
)


class TestParseGraphUrl:
    """Graph URLs select a backend and keep credentials out of identities."""

    @pytest.mark.parametrize(
        "scheme,tls",
        [("redis", False), ("rediss", True), ("falkor", False), ("falkors", True)],
    )
    def test_maps_falkor_schemes(self, scheme: str, tls: bool) -> None:
        """FalkorDB schemes set the TLS flag and default port."""
        target = parse_graph_url(f"{scheme}://host")
        assert isinstance(target, FalkorTarget)
        assert target.tls is tls
        assert target.port == 6379

    def test_parses_neo4j_credentials_and_safe_identity(self) -> None:
        """Neo4j userinfo is decoded and omitted from the URI and identity."""
        target = parse_graph_url("neo4j+s://user:p%40ss@host/?database=graphs")
        assert isinstance(target, Neo4jTarget)
        assert target.password == SecretStr("p@ss")
        assert target.uri == "neo4j+s://host"
        assert target.identity() == "neo4j+s://host:7687/graphs"
        assert "p@ss" not in repr(target)

    @pytest.mark.parametrize(
        "url",
        [
            "redis://host/0",
            "neo4j://host/db",
            "neo4j://user@host",
            "neo4j://host?policy=x",
            "redis://host:99999",
            "https://host",
            "redis:///missing",
        ],
    )
    def test_rejects_invalid_graph_urls_without_secrets(self, url: str) -> None:
        """Refused graph URLs never repeat the URL in the error text."""
        with pytest.raises(StoreUrlError) as error:
            parse_graph_url(url)
        assert url not in str(error.value)

    def test_default_identity_and_tenant_policy(self) -> None:
        """The default URL is loopback and needs no explicit tenant."""
        target = parse_graph_url(DEFAULT_GRAPH_URL)
        assert isinstance(target, FalkorTarget)
        assert target.identity() == parse_graph_url("redis://localhost").identity()
        require_explicit_tenant(
            SecretStr(DEFAULT_GRAPH_URL), tenant_given=False, hint="tenant"
        )
        require_explicit_tenant(
            SecretStr("redis://remote"), tenant_given=True, hint="tenant"
        )
        with pytest.raises(ExplicitTenantError):
            require_explicit_tenant(
                SecretStr("redis://remote"), tenant_given=False, hint="tenant"
            )


class TestParseStateUrl:
    """State URLs select SQLite or Postgres and keep DSNs secret."""

    @pytest.mark.parametrize(
        "url,path",
        [
            ("sqlite:////tmp/a%20b.db", "/tmp/a b.db"),
            ("sqlite:///relative.db", "relative.db"),
            ("sqlite:///C:/dir/file.db", "C:/dir/file.db"),
        ],
    )
    def test_parses_sqlite_paths(self, url: str, path: str) -> None:
        """SQLite paths decode and keep their absolute or relative form."""
        target = parse_state_url(url)
        assert isinstance(target, SqliteTarget)
        assert target.path == path

    @pytest.mark.parametrize("scheme", ["postgres", "postgresql"])
    def test_preserves_postgres_dsn_and_redacts_identity(self, scheme: str) -> None:
        """Both Postgres schemes pass through and hide credentials."""
        url = f"{scheme}://user:secret@db/memory"
        target = parse_state_url(url)
        assert isinstance(target, PostgresTarget)
        assert target.dsn.get_secret_value() == url
        assert "secret" not in target.identity()
        assert "secret" not in repr(target)

    @pytest.mark.parametrize(
        "url",
        ["sqlite://host/path", "sqlite:///", "memory://", "mysql://db/name"],
    )
    def test_rejects_invalid_state_urls(self, url: str) -> None:
        """Unknown state schemes and malformed SQLite paths are refused."""
        with pytest.raises(StoreUrlError):
            parse_state_url(url)

    def test_builds_sqlite_url(self) -> None:
        """The URL builder percent-encodes the POSIX path."""
        assert sqlite_url(Path("/tmp/a b.db")) == "sqlite:////tmp/a%20b.db"


class TestRequireExplicitTenant:
    """The tenant predicate allows loopback graphs and explicit tenants."""

    @pytest.mark.parametrize("host", ["localhost", "127.0.0.1", "::1"])
    def test_accepts_loopback(self, host: str) -> None:
        """Loopback hosts accept the implicit default tenant."""
        url = f"redis://[{host}]" if ":" in host else f"redis://{host}"
        require_explicit_tenant(SecretStr(url), tenant_given=False, hint="tenant")
