"""Parse graph and state store URLs without importing store drivers."""

from dataclasses import dataclass
from ipaddress import ip_address
from pathlib import Path
from typing import ClassVar
from urllib.parse import SplitResult, parse_qs, quote, unquote, urlsplit

from pydantic import SecretStr

from ontogr_mem.errors import AgentMemoryError


DEFAULT_GRAPH_URL = "redis://localhost:6379"


def _bracket_ipv6(host: str) -> str:
    """Wrap an IPv6 host in brackets so it is unambiguous in a URL authority."""
    return f"[{host}]" if ":" in host else host


class StoreUrlError(AgentMemoryError, ValueError):
    """A store URL is invalid or unsupported."""


class ExplicitTenantError(AgentMemoryError):
    """A remote graph requires an explicitly selected tenant."""


@dataclass(frozen=True, slots=True)
class FalkorTarget:
    """Connection details for a FalkorDB store."""

    backend: ClassVar[str] = "falkordb"
    host: str
    port: int
    username: str | None
    password: SecretStr | None
    tls: bool

    def identity(self) -> str:
        """Return a credential-free connection identity."""
        scheme = "rediss" if self.tls else "redis"
        return f"{scheme}://{_bracket_ipv6(self.host)}:{self.port}"


@dataclass(frozen=True, slots=True)
class Neo4jTarget:
    """Connection details for a Neo4j store."""

    backend: ClassVar[str] = "neo4j"
    uri: str
    host: str
    port: int
    username: str | None
    password: SecretStr | None
    database: str

    def identity(self) -> str:
        """Return a credential-free identity including the database name."""
        scheme = urlsplit(self.uri).scheme
        host = _bracket_ipv6(self.host)
        return f"{scheme}://{host}:{self.port}/{self.database}"


@dataclass(frozen=True, slots=True)
class SqliteTarget:
    """Connection details for a SQLite store."""

    backend: ClassVar[str] = "sqlite"
    path: str

    def identity(self) -> str:
        """Return the resolved SQLite path."""
        return f"sqlite:///{Path(self.path).expanduser().resolve()}"


@dataclass(frozen=True, slots=True)
class PostgresTarget:
    """Connection details for a PostgreSQL store."""

    backend: ClassVar[str] = "postgres"
    scheme: str
    host: str
    port: int
    database: str
    dsn: SecretStr

    def identity(self) -> str:
        """Return a credential-free PostgreSQL identity."""
        host = _bracket_ipv6(self.host)
        return f"{self.scheme}://{host}:{self.port}/{self.database}"


GraphTarget = FalkorTarget | Neo4jTarget
StateTarget = SqliteTarget | PostgresTarget


def _port(parts: SplitResult, default: int, variable: str) -> int:
    try:
        value = parts.port
    except ValueError:
        msg = f"{variable} has an invalid port"
        raise StoreUrlError(msg) from None
    if value is None:
        return default
    if not 1 <= value <= 65535:
        msg = f"{variable} has an invalid port"
        raise StoreUrlError(msg)
    return value


def parse_graph_url(value: str) -> GraphTarget:
    """Parse a supported graph URL, excluding credentials from safe fields."""
    parts = urlsplit(value)
    scheme = parts.scheme.lower()
    if scheme not in {
        "redis",
        "rediss",
        "falkor",
        "falkors",
        "bolt",
        "bolt+s",
        "bolt+ssc",
        "neo4j",
        "neo4j+s",
        "neo4j+ssc",
    }:
        msg = "MEMORY_GRAPH_URL has an unsupported scheme"
        raise StoreUrlError(msg)
    try:
        host = parts.hostname
        url_port = parts.port
    except ValueError:
        msg = "MEMORY_GRAPH_URL has an invalid authority or port"
        raise StoreUrlError(msg) from None
    if not host:
        msg = "MEMORY_GRAPH_URL requires a host"
        raise StoreUrlError(msg)
    username = (
        unquote(parts.netloc.rsplit("@", 1)[0].split(":", 1)[0])
        if "@" in parts.netloc
        else None
    )
    raw_password = (
        parts.netloc.rsplit("@", 1)[0].split(":", 1)[1]
        if "@" in parts.netloc and ":" in parts.netloc.rsplit("@", 1)[0]
        else None
    )
    password = SecretStr(unquote(raw_password)) if raw_password is not None else None
    if username is not None and password is None:
        msg = "MEMORY_GRAPH_URL username requires a password"
        raise StoreUrlError(msg)
    default_port = 6379 if scheme in {"redis", "rediss", "falkor", "falkors"} else 7687
    port = _port(parts, default_port, "MEMORY_GRAPH_URL")
    if parts.path not in {"", "/"}:
        msg = "MEMORY_GRAPH_URL must not contain a path"
        raise StoreUrlError(msg)
    if scheme in {"redis", "rediss", "falkor", "falkors"}:
        if parts.query or parts.fragment:
            msg = "MEMORY_GRAPH_URL must not contain a query or fragment"
            raise StoreUrlError(msg)
        return FalkorTarget(
            host, port, username, password, scheme in {"rediss", "falkors"}
        )
    if username == "":
        msg = "MEMORY_GRAPH_URL username must not be empty"
        raise StoreUrlError(msg)
    query = parse_qs(parts.query, keep_blank_values=True)
    if set(query) - {"database"} or len(query.get("database", [])) > 1:
        msg = "MEMORY_GRAPH_URL supports only one database query value"
        raise StoreUrlError(msg)
    database = query.get("database", ["neo4j"])[0]
    if not database:
        msg = "MEMORY_GRAPH_URL database must not be empty"
        raise StoreUrlError(msg)
    host_part = _bracket_ipv6(host)
    if url_port is not None:
        host_part += f":{url_port}"
    uri = f"{scheme}://{host_part}"
    return Neo4jTarget(uri, host, port, username, password, database)


def parse_state_url(value: str) -> StateTarget:
    """Parse a SQLite or PostgreSQL URL."""
    parts = urlsplit(value)
    if parts.scheme == "sqlite":
        if parts.netloc:
            msg = "MEMORY_STATE_URL SQLite URL must not have a host"
            raise StoreUrlError(msg)
        if parts.query or parts.fragment:
            msg = "MEMORY_STATE_URL SQLite URL must not have a query or fragment"
            raise StoreUrlError(msg)
        if not parts.path.startswith("/"):
            msg = "MEMORY_STATE_URL SQLite URL must have a path"
            raise StoreUrlError(msg)
        path = unquote(parts.path[1:])
        if not path:
            msg = "MEMORY_STATE_URL SQLite path must not be empty"
            raise StoreUrlError(msg)
        if path == ":memory:":
            msg = "MEMORY_STATE_URL SQLite URL must not be in-memory"
            raise StoreUrlError(msg)
        return SqliteTarget(path)
    if parts.scheme not in {"postgres", "postgresql"}:
        msg = "MEMORY_STATE_URL has an unsupported scheme"
        raise StoreUrlError(msg)
    try:
        host = parts.hostname
    except ValueError:
        msg = "MEMORY_STATE_URL has an invalid authority or port"
        raise StoreUrlError(msg) from None
    if not host:
        msg = "MEMORY_STATE_URL requires a host"
        raise StoreUrlError(msg)
    port = _port(parts, 5432, "MEMORY_STATE_URL")
    return PostgresTarget(
        parts.scheme, host, port, unquote(parts.path.lstrip("/")), SecretStr(value)
    )


def sqlite_url(path: Path) -> str:
    """Build a SQLite URL from a filesystem path."""
    return f"sqlite:///{quote(path.as_posix(), safe='/:')}"


def is_loopback_host(host: str) -> bool:
    """Return whether a host is a loopback name or IP address."""
    if host.casefold() == "localhost":
        return True
    try:
        return ip_address(host).is_loopback
    except ValueError:
        return False


def require_explicit_tenant(
    graph_url: SecretStr, *, tenant_given: bool, hint: str
) -> None:
    """Require an explicit tenant when the graph host is remote."""
    target = parse_graph_url(graph_url.get_secret_value())
    if not is_loopback_host(target.host) and not tenant_given:
        msg = f"A non-loopback graph needs an explicit tenant: {hint}."
        raise ExplicitTenantError(msg)
