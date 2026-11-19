"""Parse graph and state store URLs without importing store drivers."""

from dataclasses import dataclass
from ipaddress import ip_address
from pathlib import Path
from typing import ClassVar
from urllib.parse import SplitResult, parse_qs, quote, unquote, urlsplit

from pydantic import SecretStr

from ontogr_mem.errors import AgentMemoryError


DEFAULT_GRAPH_URL = "redis://localhost:6379"


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
        return f"{scheme}://{self.host}:{self.port}"


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
        return f"{urlsplit(self.uri).scheme}://{self.host}:{self.port}/{self.database}"


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
        return f"{self.scheme}://{self.host}:{self.port}/{self.database}"


GraphTarget = FalkorTarget | Neo4jTarget
StateTarget = SqliteTarget | PostgresTarget


def _port(parts: SplitResult, default: int, variable: str) -> int:
    try:
        value = parts.port
    except ValueError:
        raise StoreUrlError(f"{variable} has an invalid port") from None
    if value is None:
        return default
    if not 1 <= value <= 65535:
        raise StoreUrlError(f"{variable} has an invalid port")
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
        raise StoreUrlError("MEMORY_GRAPH_URL has an unsupported scheme")
    try:
        host = parts.hostname
        url_port = parts.port
    except ValueError:
        raise StoreUrlError(
            "MEMORY_GRAPH_URL has an invalid authority or port"
        ) from None
    if not host:
        raise StoreUrlError("MEMORY_GRAPH_URL requires a host")
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
        raise StoreUrlError("MEMORY_GRAPH_URL username requires a password")
    default_port = 6379 if scheme in {"redis", "rediss", "falkor", "falkors"} else 7687
    port = _port(parts, default_port, "MEMORY_GRAPH_URL")
    if parts.path not in {"", "/"}:
        raise StoreUrlError("MEMORY_GRAPH_URL must not contain a path")
    if scheme in {"redis", "rediss", "falkor", "falkors"}:
        if parts.query or parts.fragment:
            raise StoreUrlError("MEMORY_GRAPH_URL must not contain a query or fragment")
        return FalkorTarget(
            host, port, username, password, scheme in {"rediss", "falkors"}
        )
    if username == "":
        raise StoreUrlError("MEMORY_GRAPH_URL username must not be empty")
    query = parse_qs(parts.query, keep_blank_values=True)
    if set(query) - {"database"} or len(query.get("database", [])) > 1:
        raise StoreUrlError("MEMORY_GRAPH_URL supports only one database query value")
    database = query.get("database", ["neo4j"])[0]
    if not database:
        raise StoreUrlError("MEMORY_GRAPH_URL database must not be empty")
    host_part = f"[{host}]" if ":" in host else host
    if url_port is not None:
        host_part += f":{url_port}"
    uri = f"{scheme}://{host_part}"
    return Neo4jTarget(uri, host, port, username, password, database)


def parse_state_url(value: str) -> StateTarget:
    """Parse a SQLite or PostgreSQL URL."""
    parts = urlsplit(value)
    if parts.scheme == "sqlite":
        if parts.netloc:
            raise StoreUrlError("MEMORY_STATE_URL SQLite URL must not have a host")
        if parts.query or parts.fragment:
            raise StoreUrlError(
                "MEMORY_STATE_URL SQLite URL must not have a query or fragment"
            )
        if not parts.path.startswith("/"):
            raise StoreUrlError("MEMORY_STATE_URL SQLite URL must have a path")
        path = unquote(parts.path[1:])
        if not path:
            raise StoreUrlError("MEMORY_STATE_URL SQLite path must not be empty")
        if path == ":memory:":
            raise StoreUrlError("MEMORY_STATE_URL SQLite URL must not be in-memory")
        return SqliteTarget(path)
    if parts.scheme not in {"postgres", "postgresql"}:
        raise StoreUrlError("MEMORY_STATE_URL has an unsupported scheme")
    try:
        host = parts.hostname
    except ValueError:
        raise StoreUrlError(
            "MEMORY_STATE_URL has an invalid authority or port"
        ) from None
    if not host:
        raise StoreUrlError("MEMORY_STATE_URL requires a host")
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
        raise ExplicitTenantError(
            f"A non-loopback graph needs an explicit tenant: {hint}."
        )
