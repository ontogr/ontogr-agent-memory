"""Query execution for FalkorDB: error classification, result rows and runners."""

import asyncio
import random
from collections.abc import Awaitable, Callable
from typing import Any, Protocol

from redis.exceptions import (
    BusyLoadingError,
    ResponseError,
    TryAgainError,
)
from redis.exceptions import ConnectionError as RedisConnectionError
from redis.exceptions import TimeoutError as RedisTimeoutError

from ontogr_mem.graphstore.config import ResilienceSettings
from ontogr_mem.graphstore.cypher import GRAPH_SCOPE, TenantScope
from ontogr_mem.graphstore.errors import StoreQueryError
from ontogr_mem.graphstore.executor import ErrorKind, Executor, QuerySpec, Rows
from ontogr_mem.resilience import CircuitBreaker


PRE_EXECUTION_MESSAGES = (
    "no connection available",
    "connection refused",
    "connect call failed",
    "name or service not known",
)
"""Fragments of connection errors raised before a statement was sent.

The client reports these as a plain connection error, so the message is the only
way to tell them from a connection lost mid-statement.
"""

RESPONSE_CODES: dict[str, ErrorKind] = {
    "loading": ErrorKind.SERVER_TRANSIENT,
    "busy": ErrorKind.SERVER_TRANSIENT,
    "tryagain": ErrorKind.SERVER_TRANSIENT,
}
"""Leading words of server error replies that mean a retry may succeed."""

RESPONSE_PHRASES: tuple[tuple[str, ErrorKind], ...] = (
    ("max pending queries", ErrorKind.SERVER_TRANSIENT),
    ("timed out", ErrorKind.TIMEOUT),
    ("unique constraint", ErrorKind.CONFLICT),
)
"""Fragments of FalkorDB error replies that carry no error code."""

NOT_FOUND_PHRASES = ("empty key",)
"""Fragments of the reply FalkorDB sends for a graph that does not exist."""

ALREADY_EXISTS_PHRASES = ("already exists", "already indexed")
"""Fragments of the reply FalkorDB sends for a rule that is already there."""


class GraphLike(Protocol):
    """The slice of the falkordb async graph API the store uses."""

    async def query(
        self,
        q: str,
        params: dict[str, Any] | None = None,
        timeout: int | None = None,
    ) -> Any:
        """Run a read-write query."""
        ...

    async def ro_query(
        self,
        q: str,
        params: dict[str, Any] | None = None,
        timeout: int | None = None,
    ) -> Any:
        """Run a read-only query."""
        ...

    async def list_constraints(self) -> list[dict[str, Any]]:
        """List the constraints of the graph."""
        ...

    async def create_node_unique_constraint(self, label: str, *properties: str) -> Any:
        """Create a unique constraint on the given node properties."""
        ...


_STAT_NAMES = (
    "nodes_created",
    "nodes_deleted",
    "relationships_created",
    "relationships_deleted",
    "properties_set",
    "labels_added",
)


def _entity(value: Any) -> Any:
    """Turn falkordb Node and Edge objects into plain dicts."""
    props = getattr(value, "properties", None)
    if props is None:
        return list(map(_entity, value)) if isinstance(value, list) else value
    if hasattr(value, "labels"):
        return {"labels": sorted(value.labels), "properties": dict(props)}
    return {"type": getattr(value, "relation", None), "properties": dict(props)}


def _column_names(header: Any) -> tuple[str, ...]:
    """Return the column names of a FalkorDB result header.

    The server sends the header as a list of column names. Some replies wrap
    each name in a `[type, name]` pair, so both shapes are read here.
    """
    names: list[str] = []
    for cell in header:
        if isinstance(cell, list | tuple):
            names.append(str(cell[1]))
        else:
            names.append(str(cell))
    return tuple(names)


def to_rows(result: Any) -> Rows:
    """Convert a falkordb query result to `Rows`.

    Raises:
        StoreQueryError: If a row does not have one cell per column.
    """
    header = _column_names(result.header)
    rows: list[dict[str, Any]] = []
    for row in result.result_set:
        if len(row) != len(header):
            msg = f"result row has {len(row)} cells, expected {len(header)}"
            raise StoreQueryError(msg)
        rows.append(
            {name: _entity(cell) for name, cell in zip(header, row, strict=True)}
        )
    stats = {n: int(getattr(result, n, 0) or 0) for n in _STAT_NAMES}
    return Rows(rows, stats)


def _classify_connection(exc: BaseException) -> ErrorKind:
    if isinstance(exc, ConnectionRefusedError):
        return ErrorKind.PRE_EXECUTION
    message = str(exc).lower()
    if any(m in message for m in PRE_EXECUTION_MESSAGES):
        return ErrorKind.PRE_EXECUTION
    return ErrorKind.OUTCOME_UNKNOWN


def _classify_response(exc: BaseException) -> ErrorKind:
    message = str(exc).lower()
    code = message.split(maxsplit=1)[0] if message else ""
    if code in RESPONSE_CODES:
        return RESPONSE_CODES[code]
    for phrase, kind in RESPONSE_PHRASES:
        if phrase in message:
            return kind
    if any(phrase in message for phrase in NOT_FOUND_PHRASES):
        return ErrorKind.NOT_FOUND
    if any(phrase in message for phrase in ALREADY_EXISTS_PHRASES):
        return ErrorKind.ALREADY_EXISTS
    return ErrorKind.REJECTED


def classify(exc: BaseException) -> ErrorKind | None:
    """Map a redis or falkordb exception to an `ErrorKind`.

    The exception class decides first. The message is read only where the client
    raises a bare error for several different failures.
    """
    if isinstance(exc, BusyLoadingError | TryAgainError):
        return ErrorKind.SERVER_TRANSIENT
    if isinstance(exc, TimeoutError):
        return ErrorKind.TIMEOUT
    if isinstance(exc, RedisConnectionError | RedisTimeoutError | OSError):
        return _classify_connection(exc)
    if isinstance(exc, ResponseError):
        return _classify_response(exc)
    return None


class QueryRunner(Executor):
    """Executes statements and arbitrary calls against one FalkorDB server."""

    def __init__(
        self,
        graph_factory: Callable[[str], Awaitable[GraphLike]],
        settings: ResilienceSettings,
        breaker: CircuitBreaker | None = None,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        rng: Callable[[], float] = random.random,
    ) -> None:
        """Create a runner; `sleep` and `rng` are injectable for tests."""
        super().__init__(settings, classify, breaker, sleep, rng)
        self._graphs = graph_factory

    async def run(self, graph_name: str, spec: QuerySpec) -> Rows:
        """Execute `spec` on `graph_name`.

        The timeout is always sent, because the server default is one second.
        """
        timeout = spec.timeout_ms or self.settings.query_timeout_ms

        async def call() -> Rows:
            graph = await self._graphs(graph_name)
            method = graph.ro_query if spec.read_only else graph.query
            return to_rows(await method(spec.cypher, dict(spec.params), timeout))

        return await self.execute(call, idempotent=spec.idempotent, tag=spec.tag)

    async def list_constraints(self, graph_name: str) -> list[dict[str, Any]]:
        """Return the constraints of `graph_name`.

        The client command uses the configured query timeout and the runner's
        retry and breaker policy.
        """

        async def call() -> list[dict[str, Any]]:
            graph = await self._graphs(graph_name)
            return list(
                await asyncio.wait_for(
                    graph.list_constraints(),
                    timeout=self.settings.query_timeout_ms / 1000,
                )
            )

        return await self.execute(call, idempotent=True, tag="constraints")

    async def create_unique_constraint(
        self, graph_name: str, label: str, prop: str
    ) -> None:
        """Create a unique constraint on `label.prop` of `graph_name`.

        FalkorDB backs a uniqueness constraint with a range index and creates
        that index itself when it is missing. The command uses the configured
        query timeout and the runner's retry and breaker policy.
        """

        async def call() -> None:
            graph = await self._graphs(graph_name)
            await asyncio.wait_for(
                graph.create_node_unique_constraint(label, prop),
                timeout=self.settings.query_timeout_ms / 1000,
            )

        await self.execute(call, idempotent=True, tag="ddl:unique")


class GraphRunner:
    """A `ScopedRunner` bound to one tenant graph."""

    def __init__(self, runner: QueryRunner, graph: str) -> None:
        """Bind to a graph name."""
        self._runner = runner
        self._graph = graph
        self.scope: TenantScope = GRAPH_SCOPE

    async def run(self, spec: QuerySpec) -> Rows:
        """Execute `spec` on the bound graph."""
        return await self._runner.run(self._graph, spec)
