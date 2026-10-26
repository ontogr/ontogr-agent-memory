"""Query execution for Neo4j: error classification, result rows and runners."""

import asyncio
import random
from collections.abc import Awaitable, Callable
from typing import Any

from neo4j import (
    READ_ACCESS,
    WRITE_ACCESS,
    AsyncDriver,
    AsyncManagedTransaction,
    Query,
    unit_of_work,
)
from neo4j.exceptions import (
    Neo4jError,
    ServiceUnavailable,
    SessionExpired,
    TransientError,
)

from ontogr_mem.graphstore.config import ResilienceSettings
from ontogr_mem.graphstore.cypher import PROPERTY_SCOPE, TenantScope
from ontogr_mem.graphstore.executor import ErrorKind, Executor, QuerySpec, Rows
from ontogr_mem.resilience import CircuitBreaker


PRE_EXECUTION_MESSAGES = (
    "couldn't connect",
    "connection refused",
    "failed to establish",
    "name or service not known",
    "cannot resolve",
)
"""Fragments of connection errors raised before a statement was sent.

The driver raises one `ServiceUnavailable` for an unreachable server and for a
connection lost mid-statement, so the message is the only way to tell them
apart.
"""

TIMEOUT_CODES = (
    "Neo.ClientError.Transaction.TransactionTimedOut",
    "Neo.ClientError.Transaction.TransactionTimedOutClientConfiguration",
    "Neo.TransientError.Transaction.TransactionTimedOut",
)

CONFLICT_CODES = ("Neo.ClientError.Schema.ConstraintValidationFailed",)

ALREADY_EXISTS_CODES = (
    "Neo.ClientError.Schema.ConstraintAlreadyExists",
    "Neo.ClientError.Schema.IndexAlreadyExists",
)

_STAT_NAMES = (
    "nodes_created",
    "nodes_deleted",
    "relationships_created",
    "relationships_deleted",
    "properties_set",
    "labels_added",
)


def _entity(value: Any) -> Any:
    """Turn driver Node and Relationship objects into plain dicts."""
    if isinstance(value, list):
        return [_entity(v) for v in value]
    labels = getattr(value, "labels", None)
    if labels is not None:
        return {"labels": sorted(labels), "properties": dict(value)}
    rel_type = getattr(value, "type", None)
    if isinstance(rel_type, str):
        return {"type": rel_type, "properties": dict(value)}
    return value


def _classify_server(exc: Neo4jError) -> ErrorKind:
    code = exc.code or ""
    if code in TIMEOUT_CODES:
        return ErrorKind.TIMEOUT
    if code in CONFLICT_CODES:
        return ErrorKind.CONFLICT
    if code in ALREADY_EXISTS_CODES:
        return ErrorKind.ALREADY_EXISTS
    if isinstance(exc, TransientError):
        return ErrorKind.SERVER_TRANSIENT
    return ErrorKind.REJECTED


def classify(exc: BaseException) -> ErrorKind | None:
    """Map a Neo4j driver exception to an `ErrorKind`.

    The exception class and server code decide first. The message is read only
    to tell an unreachable server from a connection lost mid-statement.
    """
    if isinstance(exc, Neo4jError):
        return _classify_server(exc)
    if isinstance(exc, ServiceUnavailable | SessionExpired | OSError):
        message = str(exc).lower()
        if isinstance(exc, ConnectionRefusedError) or any(
            m in message for m in PRE_EXECUTION_MESSAGES
        ):
            return ErrorKind.PRE_EXECUTION
        return ErrorKind.OUTCOME_UNKNOWN
    return None


class Neo4jRunner(Executor):
    """Executes statements and arbitrary calls against one Neo4j server."""

    def __init__(
        self,
        driver_factory: Callable[[], AsyncDriver],
        settings: ResilienceSettings,
        *,
        database: str,
        breaker: CircuitBreaker | None = None,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        rng: Callable[[], float] = random.random,
    ) -> None:
        """Create a runner; `sleep` and `rng` are injectable for tests."""
        super().__init__(settings, classify, breaker, sleep, rng)
        self._driver = driver_factory
        self._database = database

    async def run(self, spec: QuerySpec, tenant: str | None = None) -> Rows:
        """Execute `spec`, passing `tenant` as the `$tenant` parameter.

        Reads and writes run in managed transactions with a server-side timeout.
        A spec with `auto_commit` runs outside a transaction.
        """
        timeout_s = (spec.timeout_ms or self.settings.query_timeout_ms) / 1000
        params = dict(spec.params)
        if tenant is not None:
            params["tenant"] = tenant

        async def work(tx: AsyncManagedTransaction) -> Rows:
            result = await _send(tx, spec.cypher, params)
            return await _collect(result)

        async def call() -> Rows:
            mode = READ_ACCESS if spec.read_only else WRITE_ACCESS
            async with self._driver().session(
                database=self._database, default_access_mode=mode
            ) as session:
                if spec.auto_commit:
                    result = await session.run(
                        _query(spec.cypher, timeout_s),
                        params,
                    )
                    return await _collect(result)
                run = session.execute_read if spec.read_only else session.execute_write
                return await run(unit_of_work(timeout=timeout_s)(work))

        return await self.execute(call, idempotent=spec.idempotent, tag=spec.tag)


def _query(cypher: str, timeout_s: float) -> Query:
    """Return the driver's query object for `cypher` with its timeout."""
    # The runtime accepts generated Cypher, though the driver type requires literals.
    return Query(cypher, timeout=timeout_s)  # ty: ignore[invalid-argument-type]


async def _send(
    tx: AsyncManagedTransaction, cypher: str, params: dict[str, Any]
) -> Any:
    """Run `cypher` in `tx` and return the driver's result."""
    # The runtime accepts generated Cypher, though the driver type requires literals.
    return await tx.run(cypher, params)  # ty: ignore[invalid-argument-type]


async def _collect(result: Any) -> Rows:
    rows = [{k: _entity(v) for k, v in record.items()} async for record in result]
    counters = (await result.consume()).counters
    return Rows(rows, {name: int(getattr(counters, name, 0)) for name in _STAT_NAMES})


class TenantRunner:
    """A `ScopedRunner` bound to one tenant."""

    def __init__(self, runner: Neo4jRunner, tenant: str) -> None:
        """Bind to a tenant."""
        self._runner = runner
        self._tenant = tenant
        self.scope: TenantScope = PROPERTY_SCOPE

    async def run(self, spec: QuerySpec) -> Rows:
        """Execute `spec`; the tenant is available as the `$tenant` parameter."""
        return await self._runner.run(spec, self._tenant)
