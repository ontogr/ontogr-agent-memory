"""Retry, breaker and metrics policy shared by every backend."""

import asyncio
import logging
import random
import time
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, TypeVar

from ontogr_mem.graphstore.config import ResilienceSettings
from ontogr_mem.graphstore.errors import (
    AlreadyExistsError,
    CircuitOpenError,
    ConstraintViolationError,
    DatabaseUnavailableError,
    DeleteRefusedError,
    NotFoundError,
    QueryTimeoutError,
    StoreQueryError,
)
from ontogr_mem.observability.metrics import Metrics
from ontogr_mem.resilience import CircuitBreaker


log = logging.getLogger(__name__)
T = TypeVar("T")


class ErrorKind(Enum):
    """What a backend exception means for the retry policy."""

    PRE_EXECUTION = "pre_execution"
    OUTCOME_UNKNOWN = "outcome_unknown"
    SERVER_TRANSIENT = "server_transient"
    TIMEOUT = "timeout"
    CONFLICT = "conflict"
    REJECTED = "rejected"
    ALREADY_EXISTS = "already_exists"
    NOT_FOUND = "not_found"
    DELETE_REFUSED = "delete_refused"


Classifier = Callable[[BaseException], ErrorKind | None]


@dataclass(frozen=True)
class QuerySpec:
    """One statement plus the facts the executor needs to run it safely.

    Attributes:
        cypher: Statement text.
        params: Query parameters.
        read_only: Route to the read-only path.
        idempotent: Safe to send again after an outcome-unknown failure.
        timeout_ms: Per-query server timeout; the settings default if None.
        tag: Label for metrics and logs.
        auto_commit: Run outside a managed transaction, for statements that
            cannot run inside one. Backends without transactions ignore it.
    """

    cypher: str
    params: Mapping[str, Any] = field(default_factory=dict)
    read_only: bool = False
    idempotent: bool = False
    timeout_ms: int | None = None
    tag: str = ""
    auto_commit: bool = False


@dataclass(frozen=True)
class Rows:
    """Query result: dict rows plus write statistics."""

    rows: list[dict[str, Any]]
    stats: Mapping[str, int]

    def scalar(self, column: str, default: Any = None) -> Any:
        """Return `column` of the first row, or `default` if there are no rows."""
        return self.rows[0].get(column, default) if self.rows else default


def answered_error(kind: ErrorKind, exc: BaseException) -> StoreQueryError:
    """Return the typed error for a failure the server answered."""
    if kind is ErrorKind.TIMEOUT:
        return QueryTimeoutError(str(exc))
    if kind is ErrorKind.CONFLICT:
        return ConstraintViolationError(str(exc))
    if kind is ErrorKind.ALREADY_EXISTS:
        return AlreadyExistsError(str(exc))
    if kind is ErrorKind.NOT_FOUND:
        return NotFoundError(str(exc))
    if kind is ErrorKind.DELETE_REFUSED:
        return DeleteRefusedError(str(exc))
    return StoreQueryError(str(exc))


class Executor:
    """Runs calls under one retry and breaker policy for one server.

    One breaker guards the whole server. The backend supplies `classify`, which
    maps its exceptions to an `ErrorKind`, or to None for exceptions that are
    not the server's fault and must propagate unchanged.
    """

    def __init__(
        self,
        settings: ResilienceSettings,
        classify: Classifier,
        breaker: CircuitBreaker | None = None,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        rng: Callable[[], float] = random.random,
    ) -> None:
        """Create an executor; `sleep` and `rng` are injectable for tests."""
        self.settings = settings
        self._classify = classify
        self.breaker = breaker or CircuitBreaker(
            settings.breaker_failure_threshold, settings.breaker_recovery_s
        )
        self._sleep = sleep
        self._rng = rng
        self.metrics = Metrics()

    async def execute(
        self,
        call: Callable[[], Awaitable[T]],
        *,
        idempotent: bool,
        tag: str = "",
    ) -> T:
        """Run `call` under the retry and breaker policy.

        A failure is retried when the call is idempotent, or when it happened
        before the statement could have reached the server.

        Raises:
            CircuitOpenError: The breaker is open.
            DatabaseUnavailableError: Attempts ran out on a transient error, or
                a non-idempotent call failed with an unknown outcome.
            QueryTimeoutError: The server killed the query.
            ConstraintViolationError: A uniqueness constraint rejected the write.
            AlreadyExistsError: The object the statement creates already exists.
            NotFoundError: The object the statement names does not exist.
            DeleteRefusedError: A node to delete still has edges.
            StoreQueryError: The server rejected the statement.
        """
        last: BaseException | None = None
        attempts = self.settings.max_attempts
        for attempt in range(attempts):
            if not self.breaker.allow_request():
                msg = "circuit breaker is open"
                raise CircuitOpenError(msg)
            started = time.perf_counter()
            try:
                result = await call()
            except asyncio.CancelledError:
                self.breaker.release_probe()
                raise
            except Exception as exc:
                kind = self._classify(exc)
                if kind is None:
                    self.breaker.release_probe()
                    raise
                last = exc
                if kind in (ErrorKind.PRE_EXECUTION, ErrorKind.OUTCOME_UNKNOWN):
                    self.breaker.record_failure()
                    self.metrics.inc("transient_errors")
                    if kind is ErrorKind.OUTCOME_UNKNOWN and not idempotent:
                        msg = (
                            f"{tag or 'query'}: outcome unknown after {exc!r}; "
                            "not retried because the statement is not idempotent"
                        )
                        raise DatabaseUnavailableError(msg) from exc
                elif kind is ErrorKind.SERVER_TRANSIENT:
                    self.breaker.record_failure()
                    self.metrics.inc("transient_errors")
                else:
                    # The server answered, so the breaker counts it as healthy.
                    self.breaker.record_success()
                    raise answered_error(kind, exc) from exc
            else:
                self.breaker.record_success()
                elapsed = (time.perf_counter() - started) * 1000
                self.metrics.inc(f"calls:{tag}")
                if elapsed > self.settings.slow_query_ms:
                    log.warning("slow query tag=%s took %.0f ms", tag, elapsed)
                return result
            if attempt + 1 < attempts:
                self.metrics.inc("retries")
                delay = self.settings.retry_delay * 2**attempt * (0.5 + self._rng())
                await self._sleep(delay)
        msg = f"{tag or 'query'} failed after {attempts} attempts: {last!r}"
        raise DatabaseUnavailableError(msg) from last
