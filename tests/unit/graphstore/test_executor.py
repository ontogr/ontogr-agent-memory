"""Tests for the retry and breaker policy."""

import asyncio

import pytest

from ontogr_mem.graphstore.config import ResilienceSettings
from ontogr_mem.graphstore.errors import (
    CircuitOpenError,
    ConstraintViolationError,
    DatabaseUnavailableError,
    QueryTimeoutError,
    StoreQueryError,
)
from ontogr_mem.graphstore.executor import ErrorKind, Executor
from ontogr_mem.resilience import BreakerState, CircuitBreaker


class DownError(Exception):
    """Failure before the statement reached the server."""


class DroppedError(Exception):
    """Connection lost after the statement was sent."""


class RejectedError(Exception):
    """The server answered with an error."""

    def __init__(self, kind: ErrorKind) -> None:
        """Remember the kind the classifier should report."""
        super().__init__(kind.value)
        self.kind = kind


def classify(exc: BaseException) -> ErrorKind | None:
    """Map the test exceptions to error kinds."""
    if isinstance(exc, DownError):
        return ErrorKind.PRE_EXECUTION
    if isinstance(exc, DroppedError):
        return ErrorKind.OUTCOME_UNKNOWN
    if isinstance(exc, RejectedError):
        return exc.kind
    return None


class Clock:
    """A clock the test moves by hand."""

    def __init__(self) -> None:
        """Start at time zero."""
        self.now = 0.0

    def __call__(self) -> float:
        """Return the current time."""
        return self.now


class Script:
    """An awaitable that plays back scripted outcomes and counts calls."""

    def __init__(self, *outcomes):
        """Store the outcomes to play back in order."""
        self.outcomes = list(outcomes)
        self.calls = 0

    async def __call__(self):
        """Play back the next outcome."""
        self.calls += 1
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


def make_executor(clock=None, **settings):
    """Build an executor that records its sleeps instead of waiting."""
    sleeps: list[float] = []

    async def sleep(seconds: float) -> None:
        sleeps.append(seconds)

    config = ResilienceSettings(retry_delay=1.0, **settings)
    breaker = CircuitBreaker(
        config.breaker_failure_threshold,
        config.breaker_recovery_s,
        clock or Clock(),
    )
    executor = Executor(config, classify, breaker, sleep=sleep, rng=lambda: 0.5)
    return executor, sleeps


class TestExecutor:
    """Behavior of Executor."""

    async def test_idempotent_call_is_retried_with_backoff_then_succeeds(self):
        """Unknown-outcome failures of an idempotent call are retried."""
        executor, sleeps = make_executor()
        call = Script(DroppedError(), DroppedError(), "done")

        result = await executor.execute(call, idempotent=True)

        assert result == "done"
        assert call.calls == 3
        assert sleeps == [1.0, 2.0]

    async def test_non_idempotent_call_is_not_resent_after_unknown_outcome(self):
        """Resending could apply a write twice, so it fails instead."""
        executor, _ = make_executor()
        call = Script(DroppedError(), "done")

        with pytest.raises(DatabaseUnavailableError, match="outcome unknown"):
            await executor.execute(call, idempotent=False)

        assert call.calls == 1

    async def test_non_idempotent_call_is_retried_when_it_never_reached_the_server(
        self,
    ):
        """A failure before sending is always safe to retry."""
        executor, _ = make_executor()
        call = Script(DownError(), "done")

        assert await executor.execute(call, idempotent=False) == "done"
        assert call.calls == 2

    async def test_max_attempts_counts_total_tries(self):
        """Attempts run out with a typed error."""
        executor, _ = make_executor(max_attempts=3)
        call = Script(*[DroppedError()] * 5)

        with pytest.raises(DatabaseUnavailableError, match="after 3 attempts"):
            await executor.execute(call, idempotent=True)

        assert call.calls == 3

    @pytest.mark.parametrize(
        ("kind", "error"),
        [
            (ErrorKind.REJECTED, StoreQueryError),
            (ErrorKind.TIMEOUT, QueryTimeoutError),
            (ErrorKind.CONFLICT, ConstraintViolationError),
        ],
    )
    async def test_server_answer_is_typed_and_not_retried(self, kind, error):
        """An answered failure is raised once with the matching error type."""
        executor, _ = make_executor()
        call = Script(RejectedError(kind))

        with pytest.raises(error):
            await executor.execute(call, idempotent=True)

        assert call.calls == 1

    async def test_server_answer_resets_the_failure_count(self):
        """A rejection proves the server is up, so earlier failures are forgotten."""
        executor, _ = make_executor(breaker_failure_threshold=2, max_attempts=1)
        with pytest.raises(DatabaseUnavailableError):
            await executor.execute(Script(DroppedError()), idempotent=True)
        with pytest.raises(StoreQueryError):
            await executor.execute(
                Script(RejectedError(ErrorKind.REJECTED)), idempotent=True
            )

        with pytest.raises(DatabaseUnavailableError):
            await executor.execute(Script(DroppedError()), idempotent=True)

        assert executor.breaker.state is BreakerState.CLOSED

    async def test_unclassified_errors_propagate_unchanged(self):
        """An error that is not the server's fault is not wrapped or retried."""
        executor, _ = make_executor()
        call = Script(KeyError("x"))

        with pytest.raises(KeyError):
            await executor.execute(call, idempotent=True)

        assert call.calls == 1

    async def test_probe_that_gets_a_server_answer_closes_the_breaker(self):
        """A rejected half-open probe still proves the server is back."""
        clock = Clock()
        executor, _ = make_executor(clock, breaker_failure_threshold=2, max_attempts=1)
        for _ in range(2):
            with pytest.raises(DatabaseUnavailableError):
                await executor.execute(Script(DroppedError()), idempotent=True)
        with pytest.raises(CircuitOpenError):
            await executor.execute(Script("x"), idempotent=True)
        clock.now = 31

        with pytest.raises(StoreQueryError):
            await executor.execute(
                Script(RejectedError(ErrorKind.REJECTED)), idempotent=True
            )

        assert executor.breaker.state is BreakerState.CLOSED
        assert await executor.execute(Script("ok"), idempotent=True) == "ok"

    async def test_cancelled_probe_frees_the_slot(self):
        """Cancelling the half-open probe does not wedge the breaker."""
        clock = Clock()
        executor, _ = make_executor(clock, breaker_failure_threshold=1, max_attempts=1)
        executor.breaker.record_failure()
        clock.now = 31

        async def hang():
            await asyncio.sleep(10)

        task = asyncio.create_task(executor.execute(hang, idempotent=True))
        await asyncio.sleep(0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        assert executor.breaker.allow_request()
