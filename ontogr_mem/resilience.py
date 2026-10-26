"""Circuit breaker, request-rate limit, and provider-error retry tools."""

import asyncio
import random
import re
import time
from collections import deque
from collections.abc import Awaitable, Callable
from enum import Enum, StrEnum
from typing import TypeVar

from pydantic import Field
from pydantic_settings import BaseSettings


T = TypeVar("T")


class ProviderResilienceSettings(BaseSettings):
    """Optional provider protections, with the caller's environment prefix.

    Attributes:
        breaker_threshold: Consecutive failures before opening, or None to disable.
        breaker_recovery_s: Seconds before allowing one recovery probe.
        requests_per_minute: Request-start limit, or None to disable.
    """

    breaker_threshold: int | None = Field(default=None, ge=1)
    breaker_recovery_s: float = Field(default=30.0, gt=0)
    requests_per_minute: int | None = Field(default=None, ge=1)


class BreakerState(Enum):
    """Breaker states."""

    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreaker:
    """Opens after `failure_threshold` consecutive failures.

    After `recovery_timeout` seconds one probe is allowed. Whatever the probe
    returns must be reported with `record_success` or `record_failure`; a probe
    that was cancelled must call `release_probe`. A non-transient server error
    proves the server answered, so callers report it as a success.

    Reading `state` promotes an open breaker to half-open once the timeout has
    passed, so the state a caller reads is the state the next request meets.
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 30.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        """Create a closed breaker."""
        self._threshold = failure_threshold
        self._recovery = recovery_timeout
        self._clock = clock
        self._state = BreakerState.CLOSED
        self._failures = 0
        self._opened_at = 0.0
        self._probe_in_flight = False

    @property
    def state(self) -> BreakerState:
        """Current state, promoting OPEN to HALF_OPEN once the timeout passed.

        Reading the state promotes it, so every reader sees the same state and
        a half-open probe is never taken twice.
        """
        return self._promote()

    def allow_request(self) -> bool:
        """Return True if a request may be sent now."""
        state = self.state
        if state is BreakerState.CLOSED:
            return True
        if state is BreakerState.OPEN:
            return False
        if self._probe_in_flight:
            return False
        self._probe_in_flight = True
        return True

    def record_success(self) -> None:
        """Close the breaker and clear the failure count."""
        self._state = BreakerState.CLOSED
        self._failures = 0
        self._probe_in_flight = False

    def record_failure(self) -> None:
        """Count a failure; reopen at once if this was the half-open probe."""
        if self._promote() is BreakerState.HALF_OPEN:
            self._trip()
            return
        self._failures += 1
        if self._failures >= self._threshold:
            self._trip()

    def release_probe(self) -> None:
        """Free the half-open probe slot after a cancelled request."""
        self._probe_in_flight = False

    def _promote(self) -> BreakerState:
        """Promote an open breaker to half-open once its timeout has passed."""
        if (
            self._state is BreakerState.OPEN
            and self._clock() - self._opened_at >= self._recovery
        ):
            self._state = BreakerState.HALF_OPEN
            self._probe_in_flight = False
        return self._state

    def _trip(self) -> None:
        self._state = BreakerState.OPEN
        self._opened_at = self._clock()
        self._probe_in_flight = False


class RateLimiter:
    """Limit request starts to a fixed number in each rolling minute."""

    def __init__(
        self,
        rpm: int,
        *,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        """Create a limiter.

        Args:
            rpm: Maximum request starts in any rolling 60-second window.
            clock: Monotonic clock used to timestamp request starts.
            sleep: Async wait function used when the window is full.

        Raises:
            ValueError: If ``rpm`` is not positive.
        """
        if rpm <= 0:
            raise ValueError("requests per minute must be positive")
        self._rpm = rpm
        self._clock = clock
        self._sleep = sleep
        self._stamps: deque[float] = deque()
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        """Wait until one request start fits in the rolling window."""
        while True:
            async with self._lock:
                now = self._clock()
                while self._stamps and self._stamps[0] <= now - 60.0:
                    self._stamps.popleft()
                if len(self._stamps) < self._rpm:
                    self._stamps.append(now)
                    return
                delay = self._stamps[0] + 60.0 - now
            await self._sleep(delay)


class Kind(StrEnum):
    """Retry classification for a provider failure."""

    TERMINAL = "terminal"
    TRANSIENT = "transient"


def provider_limits(
    settings: ProviderResilienceSettings,
    *,
    breaker: CircuitBreaker | None,
    limiter: RateLimiter | None,
    sleep: Callable[[float], Awaitable[None]],
) -> tuple[CircuitBreaker | None, RateLimiter | None]:
    """Build enabled protections, preserving explicitly supplied overrides."""
    if breaker is None and settings.breaker_threshold is not None:
        breaker = CircuitBreaker(
            settings.breaker_threshold, settings.breaker_recovery_s
        )
    if limiter is None and settings.requests_per_minute is not None:
        limiter = RateLimiter(settings.requests_per_minute, sleep=sleep)
    return breaker, limiter


async def call_with_breaker(
    call: Callable[[], Awaitable[T]],
    breaker: CircuitBreaker | None,
    classify_error: Callable[[BaseException], Kind | None],
) -> T:
    """Record an admitted call's outcome, then return or re-raise unchanged.

    Transient failures count against the breaker. Terminal failures prove the
    provider answered. Unclassified errors and cancellation release the probe.
    Admission and retry policy remain the caller's responsibility.
    """
    try:
        result = await call()
    except BaseException as exc:
        if breaker is not None:
            kind = classify_error(exc)
            if kind is Kind.TRANSIENT:
                breaker.record_failure()
            elif kind is Kind.TERMINAL:
                breaker.record_success()
            else:
                breaker.release_probe()
        raise
    if breaker is not None:
        breaker.record_success()
    return result


TERMINAL_PATTERNS = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"\b(?:400|401|402|403|404|422)\b",
        r"insufficient quota",
        r"insufficient_quota",
        r"quota exceeded",
        r"quota_exceeded",
        r"billing",
        r"credit balance",
        r"out of credits",
        r"invalid api key",
        r"unauthorized",
        r"unauthorised",
        r"permission denied",
        r"authentication",
        r"api_key is not set",
        r"content policy",
        r"context length",
        r"too large",
    )
)
"""Message patterns that mark a request as non-retryable."""


def classify(exc: BaseException) -> Kind:
    """Classify an error by checking its message and cause chain.

    Args:
        exc: Provider failure to classify.

    Returns:
        ``TERMINAL`` when any cause contains a terminal pattern, otherwise
        ``TRANSIENT``.
    """
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if any(pattern.search(str(current)) for pattern in TERMINAL_PATTERNS):
            return Kind.TERMINAL
        current = current.__cause__
    return Kind.TRANSIENT


def backoff_delay(
    attempt: int,
    *,
    base_s: float,
    max_s: float,
    rng: random.Random,
) -> float:
    """Return an equal-jitter delay bounded by the exponential backoff cap."""
    capped = min(max_s, base_s * 2**attempt)
    return capped * (0.5 + 0.5 * rng.random())
