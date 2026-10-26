"""Tests for the size constants and the token-bucket rate limiter."""

import pytest

from ontogr_mem.service.errors import RateLimitedError
from ontogr_mem.service.limits import (
    MAX_CONTENT_BYTES,
    MAX_RESULTS,
    CallerRateLimiter,
)


class Clock:
    """A clock the test sets; the limiter reads it through this seam."""

    def __init__(self) -> None:
        """Start at zero seconds."""
        self.now = 0.0

    def advance(self, seconds: float) -> None:
        """Move the clock forward by the given seconds."""
        self.now += seconds

    def __call__(self) -> float:
        """Return the current value in seconds."""
        return self.now


class TestCallerRateLimiter:
    """Token-bucket behavior per caller and per kind."""

    @pytest.mark.parametrize(("kind", "per_minute"), [("read", 3), ("write", 2)])
    def test_allows_up_to_the_per_minute_rate_then_refuses(self, kind, per_minute):
        """A caller gets one token per allowed call, then a typed refusal."""
        limiter = CallerRateLimiter(
            writes_per_minute=2,
            reads_per_minute=3,
            clock=Clock(),
        )

        for _ in range(per_minute):
            limiter.check("caller", kind)

        with pytest.raises(RateLimitedError) as excinfo:
            limiter.check("caller", kind)

        assert str(per_minute) in str(excinfo.value)
        assert kind in str(excinfo.value)

    def test_refills_one_token_per_interval(self):
        """One minute divided by the rate gives back one token."""
        clock = Clock()
        limiter = CallerRateLimiter(reads_per_minute=60, clock=clock)

        for _ in range(60):
            limiter.check("caller", "read")
        clock.advance(1.0)

        limiter.check("caller", "read")
        with pytest.raises(RateLimitedError):
            limiter.check("caller", "read")

    def test_refill_never_exceeds_capacity(self):
        """A long idle period restores the bucket to capacity, not more."""
        clock = Clock()
        limiter = CallerRateLimiter(reads_per_minute=5, clock=clock)

        for _ in range(5):
            limiter.check("caller", "read")
        clock.advance(10_000.0)

        for _ in range(5):
            limiter.check("caller", "read")
        with pytest.raises(RateLimitedError):
            limiter.check("caller", "read")

    def test_callers_have_separate_buckets(self):
        """A drained caller does not spend another caller's tokens."""
        limiter = CallerRateLimiter(writes_per_minute=1, clock=Clock())

        limiter.check("a", "write")
        with pytest.raises(RateLimitedError):
            limiter.check("a", "write")

        limiter.check("b", "write")

    def test_read_and_write_buckets_are_separate(self):
        """Draining the write bucket leaves the read bucket full."""
        limiter = CallerRateLimiter(writes_per_minute=1, clock=Clock())

        limiter.check("caller", "write")
        with pytest.raises(RateLimitedError):
            limiter.check("caller", "write")

        limiter.check("caller", "read")

    def test_default_rates_are_60_writes_and_600_reads(self):
        """The default constructor allows 60 writes and 600 reads per minute."""
        limiter = CallerRateLimiter(clock=Clock())

        for _ in range(600):
            limiter.check("caller", "read")
        with pytest.raises(RateLimitedError):
            limiter.check("caller", "read")

        for _ in range(60):
            limiter.check("caller", "write")
        with pytest.raises(RateLimitedError):
            limiter.check("caller", "write")


class TestSizeLimits:
    """The published size constants of the wire contract."""

    def test_constants_match_the_reference_limits(self):
        """The byte cap and the result cap are the published values."""
        assert MAX_CONTENT_BYTES == 65536
        assert MAX_RESULTS == 100
