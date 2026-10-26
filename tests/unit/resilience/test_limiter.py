"""Sliding-window request limiting with a controllable clock."""

import pytest

from ontogr_mem.resilience import RateLimiter


class FakeTime:
    """A clock and sleep pair that advances only when the limiter sleeps."""

    def __init__(self) -> None:
        """Start at zero with no recorded waits."""
        self.now = 0.0
        self.delays: list[float] = []

    def clock(self) -> float:
        """Return the current fake time."""
        return self.now

    async def sleep(self, seconds: float) -> None:
        """Advance fake time by the requested delay."""
        self.delays.append(seconds)
        self.now += seconds


class TestRateLimiter:
    """A request limit counts calls in a rolling 60-second window."""

    async def test_third_call_waits_for_the_oldest_stamp_to_expire(self) -> None:
        """The oldest call frees one slot after exactly 60 seconds."""
        time = FakeTime()
        limiter = RateLimiter(2, clock=time.clock, sleep=time.sleep)

        await limiter.acquire()
        await limiter.acquire()
        await limiter.acquire()

        assert time.delays == [60.0]
        assert time.now == 60.0

    async def test_stamps_older_than_sixty_seconds_do_not_count(self) -> None:
        """A request at the window boundary does not wait."""
        time = FakeTime()
        limiter = RateLimiter(1, clock=time.clock, sleep=time.sleep)

        await limiter.acquire()
        time.now = 60.1
        await limiter.acquire()

        assert time.delays == []

    def test_zero_requests_per_minute_is_refused(self) -> None:
        """A rate limit must permit at least one request per minute."""
        with pytest.raises(ValueError, match="positive"):
            RateLimiter(0, clock=lambda: 0.0, sleep=FakeTime().sleep)
