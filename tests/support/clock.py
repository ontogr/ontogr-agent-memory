"""Deterministic clocks for time-based tests."""

from datetime import UTC, datetime, timedelta


class FixedClock:
    """A `Clock` that always returns the same time."""

    def __init__(self, value: datetime | None = None) -> None:
        """Fix the clock at `value`, 2026-10-21 09:00 UTC by default."""
        self.value = value or datetime(2026, 10, 21, 9, 0, tzinfo=UTC)

    def now(self) -> datetime:
        """Return the fixed time."""
        return self.value

    def advance(self, *, seconds: float) -> None:
        """Move the fixed time forward by the requested number of seconds."""
        self.value += timedelta(seconds=seconds)


class MovingClock:
    """A test clock that advances only when requested."""

    def __init__(self, value: datetime | None = None) -> None:
        """Start at ``value`` or 2026-10-21 09:00 UTC by default."""
        self.value = value or datetime(2026, 10, 21, 9, 0, tzinfo=UTC)

    def now(self) -> datetime:
        """Return the current test time."""
        return self.value

    def advance(self, delta: timedelta) -> None:
        """Move the test time forward by ``delta``."""
        self.value += delta
