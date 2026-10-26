"""Tests for the circuit breaker."""

from ontogr_mem.resilience import BreakerState, CircuitBreaker


class Clock:
    """A clock the test moves by hand."""

    def __init__(self) -> None:
        """Start at time zero."""
        self.now = 0.0

    def __call__(self) -> float:
        """Return the current time."""
        return self.now


class TestCircuitBreaker:
    """Behavior of CircuitBreaker."""

    def test_opens_after_threshold_and_half_opens_after_timeout(self):
        """Consecutive failures open it; after the timeout one probe is allowed."""
        clock = Clock()
        breaker = CircuitBreaker(3, 30, clock)

        for _ in range(3):
            assert breaker.allow_request()
            breaker.record_failure()

        assert breaker.state is BreakerState.OPEN
        assert not breaker.allow_request()
        clock.now = 31
        assert breaker.state is BreakerState.HALF_OPEN
        assert breaker.allow_request()
        assert not breaker.allow_request()

    def test_probe_success_closes_and_probe_failure_reopens(self):
        """The half-open probe decides the next state."""
        clock = Clock()
        breaker = CircuitBreaker(1, 10, clock)
        breaker.record_failure()
        clock.now = 11
        assert breaker.allow_request()

        breaker.record_failure()

        assert breaker.state is BreakerState.OPEN
        clock.now = 22
        assert breaker.allow_request()
        breaker.record_success()
        assert breaker.state is BreakerState.CLOSED
        assert breaker.allow_request()

    def test_released_probe_can_be_taken_again(self):
        """A cancelled probe frees its slot."""
        clock = Clock()
        breaker = CircuitBreaker(1, 10, clock)
        breaker.record_failure()
        clock.now = 11
        assert breaker.allow_request()

        breaker.release_probe()

        assert breaker.allow_request()
