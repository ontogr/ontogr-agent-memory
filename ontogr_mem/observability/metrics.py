"""Plain counters. The store increments them. Exporting them is up to the caller."""

from collections import Counter, defaultdict


class Metrics:
    """Counters and latency sample counts keyed by name.

    Attributes:
        counters: Counts accumulated by metric name.
    """

    def __init__(self) -> None:
        """Start with every counter at zero."""
        self.counters: Counter[str] = Counter()
        self._latency_n: defaultdict[str, int] = defaultdict(int)

    def __repr__(self) -> str:
        """Return the counters and the number of latency samples."""
        return (
            f"Metrics(counters={dict(self.counters)}, samples={dict(self._latency_n)})"
        )

    def inc(self, name: str, n: int = 1) -> None:
        """Add `n` to counter `name`."""
        self.counters[name] += n

    def observe(self, name: str, seconds: float) -> None:
        """Record one latency sample."""
        self._latency_n[name] += 1
