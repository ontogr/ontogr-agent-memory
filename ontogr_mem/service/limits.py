"""Size limits and the in-memory token-bucket rate limiter."""

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Final, Literal

from ontogr_mem.service.errors import RateLimitedError


MAX_CONTENT_BYTES: Final = 64 * 1024
"""Most UTF-8 bytes one content field may hold."""

MAX_RESULTS: Final = 100
"""Most results a read request may ask for."""

WRITES_PER_MINUTE: Final = 60
"""Default write tokens per caller per minute."""

READS_PER_MINUTE: Final = 600
"""Default read tokens per caller per minute."""

LimitKind = Literal["read", "write"]


@dataclass
class _Bucket:
    """One caller's tokens and the time they were last refilled."""

    tokens: float
    stamp: float


class CallerRateLimiter:
    """Token bucket per (caller, kind): a read and a write bucket per caller.

    A new bucket starts full. It refills at the per-minute rate divided by 60
    per second and never holds more than one minute of tokens.

    Args:
        writes_per_minute: Write tokens per caller per minute.
        reads_per_minute: Read tokens per caller per minute.
        clock: Seconds source; tests pass a plain function instead of the
            monotonic clock.

    Raises:
        RateLimitedError: From ``check`` when the caller has no token left.
    """

    def __init__(
        self,
        *,
        writes_per_minute: int = WRITES_PER_MINUTE,
        reads_per_minute: int = READS_PER_MINUTE,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        """Set the two rates and the clock."""
        self._writes_per_minute = writes_per_minute
        self._reads_per_minute = reads_per_minute
        self._clock = clock
        # Buckets are never evicted, so memory grows with the number of callers.
        # The limits apply per process only.
        self._buckets: dict[tuple[str, LimitKind], _Bucket] = {}

    def check(self, caller_id: str, kind: LimitKind) -> None:
        """Take one token for the caller, or refuse with ``rate_limited``.

        Args:
            caller_id: Id of the calling agent, person or service.
            kind: Which bucket to spend, ``read`` or ``write``.

        Raises:
            RateLimitedError: If the caller's bucket for the kind is empty.
        """
        per_minute = (
            self._writes_per_minute if kind == "write" else self._reads_per_minute
        )
        now = self._clock()
        bucket = self._buckets.setdefault(
            (caller_id, kind), _Bucket(float(per_minute), now)
        )
        bucket.tokens = min(
            float(per_minute),
            bucket.tokens + (now - bucket.stamp) * per_minute / 60.0,
        )
        bucket.stamp = now
        if bucket.tokens < 1.0:
            msg = f"more than {per_minute} {kind}s per minute"
            raise RateLimitedError(msg, hint="Wait a few seconds and retry.")
        bucket.tokens -= 1.0
