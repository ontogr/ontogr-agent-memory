"""Feedback weights: streaming update, and decay toward neutral on read.

The update is ``w + alpha * (rating - w)`` clipped to [0, 1]. Decay counts days since
the last update on the injected clock, not conversation turns.
"""

from datetime import datetime

from ontogr_mem.consolidation.models import FeedbackWeight


NEUTRAL = 0.5


def stream_update_weight(prev: float, rating: float, alpha: float) -> float:
    """Move ``prev`` toward ``rating`` by ``alpha`` of the gap, kept inside [0, 1]."""
    return max(0.0, min(1.0, prev + alpha * (rating - prev)))


def effective_weight(
    weight: float, idle_days: float, beta: float, neutral: float = NEUTRAL
) -> float:
    """Pull ``weight`` toward ``neutral`` by ``(1 - beta) ** idle_days``."""
    return neutral + (weight - neutral) * float((1.0 - beta) ** max(0.0, idle_days))


def weight_now(weight: FeedbackWeight | None, now: datetime, beta: float) -> float:
    """Return the decayed weight of an item; neutral when it has no feedback."""
    if weight is None:
        return NEUTRAL
    idle = (now - weight.last_update_at).total_seconds() / 86400.0
    return effective_weight(weight.weight, idle, beta)
