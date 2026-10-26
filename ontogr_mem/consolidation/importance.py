"""Importance score: recency, frequency and salience, each in [0, 1]."""

import math
from datetime import datetime
from typing import NamedTuple

from ontogr_mem.consolidation.config import ScoringSettings


class Importance(NamedTuple):
    """The score and the three terms it is made of."""

    score: float
    recency: float
    frequency: float
    salience: float


def recency(last_seen: datetime, now: datetime, half_life_days: float) -> float:
    """Return exponential decay: 1.0 at ``now``, 0.5 after one half-life."""
    age_days = max(0.0, (now - last_seen).total_seconds() / 86400.0)
    return float(0.5 ** (age_days / half_life_days))


def frequency(count: int, cap: int) -> float:
    """Return a log-scaled count that reaches 1.0 at ``cap`` and stays there."""
    if count <= 0:
        return 0.0
    return min(1.0, math.log1p(count) / math.log1p(cap))


def salience(degree: int, feedback: float, degree_cap: int) -> float:
    """Return half of the capped graph degree and half of the feedback weight."""
    return 0.5 * min(1.0, degree / degree_cap) + 0.5 * feedback


def importance(
    *,
    last_seen: datetime,
    now: datetime,
    count: int,
    degree: int,
    feedback: float,
    cfg: ScoringSettings,
) -> Importance:
    """Return the weighted sum of the three terms for one item."""
    r = recency(last_seen, now, cfg.half_life_days)
    f = frequency(count, cfg.frequency_cap)
    s = salience(degree, feedback, cfg.degree_cap)
    total = cfg.w_recency * r + cfg.w_frequency * f + cfg.w_salience * s
    return Importance(total, r, f, s)
