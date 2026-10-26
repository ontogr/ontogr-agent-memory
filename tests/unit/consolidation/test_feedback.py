"""Feedback weight arithmetic: the streaming update, decay and the neutral default."""

from datetime import UTC, datetime, timedelta

import pytest

from ontogr_mem.consolidation.feedback import (
    NEUTRAL,
    effective_weight,
    stream_update_weight,
    weight_now,
)
from ontogr_mem.consolidation.models import FeedbackWeight


T0 = datetime(2030, 1, 1, tzinfo=UTC)


@pytest.mark.parametrize(
    ("prev", "rating", "alpha", "expected"),
    [
        (0.5, 1.0, 0.1, 0.55),
        (0.5, 0.0, 0.1, 0.45),
        (0.9, 0.9, 0.1, 0.9),
        (0.5, 1.0, 2.0, 1.0),
        (0.5, 0.0, 2.0, 0.0),
    ],
)
def test_stream_update_moves_by_alpha_of_the_gap_and_stays_in_bounds(
    prev: float, rating: float, alpha: float, expected: float
) -> None:
    """The update moves by alpha of the gap and stays inside [0, 1]."""
    assert stream_update_weight(prev, rating, alpha) == pytest.approx(expected)


@pytest.mark.parametrize(
    ("weight", "idle_days", "beta", "expected"),
    [
        (1.0, 0.0, 0.02, 1.0),
        (1.0, 1.0, 0.5, 0.75),
        (0.0, 1.0, 0.5, 0.25),
        (1.0, -5.0, 0.5, 1.0),
    ],
)
def test_decay_pulls_toward_neutral_and_never_runs_backwards(
    weight: float, idle_days: float, beta: float, expected: float
) -> None:
    """Decay pulls toward neutral and ignores negative idle time."""
    assert effective_weight(weight, idle_days, beta) == pytest.approx(expected)


def test_weight_now_is_neutral_without_feedback() -> None:
    """A weight with no feedback reads as neutral."""
    assert weight_now(None, T0, 0.02) == NEUTRAL


def test_weight_now_decays_by_the_days_since_the_last_update() -> None:
    """Decay counts the days since the last update."""
    weight = FeedbackWeight(target_id="x", weight=1.0, last_update_at=T0)

    decayed = weight_now(weight, T0 + timedelta(days=1), 0.5)

    assert decayed == pytest.approx(0.75)
