"""Importance terms: recency, frequency, salience and their weighted sum."""

import math
from datetime import UTC, datetime, timedelta

import pytest

from ontogr_mem.consolidation.config import ScoringSettings
from ontogr_mem.consolidation.importance import (
    frequency,
    importance,
    recency,
    salience,
)


NOW = datetime(2030, 1, 1, tzinfo=UTC)


def test_recency_halves_after_each_half_life() -> None:
    """Recency halves after each half-life."""
    assert recency(NOW, NOW, 30) == pytest.approx(1.0)
    assert recency(NOW - timedelta(days=30), NOW, 30) == pytest.approx(0.5)
    assert recency(NOW - timedelta(days=60), NOW, 30) == pytest.approx(0.25)


def test_recency_of_a_future_timestamp_is_one() -> None:
    """A timestamp in the future counts as fresh."""
    assert recency(NOW + timedelta(days=5), NOW, 30) == pytest.approx(1.0)


def test_frequency_is_zero_for_no_access_and_one_at_the_cap() -> None:
    """Frequency is zero without access and one at the cap."""
    assert frequency(0, 20) == 0.0
    assert frequency(20, 20) == pytest.approx(1.0)
    assert frequency(500, 20) == 1.0
    assert frequency(5, 20) == pytest.approx(math.log1p(5) / math.log1p(20))


def test_salience_mixes_capped_degree_and_feedback_equally() -> None:
    """Salience weighs capped degree and feedback equally."""
    assert salience(10, 0.5, 10) == pytest.approx(0.75)
    assert salience(40, 1.0, 10) == pytest.approx(1.0)
    assert salience(0, 0.0, 10) == 0.0


def test_importance_is_the_weighted_sum_of_its_terms() -> None:
    """Importance is the weighted sum of its three terms."""
    cfg = ScoringSettings(w_recency=0.5, w_frequency=0.25, w_salience=0.25)

    result = importance(
        last_seen=NOW,
        now=NOW,
        count=20,
        degree=10,
        feedback=1.0,
        cfg=cfg,
    )

    assert result.recency == pytest.approx(1.0)
    assert result.frequency == pytest.approx(1.0)
    assert result.salience == pytest.approx(1.0)
    assert result.score == pytest.approx(1.0)


def test_a_recent_item_outranks_an_old_one_when_all_else_is_equal() -> None:
    """A recent item outranks an old one when the rest is equal."""
    cfg = ScoringSettings()
    recent = importance(
        last_seen=NOW, now=NOW, count=3, degree=2, feedback=0.5, cfg=cfg
    )
    old = importance(
        last_seen=NOW - timedelta(days=365),
        now=NOW,
        count=3,
        degree=2,
        feedback=0.5,
        cfg=cfg,
    )

    assert recent.score > old.score
