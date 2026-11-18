"""Tests for modality filtering in temporal queries."""

from dataclasses import replace
from datetime import UTC, datetime

from ontogr_mem.data_models.facts import FactKind, Modality
from ontogr_mem.temporal.query import ACTUAL_ONLY, TemporalQuery, matches
from tests.unit.temporal.helpers import at, fact_of, obs


def test_actual_only_hides_plans_but_includes_events():
    """Actual-only reads include events and exclude planned facts."""
    base = fact_of("f1", [obs("o1", "acme", at(2020))])
    planned = replace(base, modality=Modality.PLANNED)
    event = replace(base, fact_id="event", fact_kind=FactKind.EVENT)
    query = TemporalQuery(now=datetime(2026, 1, 1, tzinfo=UTC), modalities=ACTUAL_ONLY)

    assert matches(query, base)
    assert matches(query, event)
    assert not matches(query, planned)


def test_unspecified_modalities_match_all_modalities():
    """An absent modality filter leaves the query unrestricted."""
    base = fact_of("f1", [obs("o1", "acme", at(2020))])
    planned = replace(base, modality=Modality.PLANNED)
    query = TemporalQuery(now=datetime(2026, 1, 1, tzinfo=UTC))

    assert matches(query, planned)
