"""Tests for trust ranks and the margin that permits a preference."""

import pytest

from ontogr_mem.data_models.observation import Observation
from ontogr_mem.temporal.trust import Trust, TrustRule
from tests.unit.temporal.helpers import at, obs


def sourced(obs_id: str, *, system: str = "", connector: str = "") -> Observation:
    """An observation from a named source system and connector."""
    base = obs(obs_id, "acme", at(2020))
    return Observation(
        **{
            **{name: getattr(base, name) for name in base.__slots__},
            "source_system": system,
            "connector": connector,
        }
    )


def test_the_best_matching_rule_gives_the_rank():
    """Among matching rules the highest rank wins, whatever the rule order."""
    trust = Trust(
        rules=(
            TrustRule(rank=1, system="crm"),
            TrustRule(rank=3, system="crm", connector="api"),
        )
    )

    assert trust.rank_of(sourced("o1", system="crm", connector="api")) == 3
    assert trust.rank_of(sourced("o2", system="crm")) == 1


def test_no_matching_rule_gives_rank_zero():
    """A source no rule names is the lowest rank."""
    trust = Trust(rules=(TrustRule(rank=5, system="crm"),))

    assert trust.rank_of(sourced("o1", system="email")) == 0


def test_a_fact_takes_the_best_rank_of_its_evidence():
    """Corroboration from a trusted source raises the fact's rank."""
    trust = Trust(rules=(TrustRule(rank=2, system="crm"),))
    evidence = [sourced("o1", system="email"), sourced("o2", system="crm")]

    assert trust.fact_rank(evidence) == 2


@pytest.mark.parametrize("margin", [0, -1])
def test_a_margin_below_one_is_refused(margin):
    """A margin of zero would let a tie count as a preference."""
    with pytest.raises(ValueError, match="margin must be at least 1"):
        Trust(margin=margin)
