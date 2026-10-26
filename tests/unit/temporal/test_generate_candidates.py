"""Tests for choosing the pairs of facts worth judging."""

from ontogr_mem.data_models.fact_versions import FactStatus
from ontogr_mem.temporal.candidates import generate_candidates
from ontogr_mem.temporal.cardinality import (
    Cardinality,
    ConflictPolicy,
    DictRelationPolicy,
    RelationRules,
)
from tests.unit.temporal.helpers import at, fact_of, obs


POLICY = DictRelationPolicy(
    {
        9: RelationRules(Cardinality.MANY, ConflictPolicy.IGNORE),
    }
)


def fact(fact_id: str, slot: str, *, rel: int = 1, status=FactStatus.ACTIVE):
    """A fact of ana with ``acme`` as the object, starting in 2010."""
    observation = obs(f"o-{fact_id}", "acme", at(2010), slot=slot, rel=rel)
    return fact_of(fact_id, [observation], slot=slot, status=status)


def test_facts_in_different_slots_that_share_an_entity_form_one_pair():
    """Two facts about the same subject in two slots are one candidate pair."""
    candidates, cut = generate_candidates([fact("f1", "s1"), fact("f2", "s2")], POLICY)

    assert [(c.a.fact_id, c.b.fact_id) for c in candidates] == [("f1", "f2")]
    assert (cut, candidates[0].shared) == (0, 2)


def test_facts_in_the_same_slot_are_not_paired():
    """The slot's own rule handles its values; it is not a cross-slot candidate."""
    candidates, _ = generate_candidates([fact("f1", "s1"), fact("f2", "s1")], POLICY)

    assert candidates == []


def test_corrected_facts_take_no_part_in_candidates():
    """A fact already judged wrong is not asked about again."""
    candidates, _ = generate_candidates(
        [fact("f1", "s1"), fact("f2", "s2", status=FactStatus.CORRECTED)],
        POLICY,
    )

    assert candidates == []


def test_relations_that_ignore_conflicts_are_not_paired():
    """A relation that opts out of conflict handling is left out."""
    candidates, _ = generate_candidates(
        [fact("f1", "s1", rel=9), fact("f2", "s2", rel=9)], POLICY
    )

    assert candidates == []


def test_the_cap_keeps_the_first_pairs_and_counts_the_rest():
    """Pairs past the cap are dropped and the number is reported."""
    facts = [fact("f1", "s1"), fact("f2", "s2"), fact("f3", "s3")]

    candidates, cut = generate_candidates(facts, POLICY, cap=2)

    assert len(candidates) == 2
    assert cut == 1
