"""Cardinality policy behavior for temporal facts."""

from ontogr_mem.data_models.facts import FactKind, Modality
from ontogr_mem.temporal.cardinality import (
    Cardinality,
    DictRelationPolicy,
    RelationRules,
    rules_for,
)


class TestRulesFor:
    """Select rules based on fact placement and modality."""

    def test_returns_many_rules_for_unplaced_facts(self):
        """Treat unplaced facts as multi-valued."""
        policy = DictRelationPolicy({1: RelationRules(Cardinality.ONE)})

        rules = rules_for(policy, None, Modality.ACTUAL, FactKind.STATE)

        assert rules.cardinality is Cardinality.MANY

    def test_returns_many_rules_for_non_actual_facts(self):
        """Treat non-actual facts as multi-valued."""
        policy = DictRelationPolicy({1: RelationRules(Cardinality.ONE)})

        rules = rules_for(policy, 1, Modality.PLANNED, FactKind.STATE)

        assert rules.cardinality is Cardinality.MANY

    def test_returns_many_rules_for_events(self):
        """Treat events as multi-valued."""
        policy = DictRelationPolicy({1: RelationRules(Cardinality.ONE)})

        rules = rules_for(policy, 1, Modality.ACTUAL, FactKind.EVENT)

        assert rules.cardinality is Cardinality.MANY

    def test_returns_policy_rules_for_actual_placed_states(self):
        """Use declared rules for actual placed state facts."""
        expected = RelationRules(Cardinality.ONE)
        policy = DictRelationPolicy({1: expected})

        rules = rules_for(policy, 1, Modality.ACTUAL, FactKind.STATE)

        assert rules is expected
