"""Fact extraction prompt guidance."""

from ontogr_mem.data_models import TypeView
from ontogr_mem.fact_extraction.prompts import FACT_RULES, relation_section
from tests.support.facts import fact_catalog


class TestRelationSection:
    """Relation guidance for the model."""

    def test_relation_section_allows_unplaced_relations(self):
        """Unknown or unsuitable relations can be emitted as text labels."""
        section = relation_section(TypeView(fact_catalog()))

        assert section[0] == (
            "Relation types: set relation_type_id to one of these ids when a type "
            "fits, between two refs in the listed direction. When none fits, set "
            "relation_type_id to null and write the relation in relation_label."
        )

    def test_fact_system_rule_allows_null_relation_ids_with_words(self):
        """The model-facing fact rules permit a textual relation without an id."""
        assert (
            '"relation_type_id" is set to a supplied id when a type fits a link '
            "between exactly two refs (source first); otherwise null."
        ) in FACT_RULES
        assert (
            "When a link has exactly two refs and no type fits, write its relation "
            'in "relation_label".'
        ) in FACT_RULES
