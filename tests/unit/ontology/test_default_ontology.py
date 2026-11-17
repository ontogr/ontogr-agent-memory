"""Tests for the ontology bundled with the package."""

from difflib import SequenceMatcher
from itertools import combinations

from ontogr_mem.ontology import (
    Ontology,
    RelationKind,
    apply_changeset,
    default_ontology,
)
from ontogr_mem.ontology.declare import build_target
from ontogr_mem.ontology.diff import diff
from ontogr_mem.ontology.names import normalize


class TestDefaultOntology:
    """The default declaration is valid and has a distinct vocabulary."""

    def test_the_default_ontology_applies_to_a_new_tenant(self):
        """A new tenant receives the bundled vocabulary plus Unknown."""
        head = Ontology.genesis()
        target = build_target(head, default_ontology())
        applied = apply_changeset(head, diff(head, target))
        assert len(applied.entities) == 11
        assert len(applied.relations) == 31

    def test_default_relation_kinds(self):
        """Only the four single-valued relations and the two judged ones vary."""
        by_kind = {}
        for relation in default_ontology().relations:
            by_kind.setdefault(relation.kind, set()).add(relation.label)
        assert by_kind[RelationKind.SINGLE] == {
            "WORKS_AT",
            "LIVES_IN",
            "MARRIED_TO",
            "LOCATED_IN",
        }
        assert by_kind[RelationKind.JUDGED] == {"PREFERS", "USES"}
        assert len(by_kind[RelationKind.MANY]) == 25

    def test_no_two_default_names_are_close_enough_to_be_ambiguous(self):
        """Names of distinct types stay below the fuzzy matching threshold."""
        declaration = default_ontology()
        for declared_types in (declaration.entities, declaration.relations):
            for left, right in combinations(declared_types, 2):
                for left_name in [left.label, *left.aliases]:
                    for right_name in [right.label, *right.aliases]:
                        ratio = SequenceMatcher(
                            None, normalize(left_name), normalize(right_name)
                        ).ratio()
                        assert ratio < 0.8, (left_name, right_name)

    def test_the_default_has_no_catch_all_relation(self):
        """No relation accepts an unspecified relationship by name."""
        assert all(
            relation.label != "RELATED_TO" for relation in default_ontology().relations
        )
