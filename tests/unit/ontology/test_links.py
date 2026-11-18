"""Tests for type compatibility derived from an ontology."""

from ontogr_mem.ontology.links import TypeLinks, TypeRel
from ontogr_mem.ontology.models import UNKNOWN_ID, Entity, Ontology


class TestTypeLinks:
    """Compatibility and identity names follow the ontology hierarchy."""

    def test_relations_follow_merges_and_distinguish_siblings(self):
        """Merged types compare equal and unrelated siblings stay distinct."""
        ontology = Ontology(
            entities=[
                Entity(type_id=UNKNOWN_ID, label="Unknown"),
                Entity(type_id=1, label="Person"),
                Entity(type_id=2, label="Employee", parent_id=1),
                Entity(type_id=3, label="Contractor", parent_id=1),
                Entity(type_id=4, label="Human", merged_into=1),
            ]
        )

        links = TypeLinks.from_ontology(ontology)

        assert links.canonical(4) == 1
        assert links.label(4) == "Person"
        assert links.relation(4, 1) is TypeRel.SAME
        assert links.relation(UNKNOWN_ID, 2) is TypeRel.NESTED
        assert links.relation(1, 2) is TypeRel.NESTED
        assert links.relation(2, 3) is TypeRel.DISTINCT

    def test_more_specific_and_identity_names_include_ancestors(self):
        """A child inherits the identity names of its ancestors."""
        ontology = Ontology(
            entities=[
                Entity(type_id=UNKNOWN_ID, label="Unknown"),
                Entity(type_id=1, label="Person", identity_attributes=["email"]),
                Entity(type_id=2, label="Employee", parent_id=1),
                Entity(type_id=3, label="Organization"),
            ]
        )

        links = TypeLinks.from_ontology(ontology)

        assert links.more_specific(UNKNOWN_ID, 2) == 2
        assert links.more_specific(1, 2) == 2
        assert links.more_specific(2, 3) == 2
        assert links.identity_names(2) == ("email",)
        assert links.identity_names(3) == ()

    def test_unknown_ids_have_unknown_label_and_no_identity_names(self):
        """An id outside the ontology has no label and no identity names."""
        links = TypeLinks.from_ontology(Ontology.genesis())

        assert links.label(99) == "Unknown"
        assert links.identity_names(99) == ()
