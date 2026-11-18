"""Behavior of ontology label and family indexes."""

from ontogr_mem.ontology.links import TypeLinks
from ontogr_mem.ontology.models import Entity, Ontology, Relation


def test_entity_family_includes_descendants_and_merged_ids():
    """Include descendants and merged ids in a resolved entity family."""
    ontology = Ontology(
        entities=[
            Entity(type_id=0, label="Unknown"),
            Entity(type_id=1, label="Person", aliases=["Human"]),
            Entity(type_id=2, label="Employee", parent_id=1),
            Entity(type_id=3, label="Old Person", merged_into=1),
        ],
        relations=[Relation(type_id=0, label="KNOWS", aliases=["Familiar"])],
    )
    links = TypeLinks.from_ontology(ontology)
    assert links.entity_family(" human ") == frozenset({1, 2, 3})
    assert links.entity_family("Unknown") == frozenset({0})


def test_ambiguous_alias_is_not_resolved():
    """Leave labels unresolved when aliases identify multiple types."""
    ontology = Ontology(
        entities=[
            Entity(type_id=0, label="Unknown"),
            Entity(type_id=1, label="Person", aliases=["Shared"]),
            Entity(type_id=2, label="Company", aliases=["Shared"]),
        ],
        relations=[
            Relation(type_id=0, label="KNOWS", aliases=["Shared"]),
            Relation(type_id=1, label="OWNS", aliases=["Shared"]),
        ],
    )
    links = TypeLinks.from_ontology(ontology)
    assert links.entity_family("Shared") is None
    assert links.relation_family("Shared") is None
