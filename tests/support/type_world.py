"""A small ontology with a hierarchy for tests of type compatibility."""

from dataclasses import dataclass

from ontogr_mem.ontology.declaration import EntityDeclaration, OntologyDeclaration
from ontogr_mem.ontology.declare import build_target
from ontogr_mem.ontology.links import TypeLinks
from ontogr_mem.ontology.models import Ontology


@dataclass(frozen=True, slots=True)
class TypeWorld:
    """An ontology, its links, and the id of each label."""

    ontology: Ontology
    links: TypeLinks
    ids: dict[str, int]


def type_world(*types: tuple[str, str | None, tuple[str, ...]]) -> TypeWorld:
    """Build a world from (label, parent label, identity names) rows."""
    declaration = OntologyDeclaration(
        entities=[
            EntityDeclaration(
                label=label, parent=parent, identity_attributes=list(names)
            )
            for label, parent, names in types
        ]
    )
    ontology = build_target(Ontology.genesis(), declaration)
    ids = {entity.label: entity.type_id for entity in ontology.entities}
    return TypeWorld(ontology, TypeLinks.from_ontology(ontology), ids)


WORLD = type_world(
    ("Person", None, ("email",)),
    ("Company", None, ()),
    ("City", None, ()),
    ("Widget", None, ()),
    ("Contract", None, ()),
    ("PurchaseOrder", "Contract", ()),
    ("Employee", "Person", ()),
    ("Language", None, ()),
    ("Work", None, ()),
)
