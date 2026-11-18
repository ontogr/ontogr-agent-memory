"""Identity metadata in ontology projections."""

from ontogr_mem.ontology.apply import apply_changeset
from ontogr_mem.ontology.declaration import EntityDeclaration, OntologyDeclaration
from ontogr_mem.ontology.declare import build_target
from ontogr_mem.ontology.diff import diff
from ontogr_mem.ontology.models import Ontology
from ontogr_mem.ontology.projection import project


def _ontology() -> Ontology:
    head = Ontology.genesis()
    declaration = OntologyDeclaration(
        entities=[
            EntityDeclaration(label="Person", identity_attributes=["email"]),
            EntityDeclaration(
                label="Employee", parent="Person", identity_attributes=["badge"]
            ),
            EntityDeclaration(label="Former", deprecated=True),
        ]
    )
    return apply_changeset(head, diff(head, build_target(head, declaration)))


class TestProjectionIdentity:
    def test_child_catalog_entry_includes_own_and_inherited_identity_names(self):
        projection = project(_ontology())

        employee = projection.view.catalog.entity_by_label("Employee")

        assert employee is not None
        assert employee.identity_attributes == ("badge", "email")

    def test_deprecated_type_is_not_projected(self):
        projection = project(_ontology())

        assert projection.view.catalog.entity_by_label("Former") is None
