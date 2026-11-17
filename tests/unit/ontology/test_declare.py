"""Tests for reconciling an ontology declaration with a stored ontology."""

import pytest

from ontogr_mem.ontology.apply import apply_changeset
from ontogr_mem.ontology.declaration import (
    EntityDeclaration,
    OntologyDeclaration,
    PatternDeclaration,
    RelationDeclaration,
)
from ontogr_mem.ontology.declare import build_target
from ontogr_mem.ontology.diff import diff
from ontogr_mem.ontology.errors import InvalidChange
from ontogr_mem.ontology.models import Ontology, RelationKind, Status


class TestDeclare:
    """Reconcile declared ontology state against the stored catalog."""

    @staticmethod
    def _build(head: Ontology, declaration: OntologyDeclaration) -> Ontology:
        target = build_target(head, declaration)
        return apply_changeset(head, diff(head, target))

    def test_a_new_declaration_builds_the_ontology_with_parents_first(self):
        """Parents are assigned before children even when listed later."""
        head = Ontology.genesis()
        declaration = OntologyDeclaration(
            entities=[
                EntityDeclaration(label="Employee", parent="Person"),
                EntityDeclaration(label="Person"),
            ]
        )

        result = self._build(head, declaration)

        person = next(entity for entity in result.entities if entity.label == "Person")
        employee = next(
            entity for entity in result.entities if entity.label == "Employee"
        )
        assert employee.parent_id == person.type_id

    def test_declaring_the_same_declaration_again_commits_nothing(self):
        """A fully described ontology is stable after its first application."""
        declaration = OntologyDeclaration(
            entities=[
                EntityDeclaration(
                    label="Person", aliases=["Human"], identity_attributes=["email"]
                ),
                EntityDeclaration(label="Employee", parent="Person"),
                EntityDeclaration(label="Old", deprecated=True),
                EntityDeclaration(label="Former", merged_into="Person"),
            ],
            relations=[
                RelationDeclaration(
                    label="OWNS",
                    kind=RelationKind.SINGLE,
                    patterns=[PatternDeclaration(source="Person", target="Employee")],
                ),
                RelationDeclaration(label="LIKES", kind=RelationKind.MANY),
                RelationDeclaration(label="JUDGES", kind=RelationKind.JUDGED),
            ],
        )
        head = self._build(Ontology.genesis(), declaration)

        assert diff(head, build_target(head, declaration)).ops == []

    def test_renaming_keeps_the_type_id_and_old_label(self):
        """A declared rename updates the label without replacing the type."""
        first = self._build(
            Ontology.genesis(),
            OntologyDeclaration(entities=[EntityDeclaration(label="Old")]),
        )
        old_id = next(
            entity.type_id for entity in first.entities if entity.label == "Old"
        )

        result = self._build(
            first,
            OntologyDeclaration(
                entities=[EntityDeclaration(label="New", renamed_from=["Old"])]
            ),
        )

        renamed = next(entity for entity in result.entities if entity.type_id == old_id)
        assert renamed.label == "New"
        assert "old" in result.labels_of(renamed)

    def test_removing_a_type_retires_its_relation_pattern_before_the_type(self):
        """Dropping an endpoint and relation retires their active pattern cleanly."""
        head = self._build(
            Ontology.genesis(),
            OntologyDeclaration(
                entities=[
                    EntityDeclaration(label="Person"),
                    EntityDeclaration(label="Organization"),
                ],
                relations=[
                    RelationDeclaration(
                        label="WORKS_AT",
                        patterns=[
                            PatternDeclaration(source="Person", target="Organization")
                        ],
                    )
                ],
            ),
        )

        result = self._build(head, OntologyDeclaration())

        relation = next(item for item in result.relations if item.label == "WORKS_AT")
        organization = next(
            item for item in result.entities if item.label == "Organization"
        )
        assert relation.status is Status.DEPRECATED
        assert all(pattern.status is Status.DEPRECATED for pattern in relation.patterns)
        assert organization.status is Status.DEPRECATED
        assert diff(result, build_target(result, OntologyDeclaration())).ops == []

    def test_reactivating_a_merged_type_is_rejected(self):
        """A merged type cannot become active while retaining its merge history."""
        declaration = OntologyDeclaration(
            entities=[
                EntityDeclaration(label="Target"),
                EntityDeclaration(label="Source", merged_into="Target"),
            ]
        )
        merged = self._build(Ontology.genesis(), declaration)

        with pytest.raises(InvalidChange, match="merged type cannot be reactivated"):
            self._build(
                merged,
                OntologyDeclaration(
                    entities=[
                        EntityDeclaration(label="Target"),
                        EntityDeclaration(label="Source"),
                    ]
                ),
            )

    def test_a_pattern_with_an_absent_endpoint_is_refused(self):
        """Relations cannot refer to entity types missing from the declaration."""
        declaration = OntologyDeclaration(
            relations=[
                RelationDeclaration(
                    label="WORKS_AT",
                    patterns=[PatternDeclaration(source="Person", target="Org")],
                )
            ]
        )

        with pytest.raises(InvalidChange, match="pattern endpoint"):
            build_target(Ontology.genesis(), declaration)

    def test_document_is_a_legal_entity_label(self):
        """Document is available as an ordinary declared entity type."""
        result = self._build(
            Ontology.genesis(),
            OntologyDeclaration(entities=[EntityDeclaration(label="Document")]),
        )

        assert any(entity.label == "Document" for entity in result.entities)

    @pytest.mark.parametrize("type_kind", ["entity", "relation"])
    @pytest.mark.parametrize("stored", ["absent", "active", "deprecated", "merged"])
    @pytest.mark.parametrize(
        "desired", ["absent", "active", "deprecated", "merged", "redirected"]
    )
    def test_stored_and_declared_states_converge(self, type_kind, stored, desired):
        """Each valid transition settles after one application."""

        def declaration(state: str) -> OntologyDeclaration:
            redirect = {"merged": "Target", "redirected": "Other"}.get(state)
            if type_kind == "entity":
                entities = [
                    EntityDeclaration(label="Target"),
                    EntityDeclaration(label="Other"),
                ]
                if state != "absent":
                    entities.append(
                        EntityDeclaration(
                            label="Source",
                            deprecated=state == "deprecated",
                            merged_into=redirect,
                        )
                    )
                return OntologyDeclaration(entities=entities)
            relations = [
                RelationDeclaration(label="Target"),
                RelationDeclaration(label="Other"),
            ]
            if state != "absent":
                relations.append(
                    RelationDeclaration(
                        label="Source",
                        deprecated=state == "deprecated",
                        merged_into=redirect,
                    )
                )
            return OntologyDeclaration(relations=relations)

        head = self._build(Ontology.genesis(), declaration(stored))
        requested = declaration(desired)
        if stored == "merged" and desired == "active":
            with pytest.raises(
                InvalidChange, match="a merged type cannot be reactivated"
            ):
                self._build(head, requested)
            return

        result = self._build(head, requested)
        assert diff(result, build_target(result, requested)).ops == []

    def test_aliases_grow_and_descriptions_can_change(self):
        """A declaration adds aliases and retains the names it omits."""
        initial = OntologyDeclaration(
            entities=[EntityDeclaration(label="Company", aliases=["Firm"])]
        )
        updated = OntologyDeclaration(
            entities=[
                EntityDeclaration(
                    label="Company", description="An employer.", aliases=["Business"]
                )
            ]
        )
        head = self._build(Ontology.genesis(), initial)
        result = self._build(head, updated)
        company = next(item for item in result.entities if item.label == "Company")
        assert company.aliases == ["Firm", "Business"]
        assert company.description == "An employer."
        assert diff(result, build_target(result, updated)).ops == []

    def test_dropping_a_merge_target_is_refused(self):
        """A declaration keeps an active destination for each retired type."""
        head = self._build(
            Ontology.genesis(),
            OntologyDeclaration(
                entities=[
                    EntityDeclaration(label="Source", merged_into="Target"),
                    EntityDeclaration(label="Target"),
                ]
            ),
        )
        with pytest.raises(InvalidChange, match="Source is merged into Target"):
            self._build(head, OntologyDeclaration())
