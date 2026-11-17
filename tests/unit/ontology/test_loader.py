"""Tests for ontology declaration loading."""

import json
from pathlib import Path

import pytest
from pydantic import BaseModel, Field

from ontogr_mem.ontology.declaration import (
    EntityDeclaration,
    OntologyDeclaration,
    PatternDeclaration,
    RelationDeclaration,
)
from ontogr_mem.ontology.errors import InvalidChange
from ontogr_mem.ontology.loader import load_ontology, loads_ontology


class TestLoader:
    """Check supported ontology declaration sources."""

    def test_yaml_json_and_declaration_load_to_sorted_shape(
        self, tmp_path: Path
    ) -> None:
        """Load equivalent YAML, JSON, and Python declarations."""
        data = {
            "entities": [{"label": "Person"}, {"label": "Organization"}],
            "relations": [
                {
                    "label": "WORKS_AT",
                    "patterns": [{"source": "Person", "target": "Organization"}],
                }
            ],
        }
        expected = OntologyDeclaration(
            entities=[
                EntityDeclaration(label="Organization"),
                EntityDeclaration(label="Person"),
            ],
            relations=[
                RelationDeclaration(
                    label="WORKS_AT",
                    patterns=[
                        PatternDeclaration(source="Person", target="Organization")
                    ],
                )
            ],
        )
        yaml_path = tmp_path / "ontology.yaml"
        yaml_path.write_text(
            "entities:\n"
            "  - label: Person\n"
            "  - label: Organization\n"
            "relations:\n"
            "  - label: WORKS_AT\n"
            "    patterns:\n"
            "      - source: Person\n"
            "        target: Organization\n",
            encoding="utf-8",
        )
        json_path = tmp_path / "ontology.json"
        json_path.write_text(json.dumps(data), encoding="utf-8")
        assert load_ontology(yaml_path) == expected
        assert load_ontology(json_path) == expected
        assert load_ontology(expected) == expected

    def test_rejects_string_paths_and_bad_suffixes(self, tmp_path: Path) -> None:
        """Reject ambiguous strings and unsupported file formats."""
        with pytest.raises(InvalidChange, match="pathlib.Path"):
            load_ontology("ontology.yaml")
        path = tmp_path / "ontology.txt"
        path.touch()
        with pytest.raises(InvalidChange, match="unsupported ontology file type"):
            load_ontology(path)

    def test_pydantic_classes_produce_entities_and_relations(self) -> None:
        """Read entity and relation declarations from model classes."""

        class Organization(BaseModel):
            """A company."""

        class Person(BaseModel):
            """A human."""

            works_at: Organization = Field(description="Employs a person.")

        declaration = load_ontology([Organization, Person])
        assert [entity.label for entity in declaration.entities] == [
            "Organization",
            "Person",
        ]
        assert declaration.relations[0].label == "WORKS_AT"
        assert declaration.relations[0].patterns == [
            PatternDeclaration(source="Person", target="Organization")
        ]

    def test_malformed_text_is_an_invalid_change(self) -> None:
        """Wrap malformed parser input in the ontology error type."""
        with pytest.raises(InvalidChange):
            loads_ontology("{", file_format="json")
