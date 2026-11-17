"""Tests for ontology declaration loading."""

import json
from pathlib import Path
from typing import ForwardRef

import pytest
from pydantic import BaseModel, Field, create_model

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
        turtle_path = tmp_path / "ontology.ttl"
        turtle_path.write_text(
            "@prefix : <https://example.org/ontology#> .\n"
            "@prefix owl: <http://www.w3.org/2002/07/owl#> .\n"
            "@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .\n"
            ":Person a owl:Class .\n"
            ":Organization a owl:Class .\n"
            ":worksAt a owl:ObjectProperty ; rdfs:domain :Person ; "
            "rdfs:range :Organization .\n",
            encoding="utf-8",
        )
        assert load_ontology(turtle_path) == expected
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

    @pytest.mark.parametrize(
        ("text", "file_format"), [("", "yaml"), ("null", "yaml"), ("null", "json")]
    )
    def test_empty_documents_are_refused(self, text, file_format):
        """A blank or null document cannot replace a tenant's ontology."""
        with pytest.raises(InvalidChange, match="mapping"):
            loads_ontology(text, file_format=file_format)

    def test_subclass_override_adds_its_own_relation_pattern(self):
        """Keep relationship targets explicitly overridden by a subclass."""

        class Organization(BaseModel):
            """An organization."""

        class Team(BaseModel):
            """A team."""

        class Person(BaseModel):
            """A person."""

            belongs_to: Organization | Team

        employee = create_model("Employee", __base__=Person, belongs_to=(Team, ...))

        declaration = load_ontology([Organization, Team, Person, employee])
        assert declaration.relations[0].patterns == [
            PatternDeclaration(source="Employee", target="Team"),
            PatternDeclaration(source="Person", target="Organization"),
            PatternDeclaration(source="Person", target="Team"),
        ]

    def test_unresolved_pydantic_field_is_refused(self):
        """An unresolved field cannot silently disappear from the ontology."""

        class Person(BaseModel):
            """A person."""

        Person.model_fields["friend"] = Field()
        Person.model_fields["friend"].annotation = ForwardRef("MissingPerson")
        with pytest.raises(InvalidChange, match="Person.friend"):
            load_ontology([Person])
