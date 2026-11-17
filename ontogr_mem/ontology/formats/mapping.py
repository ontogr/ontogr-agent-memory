"""Read a declaration from YAML or JSON text."""

import json
from typing import Any, Literal

import yaml
from pydantic import ValidationError

from ontogr_mem.ontology.declaration import OntologyDeclaration
from ontogr_mem.ontology.errors import InvalidChange


def parse_text(text: str, file_format: Literal["yaml", "json"]) -> Any:
    """Parse YAML or JSON text into plain data."""
    try:
        return yaml.safe_load(text) if file_format == "yaml" else json.loads(text)
    except (yaml.YAMLError, json.JSONDecodeError) as exc:
        raise InvalidChange(
            f"cannot read the {file_format} ontology", hint=str(exc)
        ) from exc


def declaration_from_mapping(data: Any) -> OntologyDeclaration:
    """Build a declaration from parsed data."""
    if data is None:
        data = {}
    if not isinstance(data, dict):
        raise InvalidChange("an ontology file holds a mapping at the top level")
    try:
        return OntologyDeclaration.model_validate(data)
    except ValidationError as exc:
        raise InvalidChange(
            "the ontology file does not match the format", hint=str(exc)
        ) from exc
