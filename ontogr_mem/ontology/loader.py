"""Load an ontology declaration from a file, a class list, or an object."""

import importlib
from collections.abc import Sequence
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from ontogr_mem.ontology.apply import apply_changeset
from ontogr_mem.ontology.declaration import OntologyDeclaration, check_declaration
from ontogr_mem.ontology.declare import build_target
from ontogr_mem.ontology.diff import diff
from ontogr_mem.ontology.errors import InvalidChange
from ontogr_mem.ontology.formats.mapping import declaration_from_mapping, parse_text
from ontogr_mem.ontology.formats.pydantic_models import declaration_from_models
from ontogr_mem.ontology.models import Ontology


OntologySource = OntologyDeclaration | Path | Sequence[type[BaseModel]]
"""What a host can pass as an ontology."""

_SUFFIXES: dict[str, Literal["yaml", "json", "turtle"]] = {
    ".yaml": "yaml",
    ".yml": "yaml",
    ".json": "json",
    ".ttl": "turtle",
}


def _canonical(declaration: OntologyDeclaration) -> OntologyDeclaration:
    return declaration.model_copy(
        update={
            "entities": sorted(declaration.entities, key=lambda item: item.label),
            "relations": [
                relation.model_copy(
                    update={
                        "patterns": sorted(
                            relation.patterns,
                            key=lambda pattern: (pattern.source, pattern.target),
                        )
                    }
                )
                for relation in sorted(
                    declaration.relations, key=lambda item: item.label
                )
            ],
        }
    )


def loads_ontology(
    text: str, *, file_format: Literal["yaml", "json", "turtle"]
) -> OntologyDeclaration:
    """Parse declaration text and return a checked declaration in label order."""
    try:
        if file_format == "turtle":
            module = importlib.import_module("ontogr_mem.ontology.formats.turtle")
            declaration = module.declaration_from_turtle(text)
        else:
            declaration = declaration_from_mapping(parse_text(text, file_format))
    except InvalidChange:
        raise
    except Exception as exc:
        raise InvalidChange(
            f"cannot read the {file_format} ontology", hint=str(exc)
        ) from exc
    return _validate(declaration)


def _validate(declaration: OntologyDeclaration) -> OntologyDeclaration:
    declaration = _canonical(declaration)
    check_declaration(declaration)
    genesis = Ontology.genesis()
    target = build_target(genesis, declaration)
    apply_changeset(genesis, diff(genesis, target))
    return declaration


def _read_file(path: Path) -> OntologyDeclaration:
    file_format = _SUFFIXES.get(path.suffix.lower())
    if file_format is None:
        raise InvalidChange(
            "unsupported ontology file type",
            item=path.name,
            hint=(
                "use .yaml, .yml, .json or .ttl; convert OWL or RDF/XML files to Turtle"
            ),
        )
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise InvalidChange("cannot read the ontology file", item=str(path)) from exc
    return loads_ontology(text, file_format=file_format)


def load_ontology(source: OntologySource) -> OntologyDeclaration:
    """Return a checked declaration from a declaration, path, or model classes."""
    if isinstance(source, str):
        raise InvalidChange(
            "pass a pathlib.Path for a file",
            item=source[:40],
            hint="wrap the path in Path(...)",
        )
    if isinstance(source, OntologyDeclaration):
        declaration = source
    elif isinstance(source, Path):
        return _read_file(source)
    else:
        declaration = declaration_from_models(source)
    return _validate(declaration)
