"""Read ontology declarations from Pydantic model classes."""

import inspect
import types
from collections.abc import Sequence
from typing import Annotated, Union, get_args, get_origin

from pydantic import BaseModel

from ontogr_mem.ontology.declaration import (
    EntityDeclaration,
    OntologyDeclaration,
    PatternDeclaration,
    RelationDeclaration,
)
from ontogr_mem.ontology.errors import InvalidChange
from ontogr_mem.ontology.models import RelationKind


_CONTAINERS = {list, set, frozenset, tuple, Sequence}


def _targets(annotation: object, classes: set[type[BaseModel]]) -> set[str]:
    origin = get_origin(annotation)
    arguments = get_args(annotation)
    if origin is Annotated:
        return _targets(arguments[0], classes)
    if origin in _CONTAINERS or origin in (Union, types.UnionType):
        return set().union(*(_targets(argument, classes) for argument in arguments))
    if isinstance(annotation, type) and annotation in classes:
        return {annotation.__name__}
    return set()


def declaration_from_models(models: Sequence[type[BaseModel]]) -> OntologyDeclaration:
    """Build a declaration from Pydantic classes and their typed fields."""
    if not models:
        raise InvalidChange("no model classes given")
    for model in models:
        if not isinstance(model, type) or not issubclass(model, BaseModel):
            raise InvalidChange("not a Pydantic model class", item=repr(model))
    listed = set(models)
    entities = [
        EntityDeclaration(
            label=model.__name__,
            description=inspect.cleandoc(model.__dict__.get("__doc__") or ""),
            parent=next(
                (parent.__name__ for parent in model.__mro__[1:] if parent in listed),
                None,
            ),
        )
        for model in models
    ]
    relations: dict[str, RelationDeclaration] = {}
    for model in models:
        parent = next((base for base in model.__mro__[1:] if base in listed), None)
        inherited = (
            set(parent.model_fields)
            if parent is not None and issubclass(parent, BaseModel)
            else set()
        )
        for name, field in model.model_fields.items():
            if name in inherited:
                continue
            targets = _targets(field.annotation, listed)
            if not targets:
                continue
            label = name.upper()
            relation = relations.get(label)
            patterns = [
                PatternDeclaration(source=model.__name__, target=target)
                for target in targets
            ]
            if relation is None:
                relations[label] = RelationDeclaration(
                    label=label,
                    description=field.description or "",
                    kind=RelationKind.MANY,
                    patterns=patterns,
                )
            else:
                existing = {(item.source, item.target) for item in relation.patterns}
                relation.patterns.extend(
                    pattern
                    for pattern in patterns
                    if (pattern.source, pattern.target) not in existing
                )
                if not relation.description and field.description:
                    relations[label] = relation.model_copy(
                        update={"description": field.description}
                    )
    return OntologyDeclaration(entities=entities, relations=list(relations.values()))
