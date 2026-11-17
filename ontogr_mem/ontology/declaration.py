"""The declared form of an ontology: what a host writes in a file or in code."""

from collections.abc import Sequence

from pydantic import Field

from ontogr_mem.ontology.errors import InvalidChange
from ontogr_mem.ontology.models import Frozen, RelationKind
from ontogr_mem.ontology.names import (
    MAX_NAME_LEN,
    RESERVED_LABELS,
    check_identifier,
    normalize,
)


class EntityDeclaration(Frozen):
    """An entity type as a host declares it."""

    label: str
    description: str = ""
    aliases: list[str] = Field(default_factory=list)
    parent: str | None = None
    identity_attributes: list[str] = Field(default_factory=list)
    renamed_from: list[str] = Field(default_factory=list)
    merged_into: str | None = None
    deprecated: bool = False


class PatternDeclaration(Frozen):
    """The source and target entity labels a relation may connect."""

    source: str
    target: str


class RelationDeclaration(Frozen):
    """A relation type as a host declares it."""

    label: str
    description: str = ""
    aliases: list[str] = Field(default_factory=list)
    kind: RelationKind = RelationKind.MANY
    symmetric: bool = False
    patterns: list[PatternDeclaration] = Field(default_factory=list)
    renamed_from: list[str] = Field(default_factory=list)
    merged_into: str | None = None
    deprecated: bool = False


class OntologyDeclaration(Frozen):
    """An ontology as a host declares it."""

    entities: list[EntityDeclaration] = Field(default_factory=list)
    relations: list[RelationDeclaration] = Field(default_factory=list)


def _fail(message: str, item: str) -> None:
    raise InvalidChange(message, item=item)


def _check_names(
    items: Sequence[EntityDeclaration | RelationDeclaration], kind: str
) -> None:
    names: dict[str, str] = {}
    renamed: set[str] = set()
    for item in items:
        check_identifier(item.label, kind=kind)
        key = normalize(item.label)
        if key in RESERVED_LABELS:
            _fail("reserved label", item.label)
        if key in names:
            _fail(f"name '{item.label}' is declared twice", item.label)
        names[key] = item.label
    for item in items:
        for alias in item.aliases:
            if not alias.strip() or len(alias) > MAX_NAME_LEN:
                _fail(
                    "alias must be non-empty and within the name length limit",
                    item.label,
                )
            key = normalize(alias)
            if key in names:
                _fail(f"name '{alias}' is declared twice", item.label)
            names[key] = item.label
    labels = {normalize(item.label) for item in items}
    for item in items:
        for old in item.renamed_from:
            if not old.strip() or len(old) > MAX_NAME_LEN or normalize(old) in labels:
                _fail(
                    "renamed_from must be unique and differ from declared labels",
                    item.label,
                )
            key = normalize(old)
            if key in renamed:
                _fail("renamed_from name is used more than once", item.label)
            renamed.add(key)


def check_declaration(declaration: OntologyDeclaration) -> None:
    """Check a declaration on its own, before it meets a stored ontology.

    Raises:
        InvalidChange: The first rule that the declaration breaks.
    """
    _check_names(declaration.entities, "entity")
    _check_names(declaration.relations, "relation")
    entities = {e.label: e for e in declaration.entities}
    relations = {r.label: r for r in declaration.relations}
    for entity in declaration.entities:
        _check_entity(entity, entities)
    for relation in declaration.relations:
        _check_relation(relation, relations, entities)


def _check_entity(
    entity: EntityDeclaration, entities: dict[str, EntityDeclaration]
) -> None:
    for field in (entity.parent, entity.merged_into):
        if field is not None and field not in entities:
            _fail("reference must name a declared entity", entity.label)
    if entity.label in (entity.parent, entity.merged_into):
        _fail("type cannot parent or merge into itself", entity.label)
    for ref in (entity.parent, entity.merged_into):
        if ref is not None and (entities[ref].deprecated or entities[ref].merged_into):
            _fail("reference must name an active entity", entity.label)
    seen: set[str] = set()
    for name in entity.identity_attributes:
        check_identifier(name, kind="identity attribute")
        key = normalize(name)
        if key in seen:
            _fail("identity attributes must be unique", entity.label)
        seen.add(key)


def _check_relation(
    relation: RelationDeclaration,
    relations: dict[str, RelationDeclaration],
    entities: dict[str, EntityDeclaration],
) -> None:
    if relation.merged_into is not None and relation.merged_into not in relations:
        _fail("merge target must name a declared relation", relation.label)
    if relation.merged_into == relation.label:
        _fail("type cannot merge into itself", relation.label)
    if relation.merged_into is not None:
        target = relations[relation.merged_into]
        if target.deprecated or target.merged_into:
            _fail("merge target must be active", relation.label)
    seen_patterns: set[tuple[str, str]] = set()
    for pattern in relation.patterns:
        for endpoint in (pattern.source, pattern.target):
            if endpoint not in entities:
                _fail(
                    f"pattern endpoint {endpoint} must name a declared entity",
                    relation.label,
                )
            if entities[endpoint].deprecated or entities[endpoint].merged_into:
                _fail(f"pattern endpoint {endpoint} must be active", relation.label)
        pair = (pattern.source, pattern.target)
        if pair in seen_patterns:
            _fail("pattern is declared twice", relation.label)
        seen_patterns.add(pair)
