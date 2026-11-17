"""Turn a declaration into the ontology it describes, relative to a stored one."""

from collections.abc import Sequence
from typing import TypeVar

from ontogr_mem.ontology.declaration import (
    EntityDeclaration,
    OntologyDeclaration,
    RelationDeclaration,
    check_declaration,
)
from ontogr_mem.ontology.errors import InvalidChange
from ontogr_mem.ontology.models import (
    UNKNOWN_ID,
    Entity,
    Ontology,
    Origin,
    Pattern,
    Relation,
    Status,
)
from ontogr_mem.ontology.names import normalize


Declared = EntityDeclaration | RelationDeclaration
Stored = Entity | Relation
_StoredType = TypeVar("_StoredType", Entity, Relation)


def _depth(item: EntityDeclaration, by_label: dict[str, EntityDeclaration]) -> int:
    depth = 0
    seen = {item.label}
    while item.parent is not None:
        if item.parent in seen:
            raise InvalidChange("hierarchy cycle", item=item.label)
        seen.add(item.parent)
        item = by_label[item.parent]
        depth += 1
    return depth


def _match(
    stored: Sequence[Stored], declared: Sequence[Declared], next_id: int
) -> tuple[dict[str, int], set[int], int]:
    """Match declared names to stored ids and allocate new ids in order."""
    candidates = [
        t for t in stored if not (isinstance(t, Entity) and t.type_id == UNKNOWN_ID)
    ]
    by_label = {normalize(t.label): t for t in candidates}
    by_name: dict[str, Stored] = {}
    for type_ in candidates:
        for name in (type_.label, *type_.aliases):
            by_name.setdefault(normalize(name), type_)
    ids: dict[str, int] = {}
    used: set[int] = set()
    for item in declared:
        found = by_label.get(normalize(item.label))
        if found is None:
            found = next(
                (
                    by_name[key]
                    for key in map(normalize, item.renamed_from)
                    if key in by_name
                ),
                None,
            )
        if found is None:
            ids[item.label] = next_id
            next_id += 1
        elif found.type_id in used:
            raise InvalidChange(
                "two declared types match one stored type", item=item.label
            )
        else:
            used.add(found.type_id)
            ids[item.label] = found.type_id
    return ids, used, next_id


def _names(existing: Sequence[str], label: str, declared: Sequence[str]) -> list[str]:
    result: list[str] = []
    seen = {normalize(label)}
    for name in (*existing, *declared):
        key = normalize(name)
        if key not in seen:
            seen.add(key)
            result.append(name)
    return result


def _status(item: Declared) -> Status:
    return Status.DEPRECATED if item.deprecated or item.merged_into else Status.ACTIVE


def _merged_into(
    stored: Stored | None, item: Declared, ids: dict[str, int]
) -> int | None:
    if item.merged_into is not None:
        return ids[item.merged_into]
    if stored is not None and item.deprecated:
        return stored.merged_into
    return None


def _retired(type_: _StoredType) -> _StoredType:
    changes: dict[str, object] = {"status": Status.DEPRECATED}
    if isinstance(type_, Relation):
        changes["patterns"] = [
            p.model_copy(update={"status": Status.DEPRECATED}) for p in type_.patterns
        ]
    return type_.model_copy(update=changes)


def build_target(head: Ontology, declaration: OntologyDeclaration) -> Ontology:
    """Build the target ontology described by a declaration, relative to ``head``."""
    check_declaration(declaration)
    entity_decls = {d.label: d for d in declaration.entities}
    ordered_entities = sorted(
        declaration.entities, key=lambda item: (_depth(item, entity_decls), item.label)
    )
    entity_ids, entity_used, next_entity = _match(
        head.entities,
        ordered_entities,
        head.next_entity_id,
    )
    relation_ids, relation_used, next_relation = _match(
        head.relations,
        sorted(declaration.relations, key=lambda item: item.label),
        head.next_relation_id,
    )
    old_entities = {e.type_id: e for e in head.entities}
    old_relations = {r.type_id: r for r in head.relations}
    entities: list[Entity] = []
    for item in sorted(declaration.entities, key=lambda d: entity_ids[d.label]):
        old = old_entities.get(entity_ids[item.label])
        entities.append(
            Entity(
                type_id=entity_ids[item.label],
                label=item.label,
                description=item.description,
                aliases=_names(
                    [*old.aliases, old.label] if old else (), item.label, item.aliases
                ),
                parent_id=entity_ids[item.parent] if item.parent else None,
                identity_attributes=list(item.identity_attributes),
                status=_status(item),
                merged_into=_merged_into(old, item, entity_ids),
                origin=old.origin if old else Origin.DECLARED,
            )
        )
    entities += [e for e in head.entities if e.type_id == UNKNOWN_ID]
    entities += [
        _retired(e)
        for e in head.entities
        if e.type_id != UNKNOWN_ID and e.type_id not in entity_used
    ]
    relations: list[Relation] = []
    for item in sorted(declaration.relations, key=lambda d: relation_ids[d.label]):
        old = old_relations.get(relation_ids[item.label])
        status = _status(item)
        patterns = [
            Pattern(src=entity_ids[p.source], tgt=entity_ids[p.target], status=status)
            for p in item.patterns
        ]
        keep = {(p.src, p.tgt) for p in patterns}
        patterns += [
            Pattern(src=p.src, tgt=p.tgt, status=Status.DEPRECATED)
            for p in (old.patterns if old else [])
            if (p.src, p.tgt) not in keep
        ]
        relations.append(
            Relation(
                type_id=relation_ids[item.label],
                label=item.label,
                description=item.description,
                aliases=_names(
                    [*old.aliases, old.label] if old else (), item.label, item.aliases
                ),
                symmetric=item.symmetric,
                kind=item.kind,
                patterns=patterns,
                status=status,
                merged_into=_merged_into(old, item, relation_ids),
                origin=old.origin if old else Origin.DECLARED,
            )
        )
    relations += [_retired(r) for r in head.relations if r.type_id not in relation_used]
    _check_merge_targets(entities)
    _check_merge_targets(relations)
    return head.model_copy(
        update={
            "entities": sorted(entities, key=lambda item: item.type_id),
            "relations": sorted(relations, key=lambda item: item.type_id),
            "next_entity_id": next_entity,
            "next_relation_id": next_relation,
        }
    )


def _check_merge_targets(types: Sequence[Stored]) -> None:
    """Require every merge chain to end at an active type."""
    by_id = {item.type_id: item for item in types}
    for item in types:
        if item.merged_into is None:
            continue
        target = item
        seen = {item.type_id}
        while target.merged_into is not None and target.merged_into not in seen:
            seen.add(target.merged_into)
            target = by_id[target.merged_into]
        if target.merged_into is not None or target.status is not Status.ACTIVE:
            raise InvalidChange(
                f"{item.label} is merged into {target.label}, which is not active",
                item=item.label,
                hint="keep the target active or merge the type elsewhere",
            )
