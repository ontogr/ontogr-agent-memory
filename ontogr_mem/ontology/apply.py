"""The one commit path: apply a change set to a catalog and check the result.

Every way in (user API, import) builds a
``ChangeSet`` and ends here. The function is pure; storing the result is the
store's job.
"""

from collections.abc import Callable, Iterable
from typing import Any, TypeVar

from ontogr_mem.ontology.errors import InvalidChange, VersionConflict
from ontogr_mem.ontology.models import (
    UNKNOWN_ID,
    AddAlias,
    AddEntity,
    AddPattern,
    AddRelation,
    ChangeSet,
    DeprecatePattern,
    DeprecateType,
    Entity,
    Kind,
    MergeInto,
    Ontology,
    Op,
    Pattern,
    PatternSpec,
    Reactivate,
    Ref,
    Relation,
    Rename,
    SetDescription,
    SetExclusive,
    Status,
)
from ontogr_mem.ontology.names import (
    MAX_NAME_LEN,
    RESERVED_LABELS,
    check_identifier,
    normalize,
)


MAX_DEPTH = 8


def apply_changeset(cat: Ontology, cs: ChangeSet) -> Ontology:
    """Return the catalog that results from applying ``cs`` to ``cat``.

    The change set is applied to a deep copy, one op at a time. Then the whole result
    is checked. The input catalog is never modified, so a failed change leaves nothing
    behind.

    Args:
        cat: The current catalog, the head.
        cs: The change set to apply. Its ``base_version`` must equal ``cat.version``.

    Returns:
        The new catalog: one version higher, with ``parent_version`` and
        ``content_hash`` set.

    Raises:
        VersionConflict: ``cs.base_version`` is not ``cat.version``.
        InvalidChange: An op or the resulting catalog breaks a rule.
    """
    if cs.base_version != cat.version:
        raise VersionConflict(cs.base_version)
    work = cat.model_copy(deep=True)
    index = _CatalogIndex(work)
    for op in cs.ops:
        _apply_op(work, op, index)
    _check_indexed(work, index)
    out = work.model_copy(
        update={"version": cat.version + 1, "parent_version": cat.version}
    )
    return out.sealed()


_TypeT = TypeVar("_TypeT", Entity, Relation)


def _find_by_label(
    cat: Ontology, kind: Kind, items: Iterable[_TypeT], ref: str
) -> _TypeT:
    """Return the type of ``kind`` whose label or alias is ``ref``.

    A label shared by several types prefers an active one.

    Raises:
        InvalidChange: No type of ``kind`` carries that name.
    """
    key = normalize(ref)
    found = [t for t in items if key in cat.labels_of(t)]
    if not found:
        raise InvalidChange(f"no {kind.value} type named {ref!r}", item=ref)
    active = [t for t in found if t.status is Status.ACTIVE]
    return (active or found)[0]


class _CatalogIndex:
    """Types in one working catalog, by stable id."""

    def __init__(self, cat: Ontology) -> None:
        """Index the types of ``cat``."""
        self.entities: dict[int, Entity] = {e.type_id: e for e in cat.entities}
        self.relations: dict[int, Relation] = {r.type_id: r for r in cat.relations}

    def entity(self, type_id: int) -> Entity | None:
        """Return the entity type with this id, or None."""
        return self.entities.get(type_id)

    def relation(self, type_id: int) -> Relation | None:
        """Return the relation type with this id, or None."""
        return self.relations.get(type_id)

    def get(self, kind: Kind, type_id: int) -> Entity | Relation | None:
        """Return the type of ``kind`` with this id, or None."""
        if kind is Kind.ENTITY:
            return self.entities.get(type_id)
        return self.relations.get(type_id)

    def add_entity(self, entity: Entity) -> None:
        """Add a newly declared entity type to the index."""
        self.entities[entity.type_id] = entity

    def add_relation(self, relation: Relation) -> None:
        """Add a newly declared relation type to the index."""
        self.relations[relation.type_id] = relation

    def follow(self, kind: Kind, type_id: int) -> int:
        """Follow merge pointers using indexed type lookups."""
        seen = {type_id}
        current_id = type_id
        current = self.get(kind, current_id)
        if current is None:
            return current_id
        label = current.label
        while current.merged_into is not None and current.merged_into not in seen:
            current_id = current.merged_into
            seen.add(current_id)
            current = self.get(kind, current_id)
            if current is None:
                raise InvalidChange("bad merge pointer", item=label)
        return current_id


def _entity(cat: Ontology, ref: Ref, index: _CatalogIndex) -> Entity:
    """Find the entity type with this id, label or alias.

    Raises:
        InvalidChange: No entity type matches.
    """
    if isinstance(ref, int):
        found = index.entity(ref)
        if found is None:
            raise InvalidChange(f"no entity type with id {ref}", item=str(ref))
        return found
    return _find_by_label(cat, Kind.ENTITY, cat.entity_types(), ref)


def _relation(cat: Ontology, ref: Ref, index: _CatalogIndex) -> Relation:
    """Find the relation type with this id, label or alias.

    Raises:
        InvalidChange: No relation type matches.
    """
    if isinstance(ref, int):
        found = index.relation(ref)
        if found is None:
            raise InvalidChange(f"no relation type with id {ref}", item=str(ref))
        return found
    return _find_by_label(cat, Kind.RELATION, cat.relation_types(), ref)


def _find(
    cat: Ontology, kind: Kind, ref: Ref, index: _CatalogIndex
) -> Entity | Relation:
    """Find a type of ``kind`` by id, or by label or alias.

    Raises:
        InvalidChange: No type of ``kind`` matches.
    """
    if kind is Kind.ENTITY:
        return _entity(cat, ref, index)
    return _relation(cat, ref, index)


def _no_unknown(cat: Ontology, kind: Kind, t: Entity | Relation) -> None:
    """Refuse any edit of the reserved ``Unknown`` entity type."""
    if kind is Kind.ENTITY and t.type_id == UNKNOWN_ID:
        raise InvalidChange(
            "the Unknown type is reserved and cannot be changed", item=t.label
        )


def _pattern(cat: Ontology, spec: PatternSpec, index: _CatalogIndex) -> tuple[int, int]:
    """Resolve both endpoints of a pattern spec to entity type ids."""
    return (
        _entity(cat, spec.src, index).type_id,
        _entity(cat, spec.tgt, index).type_id,
    )


OpHandler = Callable[[Ontology, Any, _CatalogIndex], None]
"""What one op does to the working catalog.

Every handler takes the op it handles, so the op classes stay typed where the
handler is written.
"""


def _apply_op(work: Ontology, op: Op, index: _CatalogIndex) -> None:
    """Apply one op to the working catalog ``work``, in place.

    The op vocabulary is a closed union, so the handler is looked up by the op's
    class. An op that repeats existing state is a no-op. Rules that need the whole
    catalog are checked afterwards by ``_check``.
    """
    _HANDLERS[type(op)](work, op, index)


def _apply_add_entity(work: Ontology, op: AddEntity, index: _CatalogIndex) -> None:
    """Add an entity type."""
    check_identifier(op.label, kind="entity")
    parent = _entity(work, op.parent, index).type_id if op.parent is not None else None
    ent = Entity(
        type_id=work.next_entity_id,
        label=op.label,
        description=op.description,
        aliases=list(op.aliases),
        parent_id=parent,
        origin=op.origin,
    )
    work.next_entity_id += 1
    work.entities.append(ent)
    index.add_entity(ent)


def _apply_add_relation(work: Ontology, op: AddRelation, index: _CatalogIndex) -> None:
    """Add a relation type."""
    check_identifier(op.label, kind="relation")
    pats = [
        Pattern(src=src, tgt=tgt)
        for src, tgt in (_pattern(work, pattern, index) for pattern in op.patterns)
    ]
    rel = Relation(
        type_id=work.next_relation_id,
        label=op.label,
        description=op.description,
        aliases=list(op.aliases),
        symmetric=op.symmetric,
        exclusive=op.exclusive,
        patterns=pats,
        origin=op.origin,
    )
    work.next_relation_id += 1
    work.relations.append(rel)
    index.add_relation(rel)


def _apply_add_pattern(work: Ontology, op: AddPattern, index: _CatalogIndex) -> None:
    """Add an ordered pattern to a relation type."""
    rel = _relation(work, op.relation, index)
    src, tgt = _pattern(work, op.pattern, index)
    for p in rel.patterns:
        if (p.src, p.tgt) == (src, tgt):
            p.status = Status.ACTIVE
            return
    rel.patterns.append(Pattern(src=src, tgt=tgt))


def _apply_add_alias(work: Ontology, op: AddAlias, index: _CatalogIndex) -> None:
    """Add an alias to a type."""
    t = _find(work, op.kind, op.target, index)
    _no_unknown(work, op.kind, t)
    alias = op.alias.strip()
    if not alias or len(alias) > MAX_NAME_LEN:
        raise InvalidChange("illegal alias", item=op.alias[:20])
    if normalize(alias) not in work.labels_of(t):
        t.aliases.append(alias)


def _apply_rename(work: Ontology, op: Rename, index: _CatalogIndex) -> None:
    """Change a type's label; the old label stays as an alias."""
    t = _find(work, op.kind, op.target, index)
    _no_unknown(work, op.kind, t)
    check_identifier(op.new_label, kind=op.kind.value)
    t.aliases = [a for a in t.aliases if normalize(a) != normalize(op.new_label)]
    if normalize(t.label) != normalize(op.new_label):
        t.aliases.append(t.label)
    t.label = op.new_label


def _apply_set_description(
    work: Ontology, op: SetDescription, index: _CatalogIndex
) -> None:
    """Replace a type's description."""
    t = _find(work, op.kind, op.target, index)
    _no_unknown(work, op.kind, t)
    t.description = op.description


def _apply_set_exclusive(
    work: Ontology, op: SetExclusive, index: _CatalogIndex
) -> None:
    """Set whether a subject holds at most one current value of a relation."""
    t = _relation(work, op.target, index)
    _no_unknown(work, Kind.RELATION, t)
    t.exclusive = op.exclusive


def _apply_deprecate_type(
    work: Ontology, op: DeprecateType, index: _CatalogIndex
) -> None:
    """Retire a type; it stays resolvable but is no longer offered."""
    t = _find(work, op.kind, op.target, index)
    _no_unknown(work, op.kind, t)
    t.status = Status.DEPRECATED


def _apply_deprecate_pattern(
    work: Ontology, op: DeprecatePattern, index: _CatalogIndex
) -> None:
    """Retire a pattern."""
    rel = _relation(work, op.relation, index)
    src, tgt = _pattern(work, op.pattern, index)
    hit = [p for p in rel.patterns if (p.src, p.tgt) == (src, tgt)]
    if not hit:
        raise InvalidChange("no such pattern", item=f"{rel.label}({src},{tgt})")
    hit[0].status = Status.DEPRECATED


def _apply_reactivate(work: Ontology, op: Reactivate, index: _CatalogIndex) -> None:
    """Make a deprecated type active again."""
    t = _find(work, op.kind, op.target, index)
    _no_unknown(work, op.kind, t)
    if t.merged_into is not None:
        raise InvalidChange(
            "a merged type cannot be reactivated",
            item=t.label,
            hint="declare a new type instead",
        )
    t.status = Status.ACTIVE


def _apply_merge_into(work: Ontology, op: MergeInto, index: _CatalogIndex) -> None:
    """Deprecate the source type and point it, and its absorbed types, at the target."""
    src_t = _find(work, op.kind, op.source, index)
    dst_t = _find(work, op.kind, op.into, index)
    _no_unknown(work, op.kind, src_t)
    if dst_t.type_id == UNKNOWN_ID and op.kind is Kind.ENTITY:
        raise InvalidChange("cannot merge into Unknown", item=src_t.label)
    if src_t.type_id == dst_t.type_id:
        raise InvalidChange("cannot merge a type into itself", item=src_t.label)
    final = index.follow(op.kind, dst_t.type_id)
    if final == src_t.type_id:
        raise InvalidChange("merge would form a cycle", item=src_t.label)
    src_t.status = Status.DEPRECATED
    src_t.merged_into = final
    for t in work.types(op.kind):
        if t.merged_into == src_t.type_id:
            t.merged_into = final


_HANDLERS: dict[type[Op], OpHandler] = {
    AddEntity: _apply_add_entity,
    AddRelation: _apply_add_relation,
    AddPattern: _apply_add_pattern,
    AddAlias: _apply_add_alias,
    Rename: _apply_rename,
    SetDescription: _apply_set_description,
    SetExclusive: _apply_set_exclusive,
    DeprecateType: _apply_deprecate_type,
    DeprecatePattern: _apply_deprecate_pattern,
    Reactivate: _apply_reactivate,
    MergeInto: _apply_merge_into,
}


def _check(cat: Ontology) -> None:
    """Check every catalog-wide rule on the result of a change set.

    The checks are the ``_check_*`` helpers below. The first violation raises
    ``InvalidChange`` naming the item and, where one exists, the way out.
    """
    _check_indexed(cat, _CatalogIndex(cat))


def _check_indexed(cat: Ontology, index: _CatalogIndex) -> None:
    """Check ``cat`` with an index that already reflects its current types."""
    _check_unknown(cat, index)
    for kind in (Kind.ENTITY, Kind.RELATION):
        _check_names(cat, kind)
    _check_hierarchy(cat, index)
    _check_patterns(cat, index)
    _check_merges(cat, index)


def _check_unknown(cat: Ontology, index: _CatalogIndex) -> None:
    """Type ids are unique, and the Unknown entity is still active and unmerged."""
    ids = [e.type_id for e in cat.entities]
    if len(set(ids)) != len(ids) or len({r.type_id for r in cat.relations}) != len(
        cat.relations
    ):
        raise InvalidChange("duplicate type id")
    unknown = index.entity(UNKNOWN_ID)
    if unknown is None or unknown.status is not Status.ACTIVE or unknown.merged_into:
        raise InvalidChange("the Unknown type must stay active", item="Unknown")


def _check_names(cat: Ontology, kind: Kind) -> None:
    """Labels are legal and not reserved, and no name collides.

    Collisions are tested across labels and aliases of one kind after normalization, so
    names that differ only by case, spaces or underscores count as the same.
    """
    seen: dict[str, int] = {}
    for t in cat.types(kind):
        is_unknown = kind is Kind.ENTITY and t.type_id == UNKNOWN_ID
        if not is_unknown:
            check_identifier(t.label, kind=kind.value)
            if t.label.casefold() in RESERVED_LABELS:
                raise InvalidChange("reserved label", item=t.label)
        for name in [t.label, *t.aliases]:
            key = normalize(name)
            other = seen.get(key)
            if other is not None and other != t.type_id:
                raise InvalidChange(
                    f"name {name!r} collides with type {other}",
                    item=t.label,
                    hint="names differ only by case, spaces or underscores",
                )
            seen[key] = t.type_id


def _check_hierarchy(cat: Ontology, index: _CatalogIndex) -> None:
    """Every parent exists and is not Unknown, with no cycle and depth at most 8."""
    for e in cat.entities:
        if e.parent_id is None:
            continue
        if e.parent_id == UNKNOWN_ID or index.entity(e.parent_id) is None:
            raise InvalidChange("bad parent", item=e.label)
        seen = {e.type_id}
        depth = 0
        cur: Entity | None = e
        while cur is not None and cur.parent_id is not None:
            if cur.parent_id in seen:
                raise InvalidChange("hierarchy cycle", item=e.label)
            seen.add(cur.parent_id)
            depth += 1
            cur = index.entity(cur.parent_id)
        if depth > MAX_DEPTH:
            raise InvalidChange(f"hierarchy deeper than {MAX_DEPTH}", item=e.label)


def _check_patterns(cat: Ontology, index: _CatalogIndex) -> None:
    """Patterns are unique per relation and end at real, non-Unknown entity types.

    An active pattern may not end at a deprecated type unless that type was merged away.
    """
    for r in cat.relations:
        seen: set[tuple[int, int]] = set()
        for p in r.patterns:
            if (p.src, p.tgt) in seen:
                raise InvalidChange("duplicate pattern", item=r.label)
            seen.add((p.src, p.tgt))
            for end in (p.src, p.tgt):
                e = index.entity(end)
                if e is None or end == UNKNOWN_ID:
                    raise InvalidChange(
                        "pattern endpoint is missing or Unknown",
                        item=f"{r.label}({p.src},{p.tgt})",
                    )
                if (
                    p.status is Status.ACTIVE
                    and e.status is not Status.ACTIVE
                    and e.merged_into is None
                ):
                    raise InvalidChange(
                        f"active pattern ends at deprecated type {e.label}",
                        item=r.label,
                        hint="deprecate the pattern first",
                    )


def _check_merges(cat: Ontology, index: _CatalogIndex) -> None:
    """Every ``merged_into`` points at a deprecated source and an active live target."""
    for kind in (Kind.ENTITY, Kind.RELATION):
        for t in cat.types(kind):
            if t.merged_into is None:
                continue
            tgt = index.get(kind, t.merged_into)
            if tgt is None or t.status is not Status.DEPRECATED:
                raise InvalidChange("bad merge pointer", item=t.label)
            live = index.follow(kind, t.type_id)
            final = index.get(kind, live)
            if (
                final is None
                or final.type_id == t.type_id
                or final.status is not Status.ACTIVE
            ):
                raise InvalidChange("merge target is not active", item=t.label)


check_catalog = _check
"""Run every catalog check; used by ``OntologyManager.load_catalog``."""
