"""The one commit path: apply a change set to a catalog and check the result.

Every way in (user API, accepted proposal, RDF import, table mapping) builds a
``ChangeSet`` and ends here. The function is pure; storing the result is the
store's job.
"""

from collections import Counter
from collections.abc import Callable, Iterable
from typing import Any, TypeVar

from ontogr_mem.ontology.errors import InvalidChange, VersionConflict
from ontogr_mem.ontology.models import (
    UNKNOWN_ID,
    AddAlias,
    AddAttribute,
    AddEntity,
    AddPattern,
    AddRelation,
    AddTable,
    Attribute,
    ChangeSet,
    Channel,
    DeprecateAttribute,
    DeprecatePattern,
    DeprecateType,
    Entity,
    Kind,
    MergeInto,
    Mode,
    Ontology,
    Op,
    Owner,
    Pattern,
    PatternSpec,
    Reactivate,
    Ref,
    Relation,
    Rename,
    SetDescription,
    SetExclusive,
    SetModes,
    Status,
    UnknownEdges,
)
from ontogr_mem.ontology.names import (
    MAX_NAME_LEN,
    RESERVED_ATTRIBUTES,
    RESERVED_LABELS,
    check_identifier,
    normalize,
)


MAX_DEPTH = 8


def apply_changeset(cat: Ontology, cs: ChangeSet) -> Ontology:
    """Return the catalog that results from applying ``cs`` to ``cat``.

    The change set is applied to a deep copy, one op at a time. Then the rule that
    makes a catalog strict when it first gets patterns runs, and the whole result is
    checked. The input catalog is never modified, so a failed change leaves nothing
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
        _apply_op(work, op, cs.channel, index)
    _default_strict_on_first_patterns(cat, work, cs)
    _check_indexed(work, index)
    out = work.model_copy(
        update={"version": cat.version + 1, "parent_version": cat.version}
    )
    return out.sealed()


def _edges_for(mode: Mode) -> UnknownEdges:
    """Default ``unknown_endpoint_edges`` for a mode: drop in strict, else keep."""
    return UnknownEdges.DROP if mode is Mode.STRICT else UnknownEdges.KEEP


def _default_strict_on_first_patterns(
    before: Ontology, work: Ontology, cs: ChangeSet
) -> None:
    """Relations become strict when a catalog first gets patterns.

    Only a user or import change set does this, and only if it does not set the
    modes itself. A proposal never changes enforcement.
    """
    if cs.channel not in (Channel.USER, Channel.IMPORT):
        return
    if any(isinstance(o, SetModes) for o in cs.ops):
        return
    if work.has_patterns() and not before.has_patterns():
        work.relation_mode = Mode.STRICT
        work.unknown_endpoint_edges = UnknownEdges.DROP


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


def _new_attribute(attr: Attribute, channel: Channel) -> Attribute:
    """Check who may declare an attribute and normalize a table-owned one.

    System attributes are never declared. A proposal cannot add a table-owned
    attribute. A table-owned attribute is forced to be structured and not extractable,
    which keeps it out of prose prompts.
    """
    if attr.owner is Owner.SYSTEM:
        raise InvalidChange(
            "system attributes are not declared",
            item=attr.name,
            hint="remove the attribute or use owner=user",
        )
    if attr.owner is Owner.TABLE and channel is Channel.PROPOSAL:
        raise InvalidChange(
            "a proposal cannot add table-owned attributes",
            item=attr.name,
            hint="declare it through a table mapping change",
        )
    if attr.owner is Owner.TABLE:
        return attr.model_copy(update={"structured": True, "extractable": False})
    return attr


def _add_attribute(t: Entity | Relation, attr: Attribute) -> None:
    """Append an attribute to a type.

    An identical existing attribute is a no-op. A different definition under the same
    name is refused: add a new attribute and deprecate the old one.
    """
    for existing in t.properties:
        if normalize(existing.name) == normalize(attr.name):
            if existing == attr:
                return
            raise InvalidChange(
                "attribute already exists with a different definition",
                item=f"{t.label}.{attr.name}",
                hint="add a new attribute and deprecate the old one",
            )
    t.properties.append(attr)


OpHandler = Callable[[Ontology, Any, Channel, _CatalogIndex], None]
"""What one op does to the working catalog.

Every handler takes the op it handles, so the op classes stay typed where the
handler is written.
"""


def _apply_op(work: Ontology, op: Op, channel: Channel, index: _CatalogIndex) -> None:
    """Apply one op to the working catalog ``work``, in place.

    The op vocabulary is a closed union, so the handler is looked up by the op's
    class. An op that repeats existing state is a no-op. Rules that need the whole
    catalog are checked afterwards by ``_check``.
    """
    _HANDLERS[type(op)](work, op, channel, index)


def _apply_add_entity(
    work: Ontology, op: AddEntity, channel: Channel, index: _CatalogIndex
) -> None:
    """Add an entity type."""
    check_identifier(op.label, kind="entity")
    parent = _entity(work, op.parent, index).type_id if op.parent is not None else None
    ent = Entity(
        type_id=work.next_entity_id,
        label=op.label,
        description=op.description,
        aliases=list(op.aliases),
        parent_id=parent,
        uri=op.uri,
        origin=op.origin,
    )
    if op.uri and any(e.uri == op.uri for e in work.entities):
        raise InvalidChange("uri already used", item=op.uri)
    work.next_entity_id += 1
    for a in op.properties:
        _add_attribute(ent, _new_attribute(a, channel))
    work.entities.append(ent)
    index.add_entity(ent)


def _apply_add_relation(
    work: Ontology, op: AddRelation, channel: Channel, index: _CatalogIndex
) -> None:
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
        uri=op.uri,
        origin=op.origin,
    )
    if op.uri and any(r.uri == op.uri for r in work.relations):
        raise InvalidChange("uri already used", item=op.uri)
    work.next_relation_id += 1
    for a in op.properties:
        _add_attribute(rel, _new_attribute(a, channel))
    work.relations.append(rel)
    index.add_relation(rel)


def _apply_add_attribute(
    work: Ontology, op: AddAttribute, channel: Channel, index: _CatalogIndex
) -> None:
    """Add an attribute to an existing type."""
    t = _find(work, op.kind, op.target, index)
    _no_unknown(work, op.kind, t)
    _add_attribute(t, _new_attribute(op.attribute, channel))


def _apply_add_pattern(
    work: Ontology, op: AddPattern, channel: Channel, index: _CatalogIndex
) -> None:
    """Add an ordered pattern to a relation type."""
    rel = _relation(work, op.relation, index)
    src, tgt = _pattern(work, op.pattern, index)
    for p in rel.patterns:
        if (p.src, p.tgt) == (src, tgt):
            p.status = Status.ACTIVE
            return
    rel.patterns.append(Pattern(src=src, tgt=tgt))


def _apply_add_alias(
    work: Ontology, op: AddAlias, channel: Channel, index: _CatalogIndex
) -> None:
    """Add an alias to a type."""
    t = _find(work, op.kind, op.target, index)
    _no_unknown(work, op.kind, t)
    alias = op.alias.strip()
    if not alias or len(alias) > MAX_NAME_LEN:
        raise InvalidChange("illegal alias", item=op.alias[:20])
    if normalize(alias) not in work.labels_of(t):
        t.aliases.append(alias)


def _apply_rename(
    work: Ontology, op: Rename, channel: Channel, index: _CatalogIndex
) -> None:
    """Change a type's label; the old label stays as an alias."""
    t = _find(work, op.kind, op.target, index)
    _no_unknown(work, op.kind, t)
    check_identifier(op.new_label, kind=op.kind.value)
    t.aliases = [a for a in t.aliases if normalize(a) != normalize(op.new_label)]
    if normalize(t.label) != normalize(op.new_label):
        t.aliases.append(t.label)
    t.label = op.new_label


def _apply_set_description(
    work: Ontology, op: SetDescription, channel: Channel, index: _CatalogIndex
) -> None:
    """Replace a type's description."""
    t = _find(work, op.kind, op.target, index)
    _no_unknown(work, op.kind, t)
    t.description = op.description


def _apply_set_exclusive(
    work: Ontology, op: SetExclusive, channel: Channel, index: _CatalogIndex
) -> None:
    """Set whether a subject holds at most one current value of a relation."""
    t = _relation(work, op.target, index)
    _no_unknown(work, Kind.RELATION, t)
    t.exclusive = op.exclusive


def _apply_deprecate_type(
    work: Ontology, op: DeprecateType, channel: Channel, index: _CatalogIndex
) -> None:
    """Retire a type; it stays resolvable but is no longer offered."""
    t = _find(work, op.kind, op.target, index)
    _no_unknown(work, op.kind, t)
    _refuse_if_mapped(work, op.kind, t)
    t.status = Status.DEPRECATED


def _apply_deprecate_attribute(
    work: Ontology, op: DeprecateAttribute, channel: Channel, index: _CatalogIndex
) -> None:
    """Retire an attribute."""
    t = _find(work, op.kind, op.target, index)
    attr = next(
        (a for a in t.properties if normalize(a.name) == normalize(op.name)), None
    )
    if attr is None:
        raise InvalidChange("no such attribute", item=f"{t.label}.{op.name}")
    for m in work.tables:
        if (
            isinstance(t, Entity)
            and m.entity_id == t.type_id
            and attr.name in m.columns.values()
        ):
            raise InvalidChange(
                "a table mapping uses this attribute",
                item=f"{t.label}.{attr.name}",
            )
    attr.status = Status.DEPRECATED


def _apply_deprecate_pattern(
    work: Ontology, op: DeprecatePattern, channel: Channel, index: _CatalogIndex
) -> None:
    """Retire a pattern."""
    rel = _relation(work, op.relation, index)
    src, tgt = _pattern(work, op.pattern, index)
    hit = [p for p in rel.patterns if (p.src, p.tgt) == (src, tgt)]
    if not hit:
        raise InvalidChange("no such pattern", item=f"{rel.label}({src},{tgt})")
    hit[0].status = Status.DEPRECATED


def _apply_reactivate(
    work: Ontology, op: Reactivate, channel: Channel, index: _CatalogIndex
) -> None:
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


def _apply_merge_into(
    work: Ontology, op: MergeInto, channel: Channel, index: _CatalogIndex
) -> None:
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
    _refuse_if_mapped(work, op.kind, src_t)
    src_t.status = Status.DEPRECATED
    src_t.merged_into = final
    for t in work.types(op.kind):
        if t.merged_into == src_t.type_id:
            t.merged_into = final


def _apply_set_modes(
    work: Ontology, op: SetModes, channel: Channel, index: _CatalogIndex
) -> None:
    """Change the enforcement modes."""
    if op.entity_mode is not None:
        work.entity_mode = op.entity_mode
    if op.relation_mode is not None:
        work.relation_mode = op.relation_mode
        if op.unknown_endpoint_edges is None:
            work.unknown_endpoint_edges = _edges_for(op.relation_mode)
    if op.unknown_endpoint_edges is not None:
        work.unknown_endpoint_edges = op.unknown_endpoint_edges


def _apply_add_table(
    work: Ontology, op: AddTable, channel: Channel, index: _CatalogIndex
) -> None:
    """Declare a table mapping; a mapping that is already there is a no-op."""
    if op.table not in work.tables:
        work.tables.append(op.table)


_HANDLERS: dict[type[Op], OpHandler] = {
    AddEntity: _apply_add_entity,
    AddRelation: _apply_add_relation,
    AddAttribute: _apply_add_attribute,
    AddPattern: _apply_add_pattern,
    AddAlias: _apply_add_alias,
    Rename: _apply_rename,
    SetDescription: _apply_set_description,
    SetExclusive: _apply_set_exclusive,
    DeprecateType: _apply_deprecate_type,
    DeprecateAttribute: _apply_deprecate_attribute,
    DeprecatePattern: _apply_deprecate_pattern,
    Reactivate: _apply_reactivate,
    MergeInto: _apply_merge_into,
    SetModes: _apply_set_modes,
    AddTable: _apply_add_table,
}


def _refuse_if_mapped(work: Ontology, kind: Kind, t: Entity | Relation) -> None:
    """Refuse to deprecate or merge an entity type that a table mapping names."""
    if kind is Kind.ENTITY:
        for m in work.tables:
            if m.entity_id == t.type_id:
                raise InvalidChange(
                    "a table mapping names this type",
                    item=f"{t.label} <- {m.source}",
                    hint="remove the mapping first",
                )


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
    typed: list[Entity | Relation] = [*cat.entities, *cat.relations]
    for t in typed:
        _check_attributes(t)
    _check_hierarchy(cat, index)
    _check_patterns(cat, index)
    _check_merges(cat, index)
    _check_modes(cat)
    _check_tables(cat, index)


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


def _check_attributes(t: Entity | Relation) -> None:
    """Attribute names are legal, not reserved and unique within a type.

    Also checks that ``max_length`` is positive, that no attribute is system-owned, and
    that table-owned attributes are marked structured.
    """
    names: Counter[str] = Counter()
    for a in t.properties:
        check_identifier(a.name, kind="attribute")
        if a.name.casefold() in RESERVED_ATTRIBUTES:
            raise InvalidChange("reserved attribute name", item=f"{t.label}.{a.name}")
        names[normalize(a.name)] += 1
        if a.max_length is not None and a.max_length < 1:
            raise InvalidChange("max_length must be positive", item=a.name)
        if a.owner is Owner.SYSTEM:
            raise InvalidChange("system attributes are not declared", item=a.name)
        if a.owner is Owner.TABLE and not a.structured:
            raise InvalidChange("table-owned attributes are structured", item=a.name)
    dup = [n for n, c in names.items() if c > 1]
    if dup:
        raise InvalidChange("duplicate attribute", item=f"{t.label}.{dup[0]}")


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


def _check_modes(cat: Ontology) -> None:
    """Strict mode needs at least one active type of its kind to enforce."""
    if cat.entity_mode is Mode.STRICT and not [
        e for e in cat.active_entities() if e.type_id != UNKNOWN_ID
    ]:
        raise InvalidChange(
            "strict entity mode over a catalog with no active types",
            hint="add types first, or use preferred",
        )
    if cat.relation_mode is Mode.STRICT and not cat.active_relations():
        raise InvalidChange(
            "strict relation mode over a catalog with no active relations",
            hint="add relations first, or use preferred",
        )


def _check_tables(cat: Ontology, index: _CatalogIndex) -> None:
    """Table mappings have unique sources and map only table-owned attributes.

    Each mapping must name an active entity type, and every mapped attribute must exist,
    be active and be owned by the table.
    """
    sources: set[str] = set()
    for m in cat.tables:
        if m.source in sources:
            raise InvalidChange("two mappings share a source", item=m.source)
        sources.add(m.source)
        e = index.entity(m.entity_id)
        if e is None or e.status is not Status.ACTIVE or m.entity_id == UNKNOWN_ID:
            raise InvalidChange("mapping names an unusable entity", item=m.source)
        by_name = {a.name: a for a in e.properties if a.status is Status.ACTIVE}
        for col, attr in m.columns.items():
            a = by_name.get(attr)
            if not col or a is None or a.owner is not Owner.TABLE:
                raise InvalidChange(
                    "mapped attribute is missing or not table-owned",
                    item=f"{m.source}.{col}->{attr}",
                )


check_catalog = _check
"""Run every catalog check; used by ``OntologyManager.load_catalog``."""
