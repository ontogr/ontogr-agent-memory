"""Structural diff between two versions of the ontology."""

from dataclasses import dataclass, field

from ontogr_mem.ontology.models import (
    AddAlias,
    AddAttribute,
    AddEntity,
    AddPattern,
    AddRelation,
    AddTable,
    ChangeSet,
    DeprecateAttribute,
    DeprecatePattern,
    DeprecateType,
    Entity,
    Kind,
    MergeInto,
    Ontology,
    Op,
    PatternSpec,
    Reactivate,
    Ref,
    Relation,
    Rename,
    SetDescription,
    SetExclusive,
    SetModes,
    Status,
)
from ontogr_mem.ontology.names import normalize


@dataclass
class _Ops:
    """The ops one diff collects, grouped by what they change."""

    edits: list[Op] = field(default_factory=list)
    attrs: list[Op] = field(default_factory=list)
    pats: list[Op] = field(default_factory=list)
    states: list[Op] = field(default_factory=list)


def diff(a: Ontology, b: Ontology) -> ChangeSet:
    """Return the ops that turn catalog ``a`` into catalog ``b``.

    Replaying the ops on ``a`` gives a catalog with the same content hash as ``b``.
    Order matters: additions come first, then edits, attributes, patterns, status
    changes, and finally modes and tables. A type that ``a`` lacks is referred to by
    its label, because the replay assigns the id. ``diff`` assumes ``b`` descends from
    ``a`` through change sets, so ids only grow.

    Args:
        a: The older catalog; the change set's base.
        b: The newer catalog.

    Returns:
        A change set with ``base_version`` equal to ``a.version``. It has no ops when
        the catalogs have the same content.
    """
    adds: list[Op] = []
    ops = _Ops()
    a_ent = {e.type_id: e for e in a.entity_types()}
    a_rel = {r.type_id: r for r in a.relation_types()}

    for e in b.entity_types():
        if e.type_id not in a_ent:
            adds.append(
                AddEntity(
                    label=e.label,
                    description=e.description,
                    aliases=e.aliases,
                    parent=e.parent_id,
                    uri=e.uri,
                    properties=e.properties,
                    origin=e.origin,
                )
            )
    for r in b.relation_types():
        if r.type_id not in a_rel:
            adds.append(
                AddRelation(
                    label=r.label,
                    description=r.description,
                    aliases=r.aliases,
                    symmetric=r.symmetric,
                    exclusive=r.exclusive,
                    uri=r.uri,
                    properties=r.properties,
                    patterns=[PatternSpec(src=p.src, tgt=p.tgt) for p in r.patterns],
                    origin=r.origin,
                )
            )
    _diff_entities(b.entity_types(), a_ent, ops)
    _diff_relations(b.relation_types(), a_rel, ops)
    tail: list[Op] = []
    adds_patterns = bool(ops.pats) or any(
        isinstance(o, AddRelation) and o.patterns for o in adds
    )
    if adds_patterns or (a.entity_mode, a.relation_mode, a.unknown_endpoint_edges) != (
        b.entity_mode,
        b.relation_mode,
        b.unknown_endpoint_edges,
    ):
        tail.append(
            SetModes(
                entity_mode=b.entity_mode,
                relation_mode=b.relation_mode,
                unknown_endpoint_edges=b.unknown_endpoint_edges,
            )
        )
    tail.extend(AddTable(table=m) for m in b.tables if m not in a.tables)
    replayed = _by_label_when_new(
        a, b, [*adds, *ops.edits, *ops.attrs, *ops.pats, *ops.states, *tail]
    )
    return ChangeSet(
        base_version=a.version,
        ops=replayed,
        actor="diff",
        reason=f"{a.version}->{b.version}",
    )


def _status(kind: Kind, t: Entity | Relation, out: list[Op], *, created: bool) -> None:
    """Emit the op that gives a type its status: merge, deprecate or reactivate."""
    if t.merged_into is not None:
        out.append(MergeInto(kind=kind, source=t.type_id, into=t.merged_into))
    elif t.status is Status.DEPRECATED:
        out.append(DeprecateType(kind=kind, target=t.type_id))
    elif not created:
        out.append(Reactivate(kind=kind, target=t.type_id))


def _diff_entities(new: list[Entity], old_map: dict[int, Entity], ops: _Ops) -> None:
    """Emit the ops for the entity types that changed.

    A type the older catalog does not have is new: it gets a status op when it was
    deprecated or merged, and its deprecated attributes are retired.
    """
    for t in new:
        old = old_map.get(t.type_id)
        if old is not None:
            _diff_existing(Kind.ENTITY, old, t, ops)
            continue
        _diff_new(Kind.ENTITY, t, ops)


def _diff_relations(
    new: list[Relation], old_map: dict[int, Relation], ops: _Ops
) -> None:
    """Emit the ops for the relation types that changed.

    A relation's patterns and its exclusive flag are covered here; a relation the
    older catalog does not have is new, and its deprecated patterns and attributes
    are retired.
    """
    for t in new:
        old = old_map.get(t.type_id)
        if old is not None:
            _diff_existing(Kind.RELATION, old, t, ops)
            _diff_relation(old, t, ops)
            continue
        _diff_new(Kind.RELATION, t, ops)
        for p in t.patterns:
            if p.status is Status.DEPRECATED:
                ops.pats.append(
                    DeprecatePattern(
                        relation=t.type_id,
                        pattern=PatternSpec(src=p.src, tgt=p.tgt),
                    )
                )


def _diff_new(kind: Kind, t: Entity | Relation, ops: _Ops) -> None:
    """Emit the ops for a type the older catalog does not have.

    A new type that is already deprecated or merged gets its status, and its
    deprecated attributes are retired.
    """
    if t.status is Status.DEPRECATED or t.merged_into is not None:
        _status(kind, t, ops.states, created=True)
    for at in t.properties:
        if at.status is Status.DEPRECATED:
            ops.attrs.append(
                DeprecateAttribute(kind=kind, target=t.type_id, name=at.name)
            )


def _diff_existing(
    kind: Kind, old: Entity | Relation, new: Entity | Relation, ops: _Ops
) -> None:
    """Emit ops for a type that exists in both catalogs.

    Covers rename (the old label becomes an alias), new aliases, a changed description,
    new or deprecated attributes, and a status or merge change. A relation's patterns
    and its exclusive flag are covered by ``_diff_relation``.
    """
    have = {normalize(old.label), *(normalize(x) for x in old.aliases)}
    if new.label != old.label:
        ops.edits.append(Rename(kind=kind, target=old.type_id, new_label=new.label))
        have = {normalize(new.label), normalize(old.label)} | {
            normalize(x) for x in old.aliases if normalize(x) != normalize(new.label)
        }
    for al in new.aliases:
        if normalize(al) not in have:
            ops.edits.append(AddAlias(kind=kind, target=old.type_id, alias=al))
    if new.description != old.description:
        ops.edits.append(
            SetDescription(kind=kind, target=old.type_id, description=new.description)
        )
    old_attrs = {a.name: a for a in old.properties}
    for at in new.properties:
        prev = old_attrs.get(at.name)
        if prev is None:
            ops.attrs.append(AddAttribute(kind=kind, target=old.type_id, attribute=at))
            if at.status is Status.DEPRECATED:
                ops.attrs.append(
                    DeprecateAttribute(kind=kind, target=old.type_id, name=at.name)
                )
        elif prev.status is Status.ACTIVE and at.status is Status.DEPRECATED:
            ops.attrs.append(
                DeprecateAttribute(kind=kind, target=old.type_id, name=at.name)
            )
    if (old.status, old.merged_into) != (new.status, new.merged_into):
        _status(kind, new, ops.states, created=False)


def _diff_relation(old: Relation, new: Relation, ops: _Ops) -> None:
    """Emit the ops for a relation that exists in both catalogs.

    Covers new, deprecated or reactivated patterns, and a change of the exclusive
    flag.
    """
    old_p = {(p.src, p.tgt): p for p in old.patterns}
    for p in new.patterns:
        spec = PatternSpec(src=p.src, tgt=p.tgt)
        prev_p = old_p.get((p.src, p.tgt))
        if prev_p is None:
            ops.pats.append(AddPattern(relation=old.type_id, pattern=spec))
            if p.status is Status.DEPRECATED:
                ops.pats.append(DeprecatePattern(relation=old.type_id, pattern=spec))
        elif prev_p.status != p.status:
            ops.pats.append(
                AddPattern(relation=old.type_id, pattern=spec)
                if p.status is Status.ACTIVE
                else DeprecatePattern(relation=old.type_id, pattern=spec)
            )
    if new.exclusive != old.exclusive:
        ops.edits.append(SetExclusive(target=old.type_id, exclusive=new.exclusive))


def _by_label_when_new(a: Ontology, b: Ontology, ops: list[Op]) -> list[Op]:
    """Refer to a type that ``a`` does not have by its label in ``b``.

    A new type gets whatever id the replay assigns, which need not be the id it
    has in ``b``. Its label is the same in both.
    """
    have = {
        Kind.ENTITY: {e.type_id for e in a.entity_types()},
        Kind.RELATION: {r.type_id for r in a.relation_types()},
    }
    label = {
        Kind.ENTITY: {e.type_id: e.label for e in b.entity_types()},
        Kind.RELATION: {r.type_id: r.label for r in b.relation_types()},
    }

    def fix(kind: Kind, ref: Ref) -> Ref:
        """Return ``ref``, or its label in ``b`` when ``a`` has no such id."""
        if isinstance(ref, int) and ref not in have[kind]:
            return label[kind][ref]
        return ref

    def spec(sp: PatternSpec) -> PatternSpec:
        """Apply ``fix`` to both endpoints of a pattern spec."""
        return PatternSpec(src=fix(Kind.ENTITY, sp.src), tgt=fix(Kind.ENTITY, sp.tgt))

    out: list[Op] = []
    for op in ops:
        fixed = op
        if isinstance(op, AddEntity):
            parent = None if op.parent is None else fix(Kind.ENTITY, op.parent)
            fixed = op.model_copy(update={"parent": parent})
        elif isinstance(op, AddRelation):
            fixed = op.model_copy(update={"patterns": [spec(p) for p in op.patterns]})
        elif isinstance(op, AddPattern | DeprecatePattern):
            fixed = op.model_copy(
                update={
                    "relation": fix(Kind.RELATION, op.relation),
                    "pattern": spec(op.pattern),
                }
            )
        elif isinstance(op, SetExclusive):
            fixed = op.model_copy(update={"target": fix(Kind.RELATION, op.target)})
        elif isinstance(op, MergeInto):
            fixed = op.model_copy(
                update={
                    "source": fix(op.kind, op.source),
                    "into": fix(op.kind, op.into),
                }
            )
        elif isinstance(
            op,
            AddAttribute
            | DeprecateAttribute
            | DeprecateType
            | Reactivate
            | Rename
            | SetDescription
            | AddAlias,
        ):
            fixed = op.model_copy(update={"target": fix(op.kind, op.target)})
        out.append(fixed)
    return out
