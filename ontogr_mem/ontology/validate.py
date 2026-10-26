"""Check one episode's extraction output against the catalog.

``validate`` is pure. Positions the LLM returned are decoded through the
``RenderedView`` the prompt was built from; a position outside it is
``type_id_out_of_range`` and a counted drop, never a fallback to a generic type.

What happens to a label the catalog does not know depends on the mode of its kind:
strict drops and counts it, preferred records it as an unmatched row, and open
lists it in ``ValidationReport.to_register`` so the manager can add the type.
"""

from collections import Counter
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import date, datetime
from enum import StrEnum

from ontogr_mem.ontology.errors import InvalidChange
from ontogr_mem.ontology.models import (
    UNKNOWN_ID,
    Attribute,
    Entity,
    Kind,
    Mode,
    Ontology,
    Reason,
    Relation,
    RenderedView,
    Status,
    UnknownEdges,
    Unmatched,
    UnmatchedKind,
    ValueType,
)
from ontogr_mem.ontology.names import RESERVED_LABELS, check_identifier, normalize
from ontogr_mem.ontology.resolve import How, TypeMatch, TypeResolver


AttributeValue = str | int | float | bool | list[str]
"""A value an attribute holds once it has been coerced to its declared type."""


@dataclass(frozen=True, slots=True)
class EntityItem:
    """An entity the extractor produced.

    Give ``position`` when the LLM chose from the numbered list. Give ``raw_type`` when
    the extractor returned a label (GLiNER, imports).
    """

    local_id: str
    position: int | None = None
    raw_type: str | None = None
    attributes: Mapping[str, object] = field(default_factory=dict)
    quote: str = ""


@dataclass(frozen=True, slots=True)
class RelationItem:
    """A relation between two entity ``local_id`` values, typed by position or label."""

    source: str
    target: str
    position: int | None = None
    raw_label: str | None = None
    attributes: Mapping[str, object] = field(default_factory=dict)
    quote: str = ""


@dataclass(frozen=True, slots=True)
class ExtractedItems:
    """One episode's extraction output: its entities and relations."""

    entities: tuple[EntityItem, ...] = ()
    relations: tuple[RelationItem, ...] = ()


@dataclass(frozen=True, slots=True)
class Context:
    """Where the items came from; recorded on every unmatched row."""

    episode_id: str


@dataclass(frozen=True, slots=True)
class EntityValidation:
    """Outcome for one entity.

    ``type_id`` is None when it was dropped, or when it was kept in open mode with a
    label that still has to be registered. ``how`` and ``score`` tell how a label
    matched.
    """

    local_id: str
    kept: bool
    type_id: int | None
    attributes: dict[str, AttributeValue]
    reasons: tuple[Reason, ...] = ()
    how: How | None = None
    score: float | None = None


@dataclass(frozen=True, slots=True)
class RelationValidation:
    """Outcome for one relation.

    ``unvalidated`` marks a relation kept in preferred mode with a label the catalog
    does not declare. ``reasons`` lists why it was dropped or flagged. In open mode
    a kept relation can have no ``type_id`` until its label is registered.
    """

    source: str
    target: str
    kept: bool
    type_id: int | None
    attributes: dict[str, AttributeValue]
    unvalidated: bool = False
    reasons: tuple[Reason, ...] = ()
    how: How | None = None
    score: float | None = None


@dataclass(frozen=True, slots=True)
class ValidationReport:
    """Per-item verdicts, reason counters, and what the manager must act on.

    ``unmatched`` is filled only for kinds in preferred mode. ``to_register`` is
    filled only for kinds in open mode and holds each new label once.
    """

    entities: tuple[EntityValidation, ...]
    relations: tuple[RelationValidation, ...]
    counters: Mapping[Reason, int]
    unmatched: tuple[Unmatched, ...]
    to_register: tuple[tuple[Kind, str], ...] = ()

    @property
    def dropped(self) -> tuple[str, ...]:
        """Local ids of dropped entities and ``src->tgt`` keys of dropped relations."""
        return tuple(
            [v.local_id for v in self.entities if not v.kept]
            + [f"{v.source}->{v.target}" for v in self.relations if not v.kept]
        )


class _UncoercibleError(Exception):
    """A value cannot be coerced to the declared attribute type."""


def _coerce_string(value: object) -> str:
    """Return ``value`` as text, or fail."""
    if isinstance(value, str):
        return value
    raise _UncoercibleError


def _coerce_boolean(value: object) -> bool:
    """Return ``value`` as a boolean, or fail."""
    if isinstance(value, bool):
        return value
    raise _UncoercibleError


def _coerce_integer(value: object) -> int:
    """Return ``value`` as a whole number, or fail.

    A boolean is not a number here, and a float with a fractional part does not
    round.
    """
    if isinstance(value, bool):
        raise _UncoercibleError
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    raise _UncoercibleError


def _coerce_float(value: object) -> float:
    """Return ``value`` as a number, or fail."""
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise _UncoercibleError
    return float(value)


def _coerce_date(value: object) -> str:
    """Return ``value`` as ISO date text, or fail.

    A datetime is not a date.
    """
    if isinstance(value, datetime):
        raise _UncoercibleError
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, str):
        try:
            return date.fromisoformat(value).isoformat()
        except ValueError:
            raise _UncoercibleError from None
    raise _UncoercibleError


def _coerce_datetime(value: object) -> str:
    """Return ``value`` as ISO datetime text, or fail."""
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value).isoformat()
        except ValueError:
            raise _UncoercibleError from None
    raise _UncoercibleError


def _coerce_list(value: object) -> list[str]:
    """Return ``value`` as a list of text, or fail.

    A single value is not wrapped in a list.
    """
    if isinstance(value, list | tuple):
        return [str(v) for v in value]
    raise _UncoercibleError


_COERCERS: dict[ValueType, Callable[[object], AttributeValue]] = {
    ValueType.STRING: _coerce_string,
    ValueType.BOOLEAN: _coerce_boolean,
    ValueType.INTEGER: _coerce_integer,
    ValueType.FLOAT: _coerce_float,
    ValueType.DATE: _coerce_date,
    ValueType.DATETIME: _coerce_datetime,
    ValueType.LIST: _coerce_list,
}


def coerce(value: object, vt: ValueType) -> AttributeValue:
    """Coerce ``value`` to the attribute type, strictly.

    Integers with a fractional part fail, booleans are not numbers, and a list is not
    wrapped for you. Dates and datetimes return ISO text.

    Raises:
        _UncoercibleError: The value does not fit ``vt``.
    """
    return _COERCERS[vt](value)


def _attributes(
    t: Entity | Relation,
    raw: Mapping[str, object],
    reasons: list[Reason],
    undeclared: list[str],
) -> dict[str, AttributeValue]:
    """Coerce an item's attributes against the type's extractable attributes.

    Undeclared names are omitted and their names added to ``undeclared``. Uncoercible or
    too-long values are omitted. A failed value is never stored as None. Each problem is
    appended to ``reasons``, including a missing required attribute.
    """
    declared: dict[str, Attribute] = {
        normalize(a.name): a
        for a in t.properties
        if a.status is Status.ACTIVE and a.extractable and not a.structured
    }
    out: dict[str, AttributeValue] = {}
    for key, value in raw.items():
        a = declared.get(normalize(key))
        if a is None:
            reasons.append(Reason.ATTRIBUTE_UNDECLARED)
            undeclared.append(key)
            continue
        try:
            val = coerce(value, a.value_type)
        except _UncoercibleError:
            reasons.append(Reason.ATTRIBUTE_UNCOERCIBLE)
            continue
        if a.max_length is not None and len(str(val)) > a.max_length:
            reasons.append(Reason.ATTRIBUTE_TOO_LONG)
            continue
        out[a.name] = val
    for a in declared.values():
        if a.required and a.name not in out:
            reasons.append(Reason.ATTRIBUTE_REQUIRED_MISSING)
    return out


class _Outcome(StrEnum):
    """What a mode does with an item whose type the catalog does not accept."""

    DROP = "drop"
    KEEP = "keep"
    AS_UNKNOWN = "as_unknown"
    UNVALIDATED = "unvalidated"
    REGISTER = "register"


_UNPLACED: dict[tuple[Kind, Mode], _Outcome] = {
    (Kind.ENTITY, Mode.STRICT): _Outcome.DROP,
    (Kind.ENTITY, Mode.PREFERRED): _Outcome.AS_UNKNOWN,
    (Kind.ENTITY, Mode.OPEN): _Outcome.REGISTER,
    (Kind.RELATION, Mode.STRICT): _Outcome.DROP,
    (Kind.RELATION, Mode.PREFERRED): _Outcome.UNVALIDATED,
    (Kind.RELATION, Mode.OPEN): _Outcome.REGISTER,
}

_DEPRECATED: dict[tuple[Kind, Mode], _Outcome] = {
    (Kind.ENTITY, Mode.STRICT): _Outcome.DROP,
    (Kind.ENTITY, Mode.PREFERRED): _Outcome.AS_UNKNOWN,
    (Kind.ENTITY, Mode.OPEN): _Outcome.KEEP,
    (Kind.RELATION, Mode.STRICT): _Outcome.DROP,
    (Kind.RELATION, Mode.PREFERRED): _Outcome.UNVALIDATED,
    (Kind.RELATION, Mode.OPEN): _Outcome.KEEP,
}


def _registrable(label: str) -> bool:
    """True if ``label`` may become a new type name."""
    if normalize(label) in RESERVED_LABELS:
        return False
    try:
        check_identifier(label, kind="type")
    except InvalidChange:
        return False
    return True


def _apply_mode(
    table: dict[tuple[Kind, Mode], _Outcome], kind: Kind, mode: Mode, raw: str
) -> _Outcome:
    """Return what ``mode`` does with an item of ``kind``.

    In open mode a label that is not a legal type name cannot be registered, so the
    item is dropped instead.
    """
    outcome = table[(kind, mode)]
    if outcome is _Outcome.REGISTER and not (raw and _registrable(raw)):
        return _Outcome.DROP
    return outcome


@dataclass(slots=True)
class _State:
    """What one validation pass collects.

    Holds the catalog being checked, the counters, the unmatched rows, the labels the
    manager must register, and the types the entity pass placed.
    """

    cat: Ontology
    ctx: Context
    counters: Counter[Reason] = field(default_factory=Counter)
    unmatched: list[Unmatched] = field(default_factory=list)
    registered: dict[tuple[Kind, str], tuple[Kind, str]] = field(default_factory=dict)
    types: dict[str, int] = field(default_factory=dict)
    kept_ids: set[str] = field(default_factory=set)
    entity_resolver: TypeResolver = field(init=False)
    relation_resolver: TypeResolver = field(init=False)

    def __post_init__(self) -> None:
        """Build the resolvers over the catalog."""
        self.entity_resolver = TypeResolver(self.cat, Kind.ENTITY)
        self.relation_resolver = TypeResolver(self.cat, Kind.RELATION)

    def resolver(self, kind: Kind) -> TypeResolver:
        """Return the resolver for ``kind``."""
        return self.entity_resolver if kind is Kind.ENTITY else self.relation_resolver

    def note(
        self,
        kind: UnmatchedKind,
        raw: str,
        quote: str,
        src: str = "",
        tgt: str = "",
        *,
        mode: Mode,
    ) -> None:
        """Add an unmatched row for ``raw``; only the preferred mode records rows."""
        if mode is not Mode.PREFERRED:
            return
        self.unmatched.append(
            Unmatched(
                kind=kind,
                raw_label=raw,
                normalized_label=normalize(raw),
                src_label=src,
                tgt_label=tgt,
                episode_id=self.ctx.episode_id,
                quote=quote,
                catalog_version=self.cat.version,
            )
        )

    def register(self, kind: Kind, label: str) -> None:
        """Ask the manager to add ``label`` as a type of ``kind``."""
        key = (kind, normalize(label))
        if key not in self.registered:
            self.registered[key] = (kind, label)


def _resolve_type(
    state: _State,
    kind: Kind,
    position: int | None,
    raw: str | None,
    view: RenderedView | None,
) -> tuple[int | None, list[Reason], str, TypeMatch | None]:
    """Resolve one item's type from a position or a raw label.

    Returns the live type id (None for no home), the reasons found, the raw label
    and the match details.
    """
    if position is not None:
        tid = view.decode(position) if view is not None else None
        if tid is None:
            return None, [Reason.TYPE_ID_OUT_OF_RANGE], raw or str(position), None
        return state.cat.follow(kind, tid), [], raw or "", None
    found = state.resolver(kind).resolve(raw or "")
    if found.match is not None:
        return found.match.type_id, [], raw or "", found.match
    reason = Reason.AMBIGUOUS_TYPE if found.ambiguous else Reason.UNKNOWN_TYPE
    return None, [reason], raw or "", None


def _place_entity(
    state: _State,
    it: EntityItem,
    tid: int | None,
    *,
    reasons: list[Reason],
    raw: str,
    mode: Mode,
) -> tuple[bool, int | None]:
    """Decide whether an entity is kept, and under which type.

    A type the catalog does not know, or one it has deprecated, is left to the mode:
    strict drops the entity, preferred keeps it as Unknown, and open registers the
    label. Returns whether the entity is kept and the type id it is kept under.
    """
    if tid is None:
        if raw:
            state.note(UnmatchedKind.ENTITY_TYPE, raw, it.quote, mode=mode)
        outcome = _apply_mode(_UNPLACED, Kind.ENTITY, mode, raw)
        if outcome is _Outcome.REGISTER:
            state.register(Kind.ENTITY, raw)
            return True, None
        if outcome is _Outcome.AS_UNKNOWN:
            return True, UNKNOWN_ID
        return False, None
    ent = state.cat.entity(tid)
    if ent is None or ent.status is not Status.DEPRECATED:
        return True, tid
    reasons.append(Reason.DEPRECATED_TYPE)
    if raw:
        state.note(UnmatchedKind.ENTITY_TYPE, raw, it.quote, mode=mode)
    outcome = _apply_mode(_DEPRECATED, Kind.ENTITY, mode, raw)
    if outcome is _Outcome.AS_UNKNOWN:
        return True, UNKNOWN_ID
    return outcome is not _Outcome.DROP, tid


def _validate_entities(
    state: _State, items: tuple[EntityItem, ...], view: RenderedView | None
) -> list[EntityValidation]:
    """Check every extracted entity against the catalog.

    Returns one verdict per item, in input order, and records the types the kept
    entities were placed in.
    """
    verdicts: list[EntityValidation] = []
    mode = state.cat.entity_mode
    for it in items:
        tid, reasons, raw, match = _resolve_type(
            state, Kind.ENTITY, it.position, it.raw_type, view
        )
        if match is not None and match.how is How.FUZZY:
            reasons.append(Reason.FUZZY_MATCH)
            state.note(UnmatchedKind.FUZZY_ENTITY, raw, it.quote, mode=mode)
        keep = True
        if Reason.TYPE_ID_OUT_OF_RANGE in reasons:
            keep = False
            if it.raw_type:
                state.note(UnmatchedKind.ENTITY_TYPE, it.raw_type, it.quote, mode=mode)
        else:
            keep, tid = _place_entity(
                state, it, tid, reasons=reasons, raw=raw, mode=mode
            )
        attrs: dict[str, AttributeValue] = {}
        if keep and tid is not None:
            ent = state.cat.entity(tid)
            if ent is not None and tid != UNKNOWN_ID:
                missing: list[str] = []
                attrs = _attributes(ent, it.attributes, reasons, missing)
                for k in missing:
                    state.note(
                        UnmatchedKind.ATTRIBUTE, k, it.quote, ent.label, mode=mode
                    )
        for r in reasons:
            state.counters[r] += 1
        verdicts.append(
            EntityValidation(
                it.local_id,
                keep,
                tid if keep else None,
                attrs,
                tuple(reasons),
                match.how if match else None,
                match.score if match else None,
            )
        )
        if keep:
            state.types[it.local_id] = UNKNOWN_ID if tid is None else tid
            state.kept_ids.add(it.local_id)
    return verdicts


def _check_pattern(
    state: _State,
    rel: Relation,
    src_type: int,
    tgt_type: int,
    *,
    reasons: list[Reason],
    quote: str,
) -> bool:
    """Check a relation against the patterns of its type.

    Returns False when the relation must be dropped.
    """
    cat = state.cat
    mode = cat.relation_mode
    src = cat.entity(src_type)
    tgt = cat.entity(tgt_type)
    src_label = src.label if src else ""
    tgt_label = tgt.label if tgt else ""
    has_patterns = any(p.status is Status.ACTIVE for p in rel.patterns)
    if has_patterns and UNKNOWN_ID in (src_type, tgt_type):
        reasons.append(Reason.UNKNOWN_ENDPOINT)
        state.note(
            UnmatchedKind.PATTERN,
            rel.label,
            quote,
            src_label,
            tgt_label,
            mode=mode,
        )
        drop = cat.unknown_endpoint_edges is UnknownEdges.DROP and mode is Mode.STRICT
        return not drop
    if has_patterns and not cat.pattern_allows(rel, src_type, tgt_type):
        if rel.symmetric and cat.pattern_allows(rel, tgt_type, src_type):
            return True
        if cat.pattern_allows(rel, tgt_type, src_type):
            reasons.append(Reason.DIRECTION_SWAPPED)
            return mode is not Mode.STRICT
        reasons.append(Reason.PATTERN_MISMATCH)
        state.note(
            UnmatchedKind.PATTERN,
            rel.label,
            quote,
            src_label,
            tgt_label,
            mode=mode,
        )
        return mode is not Mode.STRICT
    return True


def _place_relation(
    state: _State,
    rit: RelationItem,
    tid: int | None,
    *,
    reasons: list[Reason],
    raw: str,
    mode: Mode,
) -> tuple[bool, int | None, bool]:
    """Decide whether a relation is kept, under which type, and whether it is checked.

    A relation type the catalog does not know, or one it has deprecated, is left to
    the mode: strict drops the relation, preferred keeps it unvalidated, and open
    registers the label.
    """
    if tid is None:
        reasons.clear()
        reasons.append(Reason.UNDECLARED_RELATION)
        state.note(UnmatchedKind.RELATION_TYPE, raw, rit.quote, mode=mode)
        outcome = _apply_mode(_UNPLACED, Kind.RELATION, mode, raw)
        if outcome is _Outcome.REGISTER:
            state.register(Kind.RELATION, raw)
            return True, None, False
        if outcome is _Outcome.UNVALIDATED:
            return True, None, True
        return False, None, False
    rel = state.cat.relation(tid)
    if rel is None or rel.status is not Status.DEPRECATED:
        return True, tid, False
    reasons.append(Reason.DEPRECATED_TYPE)
    outcome = _apply_mode(_DEPRECATED, Kind.RELATION, mode, raw)
    if outcome is _Outcome.UNVALIDATED:
        return True, tid, True
    return outcome is not _Outcome.DROP, tid, False


def _validate_relations(
    state: _State, items: tuple[RelationItem, ...], view: RenderedView | None
) -> list[RelationValidation]:
    """Check every extracted relation against the catalog.

    A relation whose endpoint was dropped is dropped with it. Returns one verdict per
    item, in input order.
    """
    verdicts: list[RelationValidation] = []
    mode = state.cat.relation_mode
    for rit in items:
        reasons: list[Reason] = []
        if rit.source not in state.kept_ids or rit.target not in state.kept_ids:
            reasons.append(Reason.ENDPOINT_DROPPED)
            state.counters[Reason.ENDPOINT_DROPPED] += 1
            verdicts.append(
                RelationValidation(
                    rit.source, rit.target, False, None, {}, False, tuple(reasons)
                )
            )
            continue
        tid, reasons, raw, match = _resolve_type(
            state, Kind.RELATION, rit.position, rit.raw_label, view
        )
        if match is not None and match.how is How.FUZZY:
            reasons.append(Reason.FUZZY_MATCH)
            state.note(UnmatchedKind.FUZZY_RELATION, raw, rit.quote, mode=mode)
        keep = True
        unvalidated = False
        if Reason.TYPE_ID_OUT_OF_RANGE in reasons:
            keep = False
        else:
            keep, tid, unvalidated = _place_relation(
                state, rit, tid, reasons=reasons, raw=raw, mode=mode
            )
        rel_t = state.cat.relation(tid) if tid is not None else None
        if keep and rel_t is not None and mode is not Mode.OPEN:
            keep = _check_pattern(
                state,
                rel_t,
                state.types[rit.source],
                state.types[rit.target],
                reasons=reasons,
                quote=rit.quote,
            )
        attrs: dict[str, AttributeValue] = {}
        if keep and rel_t is not None:
            missing: list[str] = []
            attrs = _attributes(rel_t, rit.attributes, reasons, missing)
            for k in missing:
                state.note(
                    UnmatchedKind.ATTRIBUTE, k, rit.quote, rel_t.label, mode=mode
                )
        for r in reasons:
            state.counters[r] += 1
        verdicts.append(
            RelationValidation(
                rit.source,
                rit.target,
                keep,
                tid if keep else None,
                attrs,
                unvalidated,
                tuple(reasons),
                match.how if match else None,
                match.score if match else None,
            )
        )
    return verdicts


def validate(
    cat: Ontology,
    items: ExtractedItems,
    ctx: Context,
    entity_view: RenderedView | None = None,
    relation_view: RenderedView | None = None,
) -> ValidationReport:
    """Check one episode's extraction output against the catalog.

    Entities are decoded or resolved first, then relations. See the module docstring for
    how positions are decoded. By mode:

    - strict: an unplaced or deprecated type, an undeclared relation, a pattern mismatch
      and a swapped direction are dropped and counted. Nothing is recorded.
    - preferred: an unplaced entity becomes Unknown, an undeclared relation is kept and
      flagged ``unvalidated``, and pattern problems are kept. Unplaced labels are
      recorded as unmatched rows.
    - open: labels the catalog does not know are kept and listed in ``to_register``
      unless the label is not a legal type name, in which case the item is dropped.

    A relation is dropped if either endpoint was dropped. Edges that touch an Unknown
    entity follow the catalog's ``unknown_endpoint_edges``. Fuzzy matches are counted
    and recorded.

    Args:
        cat: The catalog the prompt was built from.
        items: The episode's output.
        ctx: The episode id for the unmatched rows.
        entity_view: The view used to render entity types, to decode positions.
        relation_view: The view used to render relation types.

    Returns:
        Verdicts, counters by reason and the unmatched rows to record.
    """
    state = _State(cat, ctx)
    ent_verdicts = _validate_entities(state, items.entities, entity_view)
    rel_verdicts = _validate_relations(state, items.relations, relation_view)
    return ValidationReport(
        tuple(ent_verdicts),
        tuple(rel_verdicts),
        dict(state.counters),
        tuple(state.unmatched),
        tuple(state.registered.values()),
    )
