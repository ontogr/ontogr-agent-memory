"""Check one episode's extraction output against the ontology.

``validate`` is pure. A relation label that the ontology does not hold is dropped
and counted, and so is a relation whose endpoint types do not fit its patterns. An
entity type that the ontology does not hold, or has deprecated, becomes Unknown and
the entity is kept.
"""

from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass, field

from ontogr_mem.ontology.models import (
    UNKNOWN_ID,
    Kind,
    Ontology,
    Reason,
    Relation,
    Status,
)
from ontogr_mem.ontology.resolve import How, TypeMatch, TypeResolver


@dataclass(frozen=True, slots=True)
class EntityItem:
    """An entity the extractor produced, typed by the label the extractor gave."""

    local_id: str
    raw_type: str | None = None


@dataclass(frozen=True, slots=True)
class RelationItem:
    """A relation between two entity ``local_id`` values, typed by a label.

    An endpoint that is not among the entity items is treated as Unknown.
    """

    source: str
    target: str
    raw_label: str | None = None


@dataclass(frozen=True, slots=True)
class ExtractedItems:
    """One episode's extraction output: its entities and relations."""

    entities: tuple[EntityItem, ...] = ()
    relations: tuple[RelationItem, ...] = ()


@dataclass(frozen=True, slots=True)
class EntityValidation:
    """Outcome for one entity.

    ``type_id`` is the Unknown type when the label is not in the ontology or names a
    deprecated type. ``how`` and ``score`` tell how a label matched.
    """

    local_id: str
    type_id: int
    reasons: tuple[Reason, ...] = ()
    how: How | None = None
    score: float | None = None


@dataclass(frozen=True, slots=True)
class RelationValidation:
    """Outcome for one relation.

    ``type_id`` is None when the relation was dropped. ``reasons`` lists why it was
    dropped or flagged.
    """

    source: str
    target: str
    kept: bool
    type_id: int | None
    reasons: tuple[Reason, ...] = ()
    how: How | None = None
    score: float | None = None


@dataclass(frozen=True, slots=True)
class ValidationReport:
    """Per-item outcomes and reason counters."""

    entities: tuple[EntityValidation, ...]
    relations: tuple[RelationValidation, ...]
    counters: Mapping[Reason, int]


@dataclass(slots=True)
class _State:
    """What one validation pass collects."""

    cat: Ontology
    counters: Counter[Reason] = field(default_factory=Counter)
    types: dict[str, int] = field(default_factory=dict)
    entity_resolver: TypeResolver = field(init=False)
    relation_resolver: TypeResolver = field(init=False)

    def __post_init__(self) -> None:
        """Build the resolvers over the ontology."""
        self.entity_resolver = TypeResolver(self.cat, Kind.ENTITY)
        self.relation_resolver = TypeResolver(self.cat, Kind.RELATION)

    def resolver(self, kind: Kind) -> TypeResolver:
        """Return the resolver for ``kind``."""
        return self.entity_resolver if kind is Kind.ENTITY else self.relation_resolver


def _resolve_type(
    state: _State, kind: Kind, raw: str | None
) -> tuple[int | None, list[Reason], TypeMatch | None]:
    """Resolve one item's type from its label.

    Returns the type id (None for no home), the reasons found and the match details.
    """
    found = state.resolver(kind).resolve(raw or "")
    if found.match is not None:
        return found.match.type_id, [], found.match
    reason = Reason.AMBIGUOUS_TYPE if found.ambiguous else Reason.UNKNOWN_TYPE
    return None, [reason], None


def _place_entity(cat: Ontology, type_id: int | None, reasons: list[Reason]) -> int:
    """Return the type an entity is kept under.

    A type the ontology does not hold, or one it has deprecated, becomes Unknown.
    """
    if type_id is None:
        return UNKNOWN_ID
    entity = cat.entity(type_id)
    if entity is not None and entity.status is Status.DEPRECATED:
        reasons.append(Reason.DEPRECATED_TYPE)
        return UNKNOWN_ID
    return type_id


def _validate_entities(
    state: _State, items: tuple[EntityItem, ...]
) -> list[EntityValidation]:
    """Check every extracted entity against the ontology, in input order."""
    verdicts: list[EntityValidation] = []
    for item in items:
        type_id, reasons, match = _resolve_type(state, Kind.ENTITY, item.raw_type)
        if match is not None and match.how is How.FUZZY:
            reasons.append(Reason.FUZZY_MATCH)
        placed = _place_entity(state.cat, type_id, reasons)
        for reason in reasons:
            state.counters[reason] += 1
        state.types[item.local_id] = placed
        verdicts.append(
            EntityValidation(
                item.local_id,
                placed,
                tuple(reasons),
                match.how if match else None,
                match.score if match else None,
            )
        )
    return verdicts


def _place_relation(
    cat: Ontology, type_id: int | None, reasons: list[Reason]
) -> int | None:
    """Return the relation type a relation keeps, or None when it is dropped.

    A label the ontology does not hold, and a deprecated relation type, drop the
    relation.
    """
    if type_id is None:
        reasons.clear()
        reasons.append(Reason.UNDECLARED_RELATION)
        return None
    relation = cat.relation(type_id)
    if relation is not None and relation.status is Status.DEPRECATED:
        reasons.append(Reason.DEPRECATED_TYPE)
        return None
    return type_id


def _check_pattern(
    cat: Ontology,
    relation: Relation,
    source_type: int,
    target_type: int,
    reasons: list[Reason],
) -> bool:
    """Check a relation against the patterns of its type.

    Returns False when the relation must be dropped. A relation with patterns is
    dropped when an endpoint is Unknown or the endpoint types do not fit.
    """
    if not any(p.status is Status.ACTIVE for p in relation.patterns):
        return True
    if UNKNOWN_ID in (source_type, target_type):
        reasons.append(Reason.UNKNOWN_ENDPOINT)
        return False
    if cat.pattern_allows(relation, source_type, target_type):
        return True
    if relation.symmetric and cat.pattern_allows(relation, target_type, source_type):
        return True
    if cat.pattern_allows(relation, target_type, source_type):
        reasons.append(Reason.DIRECTION_SWAPPED)
        return False
    reasons.append(Reason.PATTERN_MISMATCH)
    return False


def _validate_relations(
    state: _State, items: tuple[RelationItem, ...]
) -> list[RelationValidation]:
    """Check every extracted relation against the ontology, in input order."""
    verdicts: list[RelationValidation] = []
    for item in items:
        type_id, reasons, match = _resolve_type(state, Kind.RELATION, item.raw_label)
        if match is not None and match.how is How.FUZZY:
            reasons.append(Reason.FUZZY_MATCH)
        placed = _place_relation(state.cat, type_id, reasons)
        kept = placed is not None
        relation = None if placed is None else state.cat.relation(placed)
        if relation is not None and not _check_pattern(
            state.cat,
            relation,
            state.types.get(item.source, UNKNOWN_ID),
            state.types.get(item.target, UNKNOWN_ID),
            reasons,
        ):
            kept = False
        for reason in reasons:
            state.counters[reason] += 1
        verdicts.append(
            RelationValidation(
                item.source,
                item.target,
                kept,
                placed if kept else None,
                tuple(reasons),
                match.how if match else None,
                match.score if match else None,
            )
        )
    return verdicts


def validate(cat: Ontology, items: ExtractedItems) -> ValidationReport:
    """Check one episode's extraction output against the ontology.

    Entities are resolved first, then relations. An entity is always kept. A relation
    is dropped when its label is not in the ontology, when the relation type is
    deprecated, or when its endpoint types break the relation's patterns. Fuzzy
    matches are kept and counted.

    Args:
        cat: The ontology the prompt was built from.
        items: The episode's output.

    Returns:
        Outcomes per item and counters by reason.
    """
    state = _State(cat)
    entity_verdicts = _validate_entities(state, items.entities)
    relation_verdicts = _validate_relations(state, items.relations)
    return ValidationReport(
        tuple(entity_verdicts), tuple(relation_verdicts), dict(state.counters)
    )
