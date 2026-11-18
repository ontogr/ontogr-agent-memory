"""Check extracted entity types against the ontology."""

from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass, field

from ontogr_mem.ontology.models import UNKNOWN_ID, Kind, Ontology, Reason, Status
from ontogr_mem.ontology.resolve import How, TypeMatch, TypeResolver


@dataclass(frozen=True, slots=True)
class EntityItem:
    """An extracted entity with the type label its source provided."""

    local_id: str
    raw_type: str | None = None


@dataclass(frozen=True, slots=True)
class ExtractedItems:
    """Entity items from one extraction."""

    entities: tuple[EntityItem, ...] = ()


@dataclass(frozen=True, slots=True)
class EntityValidation:
    """Outcome for one entity type check."""

    local_id: str
    type_id: int
    reasons: tuple[Reason, ...] = ()
    how: How | None = None
    score: float | None = None


@dataclass(frozen=True, slots=True)
class ValidationReport:
    """Entity outcomes and reason counters."""

    entities: tuple[EntityValidation, ...]
    counters: Mapping[Reason, int]


@dataclass(slots=True)
class _State:
    """State collected during one entity validation pass."""

    cat: Ontology
    counters: Counter[Reason] = field(default_factory=Counter)
    resolver: TypeResolver = field(init=False)

    def __post_init__(self) -> None:
        self.resolver = TypeResolver(self.cat, Kind.ENTITY)


def _resolve_type(
    state: _State, raw: str | None
) -> tuple[int | None, list[Reason], TypeMatch | None]:
    """Resolve an entity type label and report any lookup issue."""
    found = state.resolver.resolve(raw or "")
    if found.match is not None:
        return found.match.type_id, [], found.match
    reason = Reason.AMBIGUOUS_TYPE if found.ambiguous else Reason.UNKNOWN_TYPE
    return None, [reason], None


def validate(cat: Ontology, items: ExtractedItems) -> ValidationReport:
    """Resolve and retain every extracted entity.

    Unknown and deprecated labels become Unknown. Fuzzy matches are retained and
    counted.
    """
    state = _State(cat)
    verdicts: list[EntityValidation] = []
    for item in items.entities:
        type_id, reasons, match = _resolve_type(state, item.raw_type)
        if match is not None and match.how is How.FUZZY:
            reasons.append(Reason.FUZZY_MATCH)
        if type_id is None:
            placed = UNKNOWN_ID
        else:
            entity = cat.entity(type_id)
            if entity is not None and entity.status is Status.DEPRECATED:
                reasons.append(Reason.DEPRECATED_TYPE)
                placed = UNKNOWN_ID
            else:
                placed = type_id
        state.counters.update(reasons)
        verdicts.append(
            EntityValidation(
                item.local_id,
                placed,
                tuple(reasons),
                match.how if match else None,
                match.score if match else None,
            )
        )
    return ValidationReport(tuple(verdicts), dict(state.counters))
