"""Point-in-time queries over fact versions.

The default query is the current belief: valid now, believed now. ``now`` is an
explicit argument, so nothing reads the wall clock.

The two time axes are independent:

* ``valid_at`` asks what was true in the world at that date (valid time).
* ``known_at`` asks what the system believed at that date (system time).

Asking both gives the past as it was understood then. ``matches`` is the
reference predicate; a store adapter must return the same versions for the same
query.
"""

from dataclasses import dataclass
from datetime import datetime

from ontogr_mem.data_models.fact_versions import Fact, FactStatus
from ontogr_mem.data_models.facts import Precision
from ontogr_mem.data_models.timestamps import to_micros
from ontogr_mem.temporal.visibility import in_current_view


@dataclass(frozen=True, slots=True)
class TemporalQuery:
    """Filters over fact versions.

    Attributes:
        now: Present time for the default valid-time and system-time filters.
        valid_at: Return facts whose valid interval contains this instant.
        known_at: Return the versions believed at this instant.
        overlaps: Half-open window that a fact's valid interval must intersect.
        include_history: Drop the default time filters so closed and past
            versions are returned.
        subject_id: Subject entity to match, or ``None`` for any.
        object_id: Object entity to match, or ``None`` for any.
        relation_type_id: Relation type to match, or ``None`` for any.
    """

    now: datetime
    valid_at: datetime | None = None
    known_at: datetime | None = None
    overlaps: tuple[datetime, datetime] | None = None
    include_history: bool = False
    subject_id: str | None = None
    object_id: str | None = None
    relation_type_id: int | None = None

    def __post_init__(self) -> None:
        """Reject incompatible time filters."""
        if self.overlaps is not None and self.overlaps[1] <= self.overlaps[0]:
            msg = "overlaps must be (from, to) with from < to"
            raise ValueError(msg)
        if self.valid_at is not None and self.overlaps is not None:
            msg = "use valid_at or overlaps, not both"
            raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class FactView:
    """A fact version as a reader sees it.

    Attributes:
        fact_id: Stable id shared by every version of the fact.
        version: Version number.
        slot_key: Slot the fact belongs to.
        subject_id: Entity on the subject side of the relation.
        object_id: Entity on the object side of the relation.
        relation_type_id: Stable relation type id.
        valid_from: Start of the valid-time interval.
        valid_to: End of the valid-time interval, or ``None`` while open.
        valid_from_precision: Coarsest unit stated for the start.
        valid_to_precision: Coarsest unit stated for the end.
        recorded_at: When this version was written.
        expired_at: When a later version replaced this one, or ``None``.
        status: Lifecycle state of this version.
        evidence: Observation ids that support the version.
        statement: The claim in plain words, when the store fills it in.
        source: Source system, when the store fills it in.
        disputed: True while the fact is in an open dispute.
        counterpart_ids: Facts it is in dispute with.
        dispute_reason: Why the dispute is open, when it is.
    """

    fact_id: str
    version: int
    slot_key: str
    subject_id: str
    object_id: str
    relation_type_id: int
    valid_from: datetime
    valid_to: datetime | None
    valid_from_precision: Precision | None
    valid_to_precision: Precision | None
    recorded_at: datetime
    expired_at: datetime | None
    status: FactStatus
    evidence: tuple[str, ...]
    statement: str = ""
    source: str = ""
    disputed: bool = False
    counterpart_ids: tuple[str, ...] = ()
    dispute_reason: str = ""


def view_of(fact: Fact) -> FactView:
    """Project a stored version to a reader view."""
    return FactView(
        fact_id=fact.fact_id,
        version=fact.version,
        slot_key=fact.slot_key,
        subject_id=fact.subject_id,
        object_id=fact.object_id,
        relation_type_id=fact.relation_type_id,
        valid_from=fact.valid_from,
        valid_to=fact.valid_to,
        valid_from_precision=fact.valid_from_precision,
        valid_to_precision=fact.valid_to_precision,
        recorded_at=fact.recorded_at,
        expired_at=fact.expired_at,
        status=fact.status,
        evidence=fact.evidence,
    )


def matches(query: TemporalQuery, fact: Fact) -> bool:
    """Return whether a fact version satisfies the query."""
    if not query.include_history and not in_current_view(fact.status):
        return False
    for wanted, actual in (
        (query.subject_id, fact.subject_id),
        (query.object_id, fact.object_id),
        (query.relation_type_id, fact.relation_type_id),
    ):
        if wanted is not None and wanted != actual:
            return False

    valid_from = to_micros(fact.valid_from)
    valid_to = to_micros(fact.valid_to)
    if query.overlaps is not None:
        window_from, window_to = query.overlaps
        if not (
            valid_from < to_micros(window_to) and valid_to > to_micros(window_from)
        ):
            return False
    elif not query.include_history or query.valid_at is not None:
        instant = to_micros(query.valid_at or query.now)
        if not (valid_from <= instant < valid_to):
            return False

    if query.known_at is not None or not query.include_history:
        known = to_micros(query.known_at or query.now)
        if not (to_micros(fact.recorded_at) <= known < to_micros(fact.expired_at)):
            return False
    return True
