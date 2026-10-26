"""Fact versions: what the system believes, and the audit row of every close.

A fact is stored as versions. A change never edits a version, it writes a new
one with a higher ``version`` and stamps ``expired_at`` on the old one.

World time (``valid_from`` and ``valid_to``) and system time (``recorded_at``
and ``expired_at``) are independent half-open intervals. ``None`` is an open end
in the domain, and stores map it to an integer sentinel.
"""

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum

from ontogr_mem.data_models.facts import Precision


class FactStatus(StrEnum):
    """Lifecycle of a fact version.

    ``CORRECTED`` is out of the current view and kept in history because a
    correction judged the fact wrong. ``OVERRIDDEN`` stays in the current view
    and retrieval flags it after a preference or a person chose another fact.
    ``WITHDRAWN`` is a tombstone: the fact lost all its evidence, for example
    when its episode was reset, and it has no observation ids left.
    """

    ACTIVE = "active"
    CLOSED = "closed"
    CORRECTED = "corrected"
    OVERRIDDEN = "overridden"
    WITHDRAWN = "withdrawn"


class SupersessionKind(StrEnum):
    """Why a fact was closed."""

    REPLACEMENT = "replacement"
    STATED_END = "stated_end"
    VALID_TO_STATED = "valid_to_stated"
    VERDICT = "verdict"


@dataclass(frozen=True, slots=True)
class Fact:
    """One version of a fact: an interval, its evidence and its belief time.

    A change never edits a version. It writes a new one with a higher
    ``version`` and stamps ``expired_at`` on the old one.

    Attributes:
        fact_id: Stable id shared by every version of the same fact.
        version: Version number, starting at 1.
        slot_key: Slot the fact belongs to.
        subject_id: Entity on the subject side of the relation.
        object_id: Entity on the object side of the relation.
        relation_type_id: Stable relation type id.
        valid_from: Start of the valid-time interval.
        valid_to: End of the valid-time interval, or ``None`` while open.
        valid_from_precision: Coarsest unit stated for ``valid_from``.
        valid_to_precision: Coarsest unit stated for ``valid_to``.
        recorded_at: When this version was written.
        expired_at: When a later version replaced this one, or ``None``.
        status: Lifecycle state of this version.
        evidence: Observation ids that support this version.
        last_confirmed_at: Newest observation time that restated the fact.
        closed_by: Observation id that closed the fact, or ``None``.
        close_kind: Why the fact was closed, or ``None``.
        inverse: True when the slot is keyed on the object.
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
    last_confirmed_at: datetime
    closed_by: str | None = None
    close_kind: SupersessionKind | None = None
    inverse: bool = False

    @property
    def current(self) -> bool:
        """True for the version the system presently believes."""
        return self.expired_at is None

    @property
    def value_id(self) -> str:
        """The side of the fact that varies inside its slot."""
        return self.subject_id if self.inverse else self.object_id

    def interval(self) -> tuple[str, datetime, datetime | None]:
        """Return ``(value, valid_from, valid_to)`` for timeline comparisons."""
        return (self.value_id, self.valid_from, self.valid_to)


@dataclass(frozen=True, slots=True)
class Supersession:
    """Audit row for one close: which observation closed which fact.

    Attributes:
        supersession_id: Deterministic id of the row.
        slot_key: Slot of the closed fact.
        old_fact_id: Fact that was closed.
        kind: What caused the close.
        source_obs_id: Observation that caused the close.
        source_episode_id: Episode of that observation.
        prior_valid_to: End of the fact before the close.
        applied_valid_to: End the close applied.
        created_at: When the row was written.
    """

    supersession_id: str
    slot_key: str
    old_fact_id: str
    kind: SupersessionKind
    source_obs_id: str
    source_episode_id: str
    prior_valid_to: datetime | None
    applied_valid_to: datetime | None
    created_at: datetime
