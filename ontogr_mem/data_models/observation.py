"""Observations: the dated statements reconciliation works from.

An observation is the stored record of one extracted fact. It carries the
claim, the quote that supports it and the window the claim holds in.
"""

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum

from ontogr_mem.data_models.facts import Precision, TimeBasis
from ontogr_mem.data_models.timestamps import require_utc


class ObservationKind(StrEnum):
    """What an observation does to its slot.

    An ``ASSERTION`` opens or confirms a fact. A ``STATED_END`` closes one, and
    carries its instant in ``valid_from``.
    """

    ASSERTION = "assertion"
    STATED_END = "stated_end"
    CORRECTION = "correction"


@dataclass(frozen=True, slots=True)
class ProvenanceStamp:
    """Who produced the extraction.

    The stamp is copied onto observations but is not part of an observation id,
    so re-extracting the same text with another model does not create a new
    observation.

    Attributes:
        extractor: Name of the extractor that produced the fact.
        model: Model that read the text.
        prompt_version: Version of the prompt the model received.
    """

    extractor: str = ""
    model: str = ""
    prompt_version: str = ""


NO_STAMP = ProvenanceStamp()
"""Provenance stamp with no extractor, model or prompt version."""


@dataclass(frozen=True, slots=True)
class Observation:
    """One dated statement about a slot, taken from one episode.

    For a ``STATED_END`` the end instant is ``valid_from``, and ``valid_to`` is
    always ``None``. Fact intervals are derived from observations, never stored
    independently.

    Attributes:
        obs_id: Deterministic id; re-ingesting the same statement is idempotent.
        slot_key: Slot this statement belongs to.
        subject_id: Entity on the subject side of the relation.
        object_id: Entity on the object side of the relation.
        relation_type_id: Stable relation type id.
        kind: Whether the statement asserts or ends the relation.
        episode_id: Episode the statement came from.
        occurred_at: When the episode says the statement happened.
        ingested_at: When the episode entered the system.
        valid_from: Start of the stated window, or ``None`` when undated.
        valid_from_precision: Coarsest unit stated for ``valid_from``.
        valid_to: Stated end, or ``None`` for an open end.
        valid_to_precision: Coarsest unit stated for ``valid_to``.
        time_basis: Where ``valid_from`` came from.
        provenance: Extractor, model and prompt that produced the statement.
        statement: The claim in plain words.
        quote: The episode text that supports the statement.
        inverse: True when the slot is keyed on the object.
        source_system: Source system the statement came from, for trust rules.
        connector: Connector that brought the statement in, for trust rules.
        actor_role: Role of the person or system behind it, for trust rules.
    """

    obs_id: str
    slot_key: str
    subject_id: str
    object_id: str
    relation_type_id: int
    kind: ObservationKind
    episode_id: str
    occurred_at: datetime
    ingested_at: datetime
    valid_from: datetime | None
    valid_from_precision: Precision | None
    valid_to: datetime | None = None
    valid_to_precision: Precision | None = None
    time_basis: TimeBasis = TimeBasis.STATED
    provenance: ProvenanceStamp = NO_STAMP
    statement: str = ""
    quote: str = ""
    inverse: bool = False
    source_system: str = ""
    connector: str = ""
    actor_role: str = ""

    def __post_init__(self) -> None:
        """Normalize times to UTC and reject impossible intervals."""
        object.__setattr__(self, "occurred_at", require_utc(self.occurred_at))
        object.__setattr__(self, "ingested_at", require_utc(self.ingested_at))
        if self.valid_from is not None:
            object.__setattr__(self, "valid_from", require_utc(self.valid_from))
        if self.valid_to is not None:
            object.__setattr__(self, "valid_to", require_utc(self.valid_to))
        if (self.time_basis is TimeBasis.NONE) != (self.valid_from is None):
            msg = "valid_from must be None exactly when time_basis is NONE"
            raise ValueError(msg)
        if self.kind is ObservationKind.STATED_END and self.valid_from is None:
            msg = "a stated end needs an instant"
            raise ValueError(msg)
        if self.kind is ObservationKind.STATED_END and self.valid_to is not None:
            msg = "a stated end carries its instant in valid_from only"
            raise ValueError(msg)
        if (
            self.valid_from is not None
            and self.valid_to is not None
            and self.valid_to <= self.valid_from
        ):
            msg = "valid_to must be after valid_from"
            raise ValueError(msg)

    @property
    def dated(self) -> bool:
        """True when the observation takes part in time ordering."""
        return self.valid_from is not None

    @property
    def value_id(self) -> str:
        """The side of the fact that varies inside its slot."""
        return self.subject_id if self.inverse else self.object_id

    @property
    def holder_id(self) -> str:
        """The side of the fact that keys its slot."""
        return self.object_id if self.inverse else self.subject_id
