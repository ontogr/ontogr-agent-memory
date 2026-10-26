"""Fact types shared by fact extraction and temporal reconciliation."""

from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import Any

from ontogr_mem.data_models.extraction import Method, Span


class FactKind(StrEnum):
    """A `STATE` holds over an interval; an `EVENT` has a point in time."""

    STATE = "state"
    EVENT = "event"


class Polarity(StrEnum):
    """Whether the text asserts the fact or its negation."""

    AFFIRMED = "affirmed"
    NEGATED = "negated"


class Modality(StrEnum):
    """Whether the fact is claimed to hold, planned, conditional, or desired."""

    ACTUAL = "actual"
    PLANNED = "planned"
    CONDITIONAL = "conditional"
    DESIRED = "desired"


class Precision(StrEnum):
    """Coarsest unit actually stated, ordered coarse to fine."""

    YEAR = "year"
    MONTH = "month"
    DAY = "day"
    TIME = "time"

    @property
    def rank(self) -> int:
        """Return 0 for the coarsest unit."""
        return list(Precision).index(self)


class TimeBasis(StrEnum):
    """Where `valid_from` and `valid_to` came from."""

    STATED = "stated"
    REFERENCE = "reference"
    NONE = "none"


@dataclass(frozen=True, slots=True)
class Evidence:
    """A supporting quote; `span` is computed by code and is None for records."""

    quote: str
    span: Span | None


@dataclass(frozen=True, slots=True)
class ExtractedFact:
    """One self-contained claim, grounded in one episode.

    Attributes:
        fact_id: Deterministic id built from the episode, the normalized text and
            the ordinal of the text within the episode.
        text: One claim, one sentence, entity names instead of pronouns.
        entity_refs: Local ids from the candidate list of the episode.
        relation_type_id: Dense relation type id from the catalog the extractor
            used, set only for a typed link between exactly two refs.
        relation_label: Free-text relation name the model proposed, or None.
        fact_kind: State or event.
        polarity: Affirmed or negated.
        modality: Actual, planned, conditional or desired.
        asserted_by: Speaker for messages; None for documents, tool output and
            records.
        valid_from: Start of the time window, cut to `precision`.
        valid_to: End of the time window, or None.
        precision: Coarsest unit the text stated, or None.
        time_expression: Literal time phrase from the episode, or None.
        time_basis: Stated, reference or none.
        evidence_quote: The episode's own text at the matched span.
        evidence_span: Character offsets into the episode text, or None for
            records.
        confidence: Value from 0 to 1.
        method: How the fact was produced.
        episode_id: Episode the fact came from.
        extra_evidence: Other quotes of exact duplicates that were merged.
        fields: Structured source fields for record-derived facts.
    """

    fact_id: str
    text: str
    entity_refs: tuple[str, ...]
    relation_type_id: int | None
    relation_label: str | None
    fact_kind: FactKind
    polarity: Polarity
    modality: Modality
    asserted_by: str | None
    valid_from: datetime | None
    valid_to: datetime | None
    precision: Precision | None
    time_expression: str | None
    time_basis: TimeBasis
    evidence_quote: str
    evidence_span: Span | None
    confidence: float
    method: Method
    episode_id: str
    extra_evidence: tuple[Evidence, ...] = ()
    fields: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Reject a confidence outside 0 to 1."""
        if not 0.0 <= self.confidence <= 1.0:
            msg = f"confidence out of range: {self.confidence}"
            raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class TypedFact:
    """A fact with its relation mapped to a stable type id.

    Attributes:
        fact: The extracted fact. Its dense `relation_type_id` is only meaningful
            with the projection it came from.
        relation_type: Stable relation type id, or None for a fact without a
            usable relation.
        catalog_version: Version of the projection whose dense ids the fact used.
    """

    fact: ExtractedFact
    relation_type: int | None
    catalog_version: str
