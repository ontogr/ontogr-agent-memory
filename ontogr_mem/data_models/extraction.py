"""Types produced by extraction and shared by the components that read them."""

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


Span = tuple[int, int]
"""Start and end character offsets into an episode text."""


class Method(StrEnum):
    """How an extracted item was produced."""

    GLINER = "gliner"
    SPACY = "spacy"
    LLM = "llm"
    RECORD = "record"


class Status(StrEnum):
    """Outcome of extracting one episode.

    `PARSE_FAILED` and `ERROR` mean the episode should be extracted again. The
    other statuses are final.
    """

    OK = "ok"
    EMPTY = "empty"
    PARSE_FAILED = "parse_failed"
    SKIPPED = "skipped"
    ERROR = "error"


RETRYABLE = frozenset({Status.PARSE_FAILED, Status.ERROR})


class DropReason(StrEnum):
    """Why an item was dropped."""

    TYPE_OUT_OF_RANGE = "type_out_of_range"
    MISSING_FIELD = "missing_field"


@dataclass(frozen=True, slots=True)
class ExtractedEntity:
    """A name found in one episode.

    Attributes:
        local_id: Id that is unique inside the episode.
        name: The name as written in the episode.
        type_id: Dense id of the entity type in the catalog the extractor used,
            or -1 when no type fits.
        confidence: Value from 0 to 1.
        method: How the entity was found.
        episode_id: Episode the entity came from.
        type_label: A type name the model chose that is not in the catalog.
        description: Short text from the model.
        attributes: Scalar attributes with evidence in the episode.
        spans: Character offsets of the mentions. Empty for the actor and for
            entities mapped from records, which have no text to point into.
    """

    local_id: str
    name: str
    type_id: int
    confidence: float
    method: Method
    episode_id: str
    type_label: str | None = None
    description: str = ""
    attributes: dict[str, Any] = field(default_factory=dict)
    spans: tuple[Span, ...] = ()

    def __post_init__(self) -> None:
        """Reject a confidence outside 0 to 1."""
        if not 0.0 <= self.confidence <= 1.0:
            msg = f"confidence out of range: {self.confidence}"
            raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class Dropped:
    """An item that failed validation, with a reason."""

    item: dict[str, Any]
    reason: DropReason
    detail: str = ""


@dataclass(slots=True)
class TokenCount:
    """Tokens and calls for one stage and model size."""

    input_tokens: int = 0
    output_tokens: int = 0
    calls: int = 0


@dataclass(slots=True)
class Usage:
    """Language model usage keyed by `"<stage>/<model size>"`."""

    buckets: dict[str, TokenCount] = field(default_factory=dict)

    def add(self, stage: str, size: str, input_tokens: int, output_tokens: int) -> None:
        """Record one call."""
        bucket = self.buckets.setdefault(f"{stage}/{size}", TokenCount())
        bucket.input_tokens += input_tokens
        bucket.output_tokens += output_tokens
        bucket.calls += 1

    def merge(self, other: "Usage") -> None:
        """Add another usage into this one."""
        for key, count in other.buckets.items():
            bucket = self.buckets.setdefault(key, TokenCount())
            bucket.input_tokens += count.input_tokens
            bucket.output_tokens += count.output_tokens
            bucket.calls += count.calls


@dataclass(slots=True)
class ExtractionResult:
    """What was extracted from one episode.

    Attributes:
        episode_id: The episode.
        status: Outcome of the extraction.
        entities: Entities found.
        dropped: Items that failed validation.
        usage: Language model usage.
        detail: Cause for `SKIPPED`, `PARSE_FAILED` and `ERROR`.
    """

    episode_id: str
    status: Status
    entities: list[ExtractedEntity] = field(default_factory=list)
    dropped: list[Dropped] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)
    detail: str = ""
