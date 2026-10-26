"""Outputs of fact extraction: drops, per-episode results and the report."""

from collections import Counter
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from ontogr_mem.data_models import RETRYABLE, Status, Usage
from ontogr_mem.data_models.facts import ExtractedFact


class FactDropReason(StrEnum):
    """Reason code for a discarded fact, or for a cleared part of one.

    Some entries mean the whole fact was dropped (`empty_text`,
    `unknown_entity_ref`, `ungrounded`, `unknown_relation`). Others mean a part
    of a kept fact was cleared and the fact stays.
    """

    EMPTY_TEXT = "empty_text"
    UNKNOWN_ENTITY_REF = "unknown_entity_ref"
    AMBIGUOUS_ENTITY_REF = "ambiguous_entity_ref"
    UNGROUNDED = "ungrounded"
    BAD_TYPE_ID = "bad_type_id"
    UNKNOWN_RELATION = "unknown_relation"
    WRONG_DIRECTION = "wrong_direction"
    SELF_RELATION = "self_relation"
    TYPES_UNAVAILABLE = "types_unavailable"
    TIME_UNPARSED = "time_unparsed"
    TIME_ORDER = "time_order"
    TIME_EXPRESSION_NOT_FOUND = "time_expression_not_found"
    EVENT_INTERVAL = "event_interval"
    DUPLICATE_MERGED = "duplicate_merged"
    MISSING_FIELD = "missing_field"


class SkipReason(StrEnum):
    """Stable code for a skipped episode; numbers and causes are in `detail`."""

    NO_CANDIDATES = "no_candidates"
    OVERSIZE = "oversize"
    UNMAPPED_RECORD = "unmapped_record"
    UNSUPPORTED_LANGUAGE = "unsupported_language"


@dataclass(frozen=True, slots=True)
class FactDropped:
    """An item, or part of one, that failed validation with a reason code."""

    item: dict[str, Any]
    reason: FactDropReason
    detail: str = ""


@dataclass(slots=True)
class FactResult:
    """Everything extracted from one episode.

    Attributes:
        episode_id: The episode.
        status: Outcome of the extraction.
        facts: Facts kept after binding.
        dropped: Every discarded or cleared item with its reason.
        usage: Language model usage.
        skip_reason: Set only when `status` is `SKIPPED`.
        detail: Cause for `SKIPPED`, `PARSE_FAILED` and `ERROR`.
    """

    episode_id: str
    status: Status
    facts: list[ExtractedFact] = field(default_factory=list)
    dropped: list[FactDropped] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)
    skip_reason: SkipReason | None = None
    detail: str = ""


def skipped(result: FactResult, reason: SkipReason, detail: str) -> FactResult:
    """Mark a result as skipped with its stable reason."""
    result.status = Status.SKIPPED
    result.skip_reason = reason
    result.detail = detail
    return result


def unusable(
    result: FactResult,
    failure: Exception | None,
    errors: list[str],
) -> FactResult:
    """Mark a result that produced no parsed facts as error or parse_failed."""
    if failure is not None:
        result.status = Status.ERROR
        result.detail = f"{type(failure).__name__}: {failure}"
    else:
        result.status = Status.PARSE_FAILED
        result.detail = "; ".join(errors)
    return result


@dataclass(slots=True)
class FactReport:
    """Results for a batch, with counts for deciding what to retry."""

    results: list[FactResult] = field(default_factory=list)

    def status_counts(self) -> dict[str, int]:
        """Return episodes per status, with every status present."""
        counts = Counter(r.status.value for r in self.results)
        return {s.value: counts.get(s.value, 0) for s in Status}

    def dropped_by_episode(self) -> dict[str, dict[str, int]]:
        """Return drop counts per reason code, per episode."""
        out: dict[str, Counter[str]] = {}
        for result in self.results:
            out.setdefault(result.episode_id, Counter()).update(
                d.reason.value for d in result.dropped
            )
        return {episode: dict(counts) for episode, counts in out.items()}

    def skipped_reasons(self) -> dict[str, int]:
        """Return skipped episodes per stable reason code."""
        return dict(
            Counter(
                r.skip_reason.value
                for r in self.results
                if r.status is Status.SKIPPED and r.skip_reason is not None
            )
        )

    def usage(self) -> Usage:
        """Return token usage summed over all episodes."""
        total = Usage()
        for result in self.results:
            total.merge(result.usage)
        return total

    @property
    def failed_episode_ids(self) -> list[str]:
        """Return the episodes that must be retried."""
        return [r.episode_id for r in self.results if r.status in RETRYABLE]

    @property
    def retryable(self) -> bool:
        """Return True if any episode is `PARSE_FAILED` or `ERROR`."""
        return bool(self.failed_episode_ids)

    @property
    def succeeded(self) -> bool:
        """Return True when nothing is retryable."""
        return not self.retryable
