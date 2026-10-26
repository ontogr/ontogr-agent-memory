"""Per-run counters and failures for the temporal layer.

Every engine call returns a ``TemporalReport``. A slot that fails to commit
goes into the report instead of being raised, so the caller decides what to do.
Reports merge, so a batch of calls rolls up into one.
"""

from collections import Counter
from dataclasses import dataclass, field
from enum import StrEnum


class Outcome(StrEnum):
    """Stable counter keys.

    The values are part of the contract: tests, dashboards and reports refer to
    them by value, so renaming one is a breaking change.
    """

    SLOT_UNCHANGED = "slot_unchanged"
    SLOT_COMMITTED = "slot_committed"
    NEW_VERSION = "new_version"
    SUPERSESSION = "supersession"
    CONFLICT = "conflict"
    UNMATCHED_END = "unmatched_end"
    ORPHAN = "orphan"
    RETRY = "retry"
    COMMIT_FAILED = "commit_failed"
    OBSERVATION_SUBMITTED = "observation_submitted"
    EPISODE_COMMITTED = "episode_committed"
    EPISODE_SKIPPED = "episode_skipped"
    EPISODE_RECOVERED = "episode_recovered"
    EPISODE_NOT_RECOVERABLE = "episode_not_recoverable"
    EPISODE_RETRACTED = "episode_retracted"
    EPISODE_RESET = "episode_reset"
    SCRUBBED = "scrubbed"
    CANDIDATES_TRUNCATED = "candidates_truncated"
    JUDGE_ASKED = "judge_asked"
    JUDGE_CACHED = "judge_cached"
    JUDGE_CONTRADICTS = "judge_contradicts"
    JUDGE_UNCERTAIN = "judge_uncertain"
    JUDGE_NEGATIVE = "judge_negative"
    JUDGE_FAILED = "judge_failed"
    JUDGE_INVALID_ID = "judge_invalid_id"
    JUDGE_LLM_ERROR = "judge_llm_error"
    JUDGE_PARSE_FAILED = "judge_parse_failed"
    TOKENS_IN = "tokens_in"
    TOKENS_OUT = "tokens_out"


@dataclass(slots=True)
class TemporalReport:
    """What one engine call did.

    Attributes:
        counts: How often each outcome happened.
        skipped: Facts the adapter did not turn into observations, by reason.
        failures: ``(slot_key, reason)`` for each slot that could not be
            committed.
    """

    counts: Counter[str] = field(default_factory=Counter)
    skipped: Counter[str] = field(default_factory=Counter)
    failures: list[tuple[str, str]] = field(default_factory=list)

    def bump(self, key: Outcome, count: int = 1) -> None:
        """Add ``count`` to a counter."""
        if count:
            self.counts[key.value] += count

    def merge(self, other: "TemporalReport") -> None:
        """Fold another report into this one."""
        self.counts.update(other.counts)
        self.skipped.update(other.skipped)
        self.failures.extend(other.failures)

    @property
    def ok(self) -> bool:
        """True when no slot failed to commit."""
        return not self.failures
