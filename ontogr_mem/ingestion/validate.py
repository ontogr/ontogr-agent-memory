"""Draft validation. The whole batch is checked before anything is written."""

from collections.abc import Sequence

from ontogr_mem.data_models import EpisodeDraft
from ontogr_mem.errors import AgentMemoryError


RESERVED_METADATA_KEYS = frozenset(
    {
        "id",
        "kind",
        "occurred_at",
        "ingested_at",
        "source",
        "actor",
        "content",
        "content_hash",
    }
)
"""Envelope field names. Metadata must not reuse them."""

_ID_SEPARATOR = "\x1f"


class DraftValidationError(AgentMemoryError, ValueError):
    """One or more drafts are invalid.

    Attributes:
        problems: Every problem found, so callers can report them together.
    """

    def __init__(self, problems: Sequence[str]) -> None:
        """Keep every problem and join them into the message."""
        self.problems = list(problems)
        super().__init__("; ".join(self.problems))


def draft_problems(draft: EpisodeDraft) -> list[str]:
    """Return every reason a draft is invalid, or an empty list when valid."""
    problems: list[str] = []
    if draft.occurred_at.tzinfo is None or draft.occurred_at.utcoffset() is None:
        problems.append("occurred_at must be timezone-aware")
    if not draft.source.system.strip():
        problems.append("source.system is empty")
    if not draft.source.ref.strip():
        problems.append("source.ref is empty")
    if _ID_SEPARATOR in draft.source.system + draft.source.ref:
        problems.append("source.system and source.ref must not contain \\x1f")
    if not draft.actor.id.strip():
        problems.append("actor.id is empty")
    if isinstance(draft.content, str):
        if not draft.content.strip():
            problems.append("content is empty")
    elif not draft.content:
        problems.append("content is empty")
    reserved = sorted(RESERVED_METADATA_KEYS & draft.metadata.keys())
    if reserved:
        problems.append(f"metadata uses reserved keys: {', '.join(reserved)}")
    return problems


def check_draft(draft: EpisodeDraft) -> None:
    """Raise `DraftValidationError` if the draft is invalid."""
    problems = draft_problems(draft)
    if problems:
        raise DraftValidationError(problems)


def validate_batch(drafts: Sequence[EpisodeDraft]) -> None:
    """Validate every draft and raise one error that lists all problems.

    Raises:
        DraftValidationError: If any draft is invalid. Each problem is prefixed
            with the draft's index in the batch.
    """
    problems = [
        f"draft[{i}]: {problem}"
        for i, draft in enumerate(drafts)
        for problem in draft_problems(draft)
    ]
    if problems:
        raise DraftValidationError(problems)
