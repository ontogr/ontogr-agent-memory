"""Entity verdicts: the remembered answers about pairs of entities."""

from enum import StrEnum

from pydantic import Field

from ontogr_mem.data_models.base import _Model


class VerdictKind(StrEnum):
    """Stored answer about a pair of entities."""

    SAME = "same"
    DISTINCT = "distinct"
    UNCERTAIN = "uncertain"


class VerdictSource(StrEnum):
    """Who produced a verdict."""

    RULE = "rule"
    LLM = "llm"
    USER = "user"


class VerifyStatus(StrEnum):
    """Per-pair outcome of one verification attempt."""

    OK = "ok"
    NO_VERDICT = "no_verdict"
    PARSE_FAILED = "parse_failed"
    ERROR = "error"
    VOTE_DISAGREED = "vote_disagreed"


class EntityVerdict(_Model):
    """A remembered answer about a pair of entities ordered by id.

    Attributes:
        verdict_id: ``"{a_id}|{b_id}"`` of the ordered pair.
        a_id: Smaller entity id.
        b_id: Larger entity id.
        verdict: The answer.
        source: Who produced the answer.
        inputs_hash: Hash of the entity data the answer was based on.
        passes: One vote per verification pass, such as ``["YES", "YES"]``.
        superseded_by: Merge that made the answer no longer live.
    """

    verdict_id: str
    a_id: str
    b_id: str
    verdict: VerdictKind
    source: VerdictSource
    inputs_hash: str
    passes: list[str] = Field(default_factory=list)
    superseded_by: str | None = None
