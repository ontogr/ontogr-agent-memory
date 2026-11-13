"""Typed answers for temporal pair judgments."""

from pydantic import BaseModel, Field, field_validator
from pydantic_ai import Agent

from ontogr_mem.temporal.prompt import SYSTEM_PROMPT
from ontogr_mem.temporal.verdicts import VerdictKind


class JudgeAnswer(BaseModel):
    """One pair judgment with optional supporting text.

    Attributes:
        id: Pair number from the prompt.
        verdict: Judgment for the pair.
        confidence: Confidence normalized to the interval from zero to one.
        kind: Kind of disagreement.
        rule: Short rule or reason label.
        reason: Plain explanation.
        quote_a: Supporting words from the first statement.
        quote_b: Supporting words from the second statement.
    """

    id: str
    verdict: VerdictKind
    confidence: float = Field(allow_inf_nan=False)
    kind: str = ""
    rule: str = ""
    reason: str = ""
    quote_a: str = ""
    quote_b: str = ""

    @field_validator("confidence", mode="before")
    @classmethod
    def numeric_confidence(cls, value: object) -> object:
        """Reject booleans and strings as confidence values."""
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("confidence must be a number")
        return value

    @field_validator("confidence")
    @classmethod
    def clamp_confidence(cls, value: float) -> float:
        """Keep numeric confidence in the supported interval."""
        return min(1.0, max(0.0, value))


class JudgeReply(BaseModel):
    """All judgments from one model request.

    Attributes:
        pairs: Typed judgments returned for the requested batch.
    """

    pairs: list[JudgeAnswer]


judge_agent: Agent[None, JudgeReply] = Agent(
    output_type=JudgeReply,
    instructions=SYSTEM_PROMPT,
)
