"""Typed answers for entity-pair verification."""

from typing import Literal

from pydantic import BaseModel, Field
from pydantic_ai import Agent

from ontogr_mem.resolution.verifier_prompt import SYSTEM


class PairVote(BaseModel):
    """One vote, addressed by the pair number shown in the prompt."""

    id: int = Field(strict=True)
    vote: Literal["YES", "NO"]
    rule: str = ""


class VerifierAnswer(BaseModel):
    """The votes returned for one window of pairs."""

    pairs: list[PairVote]


verifier_agent: Agent[None, VerifierAnswer] = Agent(
    output_type=VerifierAnswer,
    instructions=SYSTEM,
)
