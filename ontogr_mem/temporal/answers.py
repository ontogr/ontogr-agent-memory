"""Answers that are stored so the pair is not asked again."""

from dataclasses import dataclass, replace

from ontogr_mem.temporal.verdicts import VerdictKind


@dataclass(frozen=True, slots=True)
class Answer:
    """One answer for one pair.

    Attributes:
        verdict: The answer the model gave.
        confidence: How sure the model was, from 0 to 1.
        kind: Kind of disagreement the model named.
        rule: Short rule or reason label.
        reason: Plain explanation.
        quote_a: Words of the first statement that support the answer.
        quote_b: Words of the second statement that support the answer.
    """

    verdict: VerdictKind
    confidence: float
    kind: str
    rule: str
    reason: str
    quote_a: str
    quote_b: str


def unswap(answer: Answer) -> Answer:
    """Put the quotes back in their own order after a swapped pass."""
    return replace(answer, quote_a=answer.quote_b, quote_b=answer.quote_a)
