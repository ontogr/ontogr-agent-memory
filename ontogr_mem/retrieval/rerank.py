"""Optional passage scoring and its blend with reciprocal rank fusion."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True, slots=True)
class Passage:
    """Text supplied to a reranker with a stable result id.

    Attributes:
        id: Fact-version key or episode id.
        text: Statement or bounded episode text to score.
    """

    id: str
    text: str


class Reranker(Protocol):
    """Score passages for a query without changing their identities."""

    async def score(
        self, query: str, passages: Sequence[Passage]
    ) -> Mapping[str, float]:
        """Return scores by passage id; higher scores mean greater relevance.

        Missing ids receive the minimum score when blended. Raise
        ``RerankerError`` for a recoverable scoring failure.
        """
        ...


def blend(
    fused: Sequence[tuple[str, float]],
    scores: Mapping[str, float],
    *,
    weight: float,
) -> list[tuple[str, float]]:
    """Blend fused scores with min-max scaled shortlist scores.

    Args:
        fused: Candidates ordered by fused score and then id.
        scores: Reranker scores for the shortlist, keyed by passage id.
            Missing shortlist scores and candidates beyond the shortlist use
            scaled zero. Equal scores also scale to zero.
        weight: Reranker share of each blended score, from zero to one.

    Returns:
        All candidates ordered by blended score, then id. Scores are not
        rescaled or cut, so fact penalties can be applied afterward.
    """
    available = [scores[key] for key, _ in fused if key in scores]
    minimum = min(available, default=0.0)
    span = max(available, default=minimum) - minimum
    blended = []
    for key, fused_score in fused:
        scaled = (scores.get(key, minimum) - minimum) / span if span else 0.0
        blended.append((key, (1.0 - weight) * fused_score + weight * scaled))
    return sorted(blended, key=lambda item: (-item[1], item[0]))
