"""The status penalties applied to the fused facts."""

from collections.abc import Sequence
from dataclasses import replace

from ontogr_mem.data_models import FactStatus
from ontogr_mem.retrieval.models import FactResult
from ontogr_mem.retrieval.settings import RetrievalSettings
from ontogr_mem.temporal.query import FactView


def apply_penalties(
    facts: Sequence[FactResult],
    *,
    limit: int,
    settings: RetrievalSettings,
) -> list[FactResult]:
    """Apply the status penalties to the fused facts and cut the list.

    A fact in an open dispute is multiplied by the disputed penalty, and an
    overridden fact by the overridden penalty. A fact with both is multiplied
    by the product of the two. The scores are then divided by the best score
    again, so the top fact scores 1.0, and the list is sorted by score, then
    by the fact key.

    Args:
        facts: The fused facts.
        limit: The most facts the result keeps.
        settings: The penalty multipliers.

    Returns:
        The penalised facts, best first, cut at the limit.
    """
    penalised = [
        replace(fact, score=fact.score * _penalty(fact.view, settings))
        for fact in facts
    ]
    if not penalised:
        return []
    best = max(fact.score for fact in penalised)
    if best > 0.0:
        penalised = [replace(fact, score=fact.score / best) for fact in penalised]
    ordered = sorted(
        penalised,
        key=lambda fact: (-fact.score, fact.view.fact_id, fact.view.version),
    )
    return ordered[:limit]


def _penalty(view: FactView, settings: RetrievalSettings) -> float:
    """Return the multiplier a fact view earns from its status.

    Args:
        view: The fact version the reader returned.
        settings: The penalty multipliers.

    Returns:
        The multiplier to apply to the fused score of the fact.
    """
    penalty = 1.0
    if view.disputed:
        penalty *= settings.disputed_penalty
    if view.status is FactStatus.OVERRIDDEN:
        penalty *= settings.overridden_penalty
    return penalty
