"""The plan one group resolves to.

A plan is the decision the resolver reaches for one group: the entity the group
matches, or a new entity, plus the reason and the candidates that are still
open. The plan is what the writer turns into stored rows.
"""

from dataclasses import dataclass

from ontogr_mem.data_models.entities import DecisionMethod, Outcome


@dataclass(frozen=True, slots=True)
class Plan:
    """What to do with one group.

    Attributes:
        entity_id: Entity to match, or None to create one.
        method: How the mention is linked.
        outcome: Whether the entity was matched or created.
        reason: Stable reason code.
        score: Score that decided a fuzzy match.
        candidates: Candidate ids that are still pending.
        verify: Candidate ids the inline tier should verify.
    """

    entity_id: str | None
    method: DecisionMethod
    outcome: Outcome
    reason: str
    score: float | None = None
    candidates: tuple[str, ...] = ()
    verify: tuple[str, ...] = ()
