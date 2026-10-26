"""Apply the verdicts of the verifier to a collected decision.

The rules of ``ontogr_mem.resolution.decision`` collect the ids that need a
verdict. This module turns the verdicts of those ids into the final plan: one
same verdict links the group, several same verdicts leave it ambiguous, and
ids without a verdict stay pending.
"""

from collections.abc import Mapping

from ontogr_mem.data_models.entities import DecisionMethod, Outcome, VerdictKind
from ontogr_mem.resolution.decision import Collected
from ontogr_mem.resolution.plan import Plan


def apply_verdicts(
    collected: Collected,
    verified: Mapping[str, VerdictKind],
    incomplete: bool,
) -> Plan:
    """Turn the collected decision and its verdicts into a final plan."""
    if collected.plan is not None:
        return collected.plan
    ids = list(dict.fromkeys(collected.verify))
    if not ids:
        return Plan(
            None,
            DecisionMethod.NONE,
            Outcome.CREATED_PROVISIONAL if incomplete else Outcome.CREATED,
            "candidates_incomplete" if incomplete else "no_candidates",
        )
    same = [
        entity_id for entity_id in ids if verified.get(entity_id) is VerdictKind.SAME
    ]
    if len(same) == 1:
        return Plan(same[0], DecisionMethod.LLM, Outcome.MATCHED, "llm_same")
    if len(same) > 1:
        return Plan(
            None,
            DecisionMethod.NONE,
            Outcome.CREATED_PROVISIONAL,
            "llm_ambiguous",
            candidates=tuple(same),
        )
    pending = [
        entity_id
        for entity_id in ids
        if verified.get(entity_id) is not VerdictKind.DISTINCT
    ]
    if pending:
        return Plan(
            None,
            DecisionMethod.NONE,
            Outcome.CREATED_PROVISIONAL,
            collected.reason or "unverified",
            candidates=tuple(pending),
            verify=tuple(pending),
        )
    return Plan(None, DecisionMethod.NONE, Outcome.CREATED, "verified_distinct")
