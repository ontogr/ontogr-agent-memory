"""What a reader learns about conflicts: computed state and disputed facts.

A conflict row is derived output. Its lifecycle state is computed from the
stored decisions and from the facts that still exist, so it is never stored as a
state. A fact in an open conflict is returned flagged, with its counterparts and
the reason.
"""

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import replace
from enum import StrEnum

from ontogr_mem.data_models.observation import Observation
from ontogr_mem.temporal.classify import ConflictClass, Disposition
from ontogr_mem.temporal.query import FactView
from ontogr_mem.temporal.reconcile import Conflict
from ontogr_mem.temporal.verdicts import (
    ConflictDecision,
    DecisionAction,
    latest_decision,
)


class ConflictState(StrEnum):
    """Where a conflict stands."""

    OPEN = "open"
    AUTO_RESOLVED = "auto_resolved"
    RESOLVED = "resolved"
    DISMISSED = "dismissed"
    VOID = "void"


_SETTLING = frozenset({Disposition.PREFER, Disposition.CORRECT, Disposition.SUPERSEDE})


def conflict_state(
    conflict: Conflict,
    decisions: Sequence[ConflictDecision],
    live_fact_ids: Iterable[str],
    live_obs_ids: Iterable[str],
    facts: Sequence[str] | None = None,
) -> ConflictState:
    """Compute a conflict's state from decisions and from what still exists.

    Args:
        conflict: The conflict to classify.
        decisions: Person decisions that may apply to it.
        live_fact_ids: Fact ids that still exist.
        live_obs_ids: Observation ids that still exist.
        facts: Facts the conflict is about. Defaults to the conflict's own
            fact ids; a tie that names observations only has none.

    Returns:
        The state. The first matching rule wins: a conflict about something that
        no longer exists is void, a person's decision beats any rule, a settling
        disposition means a rule already resolved it, and otherwise it is open.
    """
    about = tuple(facts if facts is not None else conflict.fact_ids)
    live_facts, live_obs = set(live_fact_ids), set(live_obs_ids)
    if about and not set(about) <= live_facts:
        return ConflictState.VOID
    if conflict.obs_ids and not set(conflict.obs_ids) & live_obs:
        return ConflictState.VOID
    latest = latest_decision(decisions, about)
    if latest is not None:
        if latest.action in (
            DecisionAction.MARK_BOTH_VALID,
            DecisionAction.MARK_COMPATIBLE,
        ):
            return ConflictState.DISMISSED
        return ConflictState.RESOLVED
    if conflict.disposition in _SETTLING:
        return ConflictState.AUTO_RESOLVED
    return ConflictState.OPEN


def annotate_views(
    views: Sequence[FactView],
    observations: Mapping[str, Observation],
    conflicts: Sequence[Conflict],
    decisions: Sequence[ConflictDecision],
) -> tuple[FactView, ...]:
    """Fill statement, source and dispute information on each view.

    Only an open dispute or an open uncertainty flags a fact. Settled, dismissed
    and void conflicts leave the fact unflagged.
    """
    live_facts = {v.fact_id for v in views}
    out: list[FactView] = []
    for view in views:
        first = next(
            (observations[i] for i in view.evidence if i in observations), None
        )
        counterparts: set[str] = set()
        reasons: list[str] = []
        for conflict in conflicts:
            if conflict.slot_key != view.slot_key and view.fact_id not in (
                conflict.fact_ids
            ):
                continue
            mine = view.fact_id in conflict.fact_ids or bool(
                set(view.evidence) & set(conflict.obs_ids)
            )
            if not mine:
                continue
            about = conflict.fact_ids or tuple(
                sorted(
                    w.fact_id
                    for w in views
                    if w.slot_key == view.slot_key
                    and set(w.evidence) & set(conflict.obs_ids)
                )
            )
            state = conflict_state(
                conflict,
                decisions,
                live_facts | set(conflict.fact_ids),
                observations.keys() | set(view.evidence),
                about,
            )
            if state is not ConflictState.OPEN or conflict.cls not in (
                ConflictClass.DISPUTE,
                ConflictClass.UNCERTAIN,
            ):
                continue
            reasons.append(conflict.reason)
            counterparts |= set(about) - {view.fact_id}
        out.append(
            replace(
                view,
                statement=first.statement if first else "",
                source=first.source_system if first else "",
                disputed=bool(reasons),
                counterpart_ids=tuple(sorted(counterparts)),
                dispute_reason="; ".join(sorted(set(reasons))),
            )
        )
    return tuple(out)
