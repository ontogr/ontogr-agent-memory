"""Plan one slot commit: diff the derived timeline against the stored versions.

Nothing is overwritten. A fact whose interval changed gets a new version, and
the old version is stamped expired at the commit's system time. A live fact that
no run claims any more becomes a ``WITHDRAWN`` tombstone: a new version with no
evidence, so the history shows that the belief went away.

The records a commit carries are built in ``ontogr_mem.temporal.records``.
"""

from collections.abc import Sequence
from dataclasses import replace
from datetime import datetime

from ontogr_mem.data_models.fact_versions import Fact, FactStatus, Supersession
from ontogr_mem.data_models.ids import stable_id
from ontogr_mem.data_models.observation import Observation
from ontogr_mem.temporal.reconcile import Conflict, Derived, Run
from ontogr_mem.temporal.records import (
    candidate_fact,
    ledger_entries,
    same_content,
    supersession_record,
    withdrawal,
)
from ontogr_mem.temporal.temporal_store import SlotCommit, SlotState
from ontogr_mem.temporal.visibility import is_withdrawable


def plan_commit(
    state: SlotState,
    observations: Sequence[Observation],
    derived: Derived,
    recorded_at: datetime,
    conflicts: Sequence[Conflict] = (),
) -> SlotCommit:
    """Return the commit that brings the stored versions to the derived timeline.

    Args:
        state: The slot as the writer read it. Its revision is the base.
        observations: Every observation of the slot.
        derived: The timeline derived from `observations`.
        recorded_at: System time of the commit. It must not be earlier than
            any version the slot already holds. Equal times supersede the old
            version immediately, which is correct on a frozen or coarse clock.
        conflicts: The slot's pair conflicts after this commit. They replace the
            stored set when they differ from it.

    Returns:
        The commit. It is empty when the stored versions and conflicts match.
    """
    current = [fact for fact in state.facts if fact.expired_at is None]
    withdrawable = [fact for fact in state.facts if is_withdrawable(fact)]
    if not observations and not withdrawable and not conflicts and not state.conflicts:
        return SlotCommit(state.slot_key, state.revision)
    latest = _latest_versions(state.facts)
    current_by_id = {fact.fact_id: fact for fact in current}
    observations_by_id = {o.obs_id: o for o in observations}

    expired: list[Fact] = []
    new_facts: list[Fact] = []
    supersessions: list[Supersession] = []
    kept: set[str] = set()
    placed: list[tuple[Run, FactStatus | None]] = [(run, None) for run in derived.runs]
    placed.extend((run, status) for run, status in derived.shadow)
    if observations:
        sample = observations[0]
        for (run, status), (fact_id, old) in zip(
            placed,
            _assign(state.slot_key, [run for run, _ in placed], withdrawable),
            strict=True,
        ):
            kept.add(fact_id)
            prior = latest.get(fact_id)
            candidate = candidate_fact(
                fact_id,
                prior.version + 1 if prior else 1,
                run,
                sample,
                recorded_at,
                status=status,
            )
            if old is not None and same_content(old, candidate):
                continue
            # A tombstone that a run brings back is expired like any other version.
            stale = old if old is not None else current_by_id.get(fact_id)
            if stale is not None:
                expired.append(replace(stale, expired_at=recorded_at))
            new_facts.append(candidate)
            prior_valid_to = old.valid_to if old is not None else None
            if (
                run.close_kind is not None
                and run.closed_by is not None
                and run.valid_to is not None
                and prior_valid_to != run.valid_to
            ):
                supersessions.append(
                    supersession_record(
                        candidate,
                        run,
                        observations_by_id,
                        prior_valid_to,
                        recorded_at,
                    )
                )

    for fact in withdrawable:
        if fact.fact_id in kept:
            continue
        expired.append(replace(fact, expired_at=recorded_at))
        new_facts.append(
            withdrawal(fact, latest[fact.fact_id].version + 1, recorded_at)
        )

    ledger = ledger_entries(
        state,
        recorded_at,
        new_facts,
        supersessions,
    )
    conflict_set = tuple(sorted(conflicts, key=lambda c: c.conflict_id))
    conflicts_changed = conflict_set != tuple(
        sorted(state.conflicts, key=lambda c: c.conflict_id)
    )
    return SlotCommit(
        slot_key=state.slot_key,
        base_revision=state.revision,
        new_facts=tuple(new_facts),
        expired_facts=tuple(expired),
        supersessions=tuple(supersessions),
        ledger=ledger,
        conflicts=conflict_set,
        conflicts_changed=conflicts_changed,
    )


def assign_fact_ids(
    slot_key: str, runs: Sequence[Run], current: Sequence[Fact]
) -> list[str]:
    """Return the fact id of each run, the same ids a commit would write."""
    return [fact_id for fact_id, _ in _assign(slot_key, runs, current)]


def _latest_versions(facts: Sequence[Fact]) -> dict[str, Fact]:
    latest: dict[str, Fact] = {}
    for fact in facts:
        held = latest.get(fact.fact_id)
        if held is None or fact.version > held.version:
            latest[fact.fact_id] = fact
    return latest


def _assign(
    slot_key: str, runs: Sequence[Run], current: Sequence[Fact]
) -> list[tuple[str, Fact | None]]:
    """Pair each run with the stored version it continues, if any.

    Runs and stored facts are matched by shared evidence, largest overlap first,
    so a fact keeps its id when a late observation moves its start date. A run
    with no match gets an id derived from its first observation.
    """
    pairs: list[tuple[int, datetime, str, datetime, int, int]] = []
    for index, run in enumerate(runs):
        for position, fact in enumerate(current):
            if fact.value_id != run.value_id:
                continue
            overlap = len(set(run.obs_ids) & set(fact.evidence))
            if overlap:
                pairs.append(
                    (
                        -overlap,
                        fact.valid_from,
                        fact.fact_id,
                        run.valid_from,
                        index,
                        position,
                    )
                )
    pairs.sort()
    matched: dict[int, Fact] = {}
    taken: set[int] = set()
    for *_, index, position in pairs:
        if index in matched or position in taken:
            continue
        matched[index] = current[position]
        taken.add(position)

    out: list[tuple[str, Fact | None]] = []
    for index, run in enumerate(runs):
        if index in matched:
            out.append((matched[index].fact_id, matched[index]))
        else:
            fact_id = stable_id("fact", slot_key, run.value_id, run.first_obs_id)
            out.append((fact_id, None))
    return out
