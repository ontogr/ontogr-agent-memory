"""The records a commit writes: fact versions, withdrawals, supersessions, ledger.

Nothing here reads or writes a store. Each function takes a derived run and the
stored version it continues, and returns the record the commit should carry.
"""

from collections.abc import Sequence
from dataclasses import replace
from datetime import datetime

from ontogr_mem.data_models.fact_versions import (
    Fact,
    FactStatus,
    Supersession,
    SupersessionKind,
)
from ontogr_mem.data_models.ids import stable_id
from ontogr_mem.data_models.ledger import LedgerEntry, LedgerOp
from ontogr_mem.data_models.observation import Observation
from ontogr_mem.data_models.timestamps import to_micros
from ontogr_mem.temporal.ledger import link, slot_chain
from ontogr_mem.temporal.reconcile import Run
from ontogr_mem.temporal.temporal_store import SlotState


def candidate_fact(
    fact_id: str,
    version: int,
    run: Run,
    sample: Observation,
    recorded_at: datetime,
    *,
    status: FactStatus | None = None,
) -> Fact:
    """Return the fact version a run asks for."""
    holder = sample.holder_id
    inverse = sample.inverse
    return Fact(
        fact_id=fact_id,
        version=version,
        slot_key=sample.slot_key,
        subject_id=run.value_id if inverse else holder,
        object_id=holder if inverse else run.value_id,
        relation_type_id=sample.relation_type_id,
        valid_from=run.valid_from,
        valid_to=run.valid_to,
        valid_from_precision=run.valid_from_precision,
        valid_to_precision=run.valid_to_precision,
        recorded_at=recorded_at,
        expired_at=None,
        status=status or open_status(run),
        evidence=run.obs_ids,
        last_confirmed_at=run.last_confirmed_at,
        closed_by=run.closed_by,
        close_kind=run.close_kind,
        inverse=inverse,
    )


def withdrawal(fact: Fact, version: int, recorded_at: datetime) -> Fact:
    """Return the tombstone that replaces a fact with no evidence left."""
    return replace(
        fact,
        version=version,
        recorded_at=recorded_at,
        expired_at=None,
        status=FactStatus.WITHDRAWN,
        evidence=(),
        closed_by=None,
        close_kind=None,
    )


def open_status(run: Run) -> FactStatus:
    """Return the status of a run that the walk derived."""
    return FactStatus.CLOSED if run.valid_to is not None else FactStatus.ACTIVE


def same_content(old: Fact, new: Fact) -> bool:
    """True when re-deriving would give the stored content again.

    System-time fields are left out: they differ on every run.
    """
    return (
        old.status is new.status
        and old.valid_from == new.valid_from
        and old.valid_to == new.valid_to
        and old.valid_from_precision == new.valid_from_precision
        and old.valid_to_precision == new.valid_to_precision
        and old.evidence == new.evidence
        and old.closed_by == new.closed_by
    )


def supersession_record(
    fact: Fact,
    run: Run,
    observations_by_id: dict[str, Observation],
    prior_valid_to: datetime | None,
    recorded_at: datetime,
) -> Supersession:
    """Return the record of what closed the version the fact replaced."""
    cause = observations_by_id.get(run.closed_by or "")
    kind = run.close_kind or SupersessionKind.REPLACEMENT
    return Supersession(
        supersession_id=stable_id("sup", fact.fact_id, str(fact.version)),
        slot_key=fact.slot_key,
        old_fact_id=fact.fact_id,
        kind=kind,
        source_obs_id=run.closed_by or "",
        source_episode_id=cause.episode_id if cause is not None else "",
        prior_valid_to=prior_valid_to,
        applied_valid_to=run.valid_to,
        created_at=recorded_at,
    )


def ledger_entries(
    state: SlotState,
    recorded_at: datetime,
    new_facts: Sequence[Fact],
    supersessions: Sequence[Supersession],
) -> tuple[LedgerEntry, ...]:
    """One entry per new version and supersession, in order."""
    entries: list[LedgerEntry] = []
    head = (state.ledger_seq, state.ledger_head)

    def add(op: LedgerOp, subject: str, payload: dict[str, object]) -> None:
        nonlocal head
        entry = link(
            slot_chain(state.slot_key),
            head,
            recorded_at,
            op,
            subject,
            "engine",
            payload,
        )
        entries.append(entry)
        head = (entry.seq, entry.checksum)

    for fact in sorted(new_facts, key=lambda f: (f.fact_id, f.version)):
        if fact.status is FactStatus.WITHDRAWN:
            op = LedgerOp.FACT_WITHDRAWN
        elif fact.version == 1:
            op = LedgerOp.FACT_ADDED
        else:
            op = LedgerOp.FACT_VERSIONED
        add(
            op,
            fact.fact_id,
            {
                "version": fact.version,
                "status": fact.status.value,
                "valid_from": to_micros(fact.valid_from),
                "valid_to": to_micros(fact.valid_to),
                "evidence": list(fact.evidence),
            },
        )
    for row in sorted(supersessions, key=lambda s: s.supersession_id):
        add(
            LedgerOp.FACT_SUPERSEDED,
            row.old_fact_id,
            {
                "supersession_id": row.supersession_id,
                "kind": row.kind.value,
                "source_obs_id": row.source_obs_id,
                "applied_valid_to": to_micros(row.applied_valid_to),
            },
        )
    return tuple(entries)
