"""Invariant checks over everything the temporal store holds for a tenant.

``audit`` reads one snapshot and returns violations. An empty list means every
check holds. It does not repair anything and it does not raise. Each violation
has a short stable ``code``.

A fact whose entity was deleted is not a violation. Deleting an entity removes
its tombstone edges, and the ledger keeps only hashes. The ledger therefore
never names a fact the audit can check against the graph.
"""

from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from ontogr_mem.data_models.fact_versions import FactStatus, SupersessionKind
from ontogr_mem.temporal.classify import Disposition
from ontogr_mem.temporal.ledger import TENANT_CHAIN, slot_chain, verify
from ontogr_mem.temporal.settings import TemporalSettings
from ontogr_mem.temporal.temporal_store import TemporalSnapshot, TemporalStore
from ontogr_mem.temporal.verdicts import DecisionAction
from ontogr_mem.temporal.visibility import is_withdrawable


_EARLIEST_SYSTEM_TIME = datetime(2000, 1, 1, tzinfo=UTC)
"""System times earlier than this are a unit mix-up, not a clock reading."""


@dataclass(frozen=True, slots=True)
class Violation:
    """One broken invariant.

    Attributes:
        code: Stable name of the check that failed.
        subject: Fact id, slot key, episode id or chain the check ran on.
        detail: Extra text, such as the status or the problem found.
    """

    code: str
    subject: str
    detail: str = ""


def _conflict_violations(snap: TemporalSnapshot) -> list[Violation]:
    """Check stored conflicts, person decisions and verdict identity."""
    out: list[Violation] = []
    live = {f.fact_id: f for f in snap.facts if is_withdrawable(f)}
    out += [
        Violation("duplicate_conflict", cid, str(n))
        for cid, n in Counter(c.conflict_id for c in snap.conflicts).items()
        if n > 1
    ]
    out += [
        Violation("duplicate_verdict", vid, str(n))
        for vid, n in Counter(v.verdict_id for v in snap.verdicts).items()
        if n > 1
    ]
    for conflict in snap.conflicts:
        for fact_id in conflict.fact_ids:
            if fact_id not in live:
                out.append(
                    Violation("conflict_on_missing_fact", conflict.conflict_id, fact_id)
                )
        if conflict.disposition not in (Disposition.PREFER, Disposition.CORRECT):
            continue
        if not conflict.reason:
            out.append(Violation("resolution_without_reason", conflict.conflict_id))
        loser = (
            FactStatus.OVERRIDDEN
            if conflict.disposition is Disposition.PREFER
            else FactStatus.CORRECTED
        )
        if not any(
            fact_id in live and live[fact_id].status is loser
            for fact_id in conflict.fact_ids
        ):
            out.append(Violation("resolution_without_loser", conflict.conflict_id))
    for decision in snap.decisions:
        chosen = live.get(decision.chosen_fact_id)
        if (
            decision.action is DecisionAction.CHOOSE
            and chosen is not None
            and chosen.status is FactStatus.OVERRIDDEN
        ):
            out.append(
                Violation(
                    "chosen_fact_overridden", decision.decision_id, chosen.fact_id
                )
            )
    return out


def _fact_violations(snap: TemporalSnapshot, live_obs: set[str]) -> list[Violation]:
    """Check fact versions, their intervals, their evidence and their statuses."""
    out: list[Violation] = []
    current = [f for f in snap.facts if f.expired_at is None]
    keys = Counter((f.fact_id, f.version) for f in snap.facts)
    out += [
        Violation("duplicate_version", fact_id, str(version))
        for (fact_id, version), count in keys.items()
        if count > 1
    ]
    heads = Counter(f.fact_id for f in current)
    out += [
        Violation("two_current_versions", fact_id)
        for fact_id, count in heads.items()
        if count > 1
    ]
    for fact in snap.facts:
        if fact.valid_to is not None and fact.valid_to <= fact.valid_from:
            out.append(Violation("empty_valid_interval", fact.fact_id))
        # A version superseded in the instant it was recorded has an empty
        # system interval. Only an end before the start is broken data.
        if fact.expired_at is not None and fact.expired_at < fact.recorded_at:
            out.append(
                Violation("empty_system_interval", f"{fact.fact_id}:{fact.version}")
            )
        for label, when in (
            ("recorded_at", fact.recorded_at),
            ("expired_at", fact.expired_at),
        ):
            if when is not None and when < _EARLIEST_SYSTEM_TIME:
                out.append(
                    Violation(
                        "implausible_system_time",
                        f"{fact.fact_id}:{fact.version}",
                        label,
                    )
                )

    sup_for = {row.old_fact_id for row in snap.supersessions}
    for fact in current:
        if fact.status is FactStatus.CLOSED and not (
            fact.fact_id in sup_for or fact.close_kind is not None
        ):
            out.append(Violation("closed_without_record", fact.fact_id))
        if fact.status in (FactStatus.OVERRIDDEN, FactStatus.CORRECTED) and not (
            fact.evidence
        ):
            out.append(Violation("excluded_without_evidence", fact.fact_id))
        if fact.status is FactStatus.WITHDRAWN and fact.evidence:
            out.append(Violation("withdrawn_with_evidence", fact.fact_id))
        if (
            fact.status is not FactStatus.WITHDRAWN
            and fact.evidence
            and not set(fact.evidence) & live_obs
        ):
            out.append(Violation("fact_without_evidence", fact.fact_id))

    return out


async def audit(
    store: TemporalStore,
    *,
    now: datetime | None = None,
    episode_ttl: timedelta | None = None,
) -> list[Violation]:
    """Return every violated invariant of the tenant, empty when all hold.

    Pass ``now`` to also report episodes left ``PENDING`` for longer than
    ``episode_ttl``. The default TTL is the ``episode_ttl_s`` setting.
    """
    snap = await store.snapshot()
    out: list[Violation] = []
    live_obs = {o.obs_id for o in snap.observations}

    out += _fact_violations(snap, live_obs)
    out += _conflict_violations(snap)
    for row in snap.supersessions:
        if row.source_obs_id and row.source_obs_id not in live_obs:
            out.append(
                Violation("applied_supersession_without_cause", row.supersession_id)
            )
        if row.kind is SupersessionKind.VERDICT and row.source_episode_id:
            out.append(
                Violation("verdict_supersession_with_episode", row.supersession_id)
            )

    for slot_key in snap.slot_keys:
        out += [
            Violation("ledger_chain", slot_chain(slot_key), problem)
            for problem in verify(await store.ledger_entries(slot_chain(slot_key)))
        ]
    out += [
        Violation("ledger_chain", TENANT_CHAIN, problem)
        for problem in verify(await store.ledger_entries(TENANT_CHAIN))
    ]

    if now is not None:
        ttl = episode_ttl or timedelta(seconds=TemporalSettings().episode_ttl_s)
        out += [
            Violation("episode_not_committed", rec.episode_id, "pending")
            for rec in await store.incomplete_episodes()
            if now - rec.ingested_at > ttl
        ]
    return out
