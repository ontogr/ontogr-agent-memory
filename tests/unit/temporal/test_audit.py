"""Audit violations on hand-built temporal states, one kind per case."""

from dataclasses import replace
from datetime import UTC, datetime, timedelta

import pytest

from ontogr_mem.data_models.fact_versions import (
    Fact,
    FactStatus,
    Supersession,
    SupersessionKind,
)
from ontogr_mem.data_models.ledger import LedgerEntry, LedgerOp
from ontogr_mem.data_models.observation import Observation, ObservationKind
from ontogr_mem.graphstore.memory import (
    MemoryEpisodeStore,
    MemoryResolverStore,
    MemoryTemporalStore,
)
from ontogr_mem.temporal.audit import audit
from ontogr_mem.temporal.classify import Disposition
from ontogr_mem.temporal.ledger import GENESIS, link, slot_chain
from ontogr_mem.temporal.reconcile import Conflict, ConflictKind
from ontogr_mem.temporal.temporal_store import IncompleteEpisode, TemporalSnapshot
from ontogr_mem.temporal.verdicts import (
    ConflictDecision,
    DecisionAction,
    VerdictSource,
)
from tests.contracts.temporal_store import contradiction


SLOT = "slot-ana-employer"
AT = datetime(2026, 1, 1, tzinfo=UTC)
LATER = datetime(2026, 2, 1, tzinfo=UTC)


def fact(
    fact_id: str = "f1",
    version: int = 1,
    *,
    status: FactStatus = FactStatus.ACTIVE,
    evidence: tuple[str, ...] = ("obs-1",),
    recorded: datetime = AT,
    expired_at: datetime | None = None,
    valid_to: datetime | None = None,
    closed_by: str | None = None,
    close_kind: SupersessionKind | None = None,
    obj: str = "acme",
) -> Fact:
    """Build one fact version between Ana and `obj`."""
    return Fact(
        fact_id=fact_id,
        version=version,
        slot_key=SLOT,
        subject_id="ana",
        object_id=obj,
        relation_type_id=1,
        valid_from=AT,
        valid_to=valid_to,
        valid_from_precision=None,
        valid_to_precision=None,
        recorded_at=recorded,
        expired_at=expired_at,
        status=status,
        evidence=evidence,
        last_confirmed_at=AT,
        closed_by=closed_by,
        close_kind=close_kind,
    )


def observation(obs_id: str = "obs-1") -> Observation:
    """Build one live observation of the slot."""
    return Observation(
        obs_id=obs_id,
        slot_key=SLOT,
        subject_id="ana",
        object_id="acme",
        relation_type_id=1,
        kind=ObservationKind.ASSERTION,
        episode_id="ep-1",
        occurred_at=AT,
        ingested_at=AT,
        valid_from=AT,
        valid_from_precision=None,
    )


def snapshot(
    *,
    facts: tuple[Fact, ...] = (),
    observations: tuple[Observation, ...] = (observation(),),
    supersessions: tuple[Supersession, ...] = (),
    verdicts: tuple = (),
    decisions: tuple[ConflictDecision, ...] = (),
    conflicts: tuple[Conflict, ...] = (),
) -> TemporalSnapshot:
    """Build a snapshot of one slot with the given rows."""
    return TemporalSnapshot(
        facts=facts,
        observations=observations,
        supersessions=supersessions,
        verdicts=verdicts,
        decisions=decisions,
        conflicts=conflicts,
        slot_keys=(SLOT,),
    )


def conflict(
    *,
    fact_ids: tuple[str, ...],
    disposition: Disposition = Disposition.PREFER,
    reason: str = "newer statement wins",
    conflict_id: str = "c1",
) -> Conflict:
    """Build a pair conflict over the given facts."""
    return Conflict(
        conflict_id=conflict_id,
        slot_key=SLOT,
        kind=ConflictKind.PAIR,
        obs_ids=(),
        episode_ids=(),
        reason=reason,
        disposition=disposition,
        fact_ids=fact_ids,
    )


def decision(chosen: str) -> ConflictDecision:
    """Build a person's choice of one fact."""
    return ConflictDecision(
        decision_id="d1",
        conflict_id="c1",
        fact_ids=("f1", "f2"),
        action=DecisionAction.CHOOSE,
        decided_by=VerdictSource.USER,
        at=LATER,
        chosen_fact_id=chosen,
    )


class BrokenStore(MemoryTemporalStore):
    """A twin whose snapshot, chains and pending episodes are set by the test."""

    def __init__(
        self,
        snap: TemporalSnapshot,
        chains: dict[str, list[LedgerEntry]] | None = None,
        pending: list[IncompleteEpisode] | None = None,
    ) -> None:
        """Hold the hand-built state in place of the twin's own rows."""
        super().__init__(MemoryEpisodeStore(), MemoryResolverStore())
        self._rows = snap
        self._chain_rows = chains or {}
        self._pending_rows = pending or []

    async def snapshot(self) -> TemporalSnapshot:
        """Return the hand-built snapshot."""
        return self._rows

    async def ledger_entries(self, chain: str) -> list[LedgerEntry]:
        """Return the hand-built entries of a chain."""
        return list(self._chain_rows.get(chain, []))

    async def incomplete_episodes(self) -> list[IncompleteEpisode]:
        """Return the hand-built pending episodes."""
        return list(self._pending_rows)


def tampered_chain() -> dict[str, list[LedgerEntry]]:
    """A slot chain whose only entry has a checksum that does not match."""
    entry = link(
        slot_chain(SLOT),
        (0, GENESIS),
        AT,
        LedgerOp.FACT_ADDED,
        "f1",
        "engine",
        {"version": 1},
    )
    return {slot_chain(SLOT): [replace(entry, checksum="0" * 64)]}


CASES = [
    pytest.param(
        BrokenStore(snapshot(facts=(fact(), fact())), {}),
        "duplicate_version",
        id="duplicate_version",
    ),
    pytest.param(
        BrokenStore(snapshot(facts=(fact(version=1), fact(version=2)))),
        "two_current_versions",
        id="two_current_versions",
    ),
    pytest.param(
        BrokenStore(snapshot(facts=(fact(valid_to=AT),))),
        "empty_valid_interval",
        id="empty_valid_interval",
    ),
    pytest.param(
        BrokenStore(snapshot(facts=(fact(recorded=LATER, expired_at=AT),))),
        "empty_system_interval",
        id="empty_system_interval",
    ),
    pytest.param(
        BrokenStore(snapshot(facts=(fact(recorded=datetime(1970, 1, 1, tzinfo=UTC)),))),
        "implausible_system_time",
        id="implausible_system_time",
    ),
    pytest.param(
        BrokenStore(snapshot(facts=(fact(status=FactStatus.CLOSED),))),
        "closed_without_record",
        id="closed_without_record",
    ),
    pytest.param(
        BrokenStore(snapshot(facts=(fact(status=FactStatus.OVERRIDDEN, evidence=()),))),
        "excluded_without_evidence",
        id="excluded_without_evidence",
    ),
    pytest.param(
        BrokenStore(
            snapshot(facts=(fact(status=FactStatus.WITHDRAWN, evidence=("obs-1",)),))
        ),
        "withdrawn_with_evidence",
        id="withdrawn_with_evidence",
    ),
    pytest.param(
        BrokenStore(snapshot(facts=(fact(evidence=("obs-gone",)),))),
        "fact_without_evidence",
        id="fact_without_evidence",
    ),
    pytest.param(
        BrokenStore(
            snapshot(
                facts=(
                    fact(
                        status=FactStatus.CLOSED,
                        close_kind=SupersessionKind.REPLACEMENT,
                    ),
                ),
                supersessions=(
                    Supersession(
                        supersession_id="s1",
                        slot_key=SLOT,
                        old_fact_id="f1",
                        kind=SupersessionKind.REPLACEMENT,
                        source_obs_id="obs-gone",
                        source_episode_id="ep-1",
                        prior_valid_to=None,
                        applied_valid_to=LATER,
                        created_at=LATER,
                    ),
                ),
            )
        ),
        "applied_supersession_without_cause",
        id="applied_supersession_without_cause",
    ),
    pytest.param(
        BrokenStore(
            snapshot(
                facts=(
                    fact(status=FactStatus.CLOSED, close_kind=SupersessionKind.VERDICT),
                ),
                supersessions=(
                    Supersession(
                        supersession_id="s1",
                        slot_key=SLOT,
                        old_fact_id="f1",
                        kind=SupersessionKind.VERDICT,
                        source_obs_id="obs-1",
                        source_episode_id="ep-1",
                        prior_valid_to=None,
                        applied_valid_to=LATER,
                        created_at=LATER,
                    ),
                ),
            )
        ),
        "verdict_supersession_with_episode",
        id="verdict_supersession_with_episode",
    ),
    pytest.param(
        BrokenStore(snapshot(facts=(fact(),)), tampered_chain()),
        "ledger_chain",
        id="ledger_chain",
    ),
    pytest.param(
        BrokenStore(
            snapshot(),
            pending=[IncompleteEpisode("ep-1", AT, "hash-ep-1")],
        ),
        "episode_not_committed",
        id="episode_not_committed",
    ),
    pytest.param(
        BrokenStore(
            snapshot(
                facts=(fact(),),
                conflicts=(
                    conflict(fact_ids=("f1",), conflict_id="c1"),
                    conflict(fact_ids=("f1",), conflict_id="c1"),
                ),
            )
        ),
        "duplicate_conflict",
        id="duplicate_conflict",
    ),
    pytest.param(
        BrokenStore(
            snapshot(
                facts=(fact(),),
                verdicts=(
                    contradiction("f1", "f2", "h1"),
                    contradiction("f1", "f2", "h1"),
                ),
            )
        ),
        "duplicate_verdict",
        id="duplicate_verdict",
    ),
    pytest.param(
        BrokenStore(
            snapshot(
                facts=(
                    fact(evidence=("obs-1",)),
                    fact("f2", status=FactStatus.OVERRIDDEN),
                ),
                conflicts=(conflict(fact_ids=("f1", "f2"), reason=""),),
            )
        ),
        "resolution_without_reason",
        id="resolution_without_reason",
    ),
    pytest.param(
        BrokenStore(
            snapshot(
                facts=(fact(), fact("f2")),
                conflicts=(conflict(fact_ids=("f1", "f2")),),
            )
        ),
        "resolution_without_loser",
        id="resolution_without_loser",
    ),
    pytest.param(
        BrokenStore(
            snapshot(
                facts=(fact(status=FactStatus.OVERRIDDEN), fact("f2")),
                decisions=(decision("f1"),),
            )
        ),
        "chosen_fact_overridden",
        id="chosen_fact_overridden",
    ),
    pytest.param(
        BrokenStore(
            snapshot(
                facts=(fact(), fact("f2", status=FactStatus.OVERRIDDEN)),
                conflicts=(conflict(fact_ids=("f1", "ghost")),),
            )
        ),
        "conflict_on_missing_fact",
        id="conflict_on_missing_fact",
    ),
]


class TestTemporalAudit:
    """Each invariant the audit checks fails on the state that breaks it."""

    @pytest.mark.parametrize(("store", "code"), CASES)
    async def test_reports_the_violated_invariant(self, store, code):
        """The named check reports its violation on the broken state."""
        violations = await audit(store, now=LATER, episode_ttl=timedelta(days=30))

        assert code in [v.code for v in violations]

    async def test_reports_nothing_for_a_consistent_state(self):
        """A live fact with its evidence and no other rows has no violation."""
        store = BrokenStore(snapshot(facts=(fact(),)))

        assert await audit(store) == []

    async def test_pending_episode_inside_the_ttl_is_not_reported(self):
        """A pending episode younger than the TTL is still waiting for recover."""
        store = BrokenStore(
            snapshot(),
            pending=[IncompleteEpisode("ep-1", AT, "hash-ep-1")],
        )

        violations = await audit(store, now=AT, episode_ttl=timedelta(hours=1))

        assert violations == []

    async def test_a_resolved_conflict_with_a_loser_is_consistent(self):
        """A preference whose loser is overridden and has a reason is clean."""
        store = BrokenStore(
            snapshot(
                facts=(fact(), fact("f2", status=FactStatus.OVERRIDDEN)),
                conflicts=(conflict(fact_ids=("f1", "f2")),),
                decisions=(),
            )
        )

        assert await audit(store) == []

    async def test_a_choice_of_the_surviving_fact_is_consistent(self):
        """Choosing a fact that is not overridden does not violate the audit."""
        store = BrokenStore(
            snapshot(
                facts=(fact(), fact("f2", status=FactStatus.OVERRIDDEN)),
                decisions=(decision("f1"),),
            )
        )

        assert await audit(store) == []
