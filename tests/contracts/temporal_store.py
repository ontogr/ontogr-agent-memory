"""Contract every `TemporalStore` backend must satisfy.

A backend test module subclasses `TemporalStoreContract` and provides a
`make_harness` fixture that returns a factory. Each call to the factory makes a
harness on a fresh tenant. The class has no `Test` prefix, so pytest collects
only its subclasses.
"""

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import replace
from datetime import UTC, datetime

import pytest

from ontogr_mem.data_models.episode import TemporalStatus
from ontogr_mem.data_models.fact_versions import (
    Fact,
    FactStatus,
    Supersession,
    SupersessionKind,
)
from ontogr_mem.data_models.facts import Precision, TimeBasis
from ontogr_mem.data_models.ledger import LedgerEntry, LedgerOp
from ontogr_mem.data_models.observation import Observation, ObservationKind
from ontogr_mem.temporal.classify import ConflictClass, Disposition, RelationToTime
from ontogr_mem.temporal.errors import (
    MissingEntityError,
    MissingEpisodeError,
    MissingFactVersionError,
    RevisionConflict,
)
from ontogr_mem.temporal.ledger import GENESIS, link, verify
from ontogr_mem.temporal.query import TemporalQuery
from ontogr_mem.temporal.reconcile import Conflict, ConflictKind
from ontogr_mem.temporal.temporal_store import (
    EpisodeState,
    SlotCommit,
    SourceSelector,
)
from ontogr_mem.temporal.verdicts import (
    ConflictDecision,
    DecisionAction,
    FactVerdict,
    VerdictKind,
    VerdictSource,
    pair_key,
)
from tests.support.temporal import TemporalHarness


SLOT = "slot-ana-employer"
OTHER_SLOT = "slot-ana-city"
RECORDED_1 = datetime(2026, 2, 1, tzinfo=UTC)
RECORDED_2 = datetime(2026, 3, 1, tzinfo=UTC)
KNOWN_BETWEEN = datetime(2026, 2, 15, tzinfo=UTC)
NOW = datetime(2026, 4, 1, tzinfo=UTC)
WRITERS = 30
TENANTS = 20
LONG_LIST = 60


def at(year: int) -> datetime:
    """Return the first instant of a year, in UTC."""
    return datetime(year, 1, 1, tzinfo=UTC)


START = at(2020)
REPLACEMENT = Supersession(
    supersession_id="sup-1",
    slot_key=SLOT,
    old_fact_id="fact-1",
    kind=SupersessionKind.REPLACEMENT,
    source_obs_id="obs-2",
    source_episode_id="ep-2",
    prior_valid_to=None,
    applied_valid_to=at(2022),
    created_at=RECORDED_2,
)


def observation(
    obs_id: str,
    *,
    episode_id: str = "ep-1",
    slot_key: str = SLOT,
    obj: str = "acme",
    statement: str = "",
    quote: str = "",
) -> Observation:
    """Build an assertion that Ana works at `obj` from 2020."""
    return Observation(
        obs_id=obs_id,
        slot_key=slot_key,
        subject_id="ana",
        object_id=obj,
        relation_type_id=1,
        kind=ObservationKind.ASSERTION,
        episode_id=episode_id,
        occurred_at=START,
        ingested_at=START,
        valid_from=START,
        valid_from_precision=Precision.YEAR,
        time_basis=TimeBasis.STATED,
        statement=statement,
        quote=quote,
    )


def version(
    fact_id: str,
    version_no: int,
    *,
    subject: str = "ana",
    obj: str = "acme",
    slot_key: str = SLOT,
    recorded: datetime = RECORDED_1,
    valid_from: datetime = START,
    evidence: tuple[str, ...] = ("obs-1",),
) -> Fact:
    """Build one open version of a fact between `subject` and `obj`."""
    return Fact(
        fact_id=fact_id,
        version=version_no,
        slot_key=slot_key,
        subject_id=subject,
        object_id=obj,
        relation_type_id=1,
        valid_from=valid_from,
        valid_to=None,
        valid_from_precision=Precision.YEAR,
        valid_to_precision=None,
        recorded_at=recorded,
        expired_at=None,
        status=FactStatus.ACTIVE,
        evidence=evidence,
        last_confirmed_at=START,
    )


def ledger_for(
    chain: str, head: tuple[int, str], writer: str = "engine"
) -> LedgerEntry:
    """Build the entry that follows `head` on `chain`."""
    return link(
        chain,
        head,
        RECORDED_1,
        LedgerOp.FACT_ADDED,
        "fact-1",
        writer,
        {"writer": writer},
    )


async def race(
    commit: Callable[[int], Awaitable[int]], writers: int
) -> tuple[list[int], list[Exception]]:
    """Run `writers` commits at once and split the results into wins and errors."""
    outcomes = await asyncio.gather(
        *(commit(index) for index in range(writers)),
        return_exceptions=True,
    )
    wins = [outcome for outcome in outcomes if isinstance(outcome, int)]
    losses = [outcome for outcome in outcomes if isinstance(outcome, Exception)]
    return wins, losses


class TemporalStoreContract:
    """Behavior every temporal store backend must show.

    Subclasses provide the `make_harness` fixture, which returns an async factory
    of `TemporalHarness`. Each harness is bound to a fresh tenant.
    """

    async def _seeded(
        self,
        make_harness: Callable[[], Awaitable[TemporalHarness]],
    ) -> TemporalHarness:
        harness = await make_harness()
        await harness.seed_entities(["ana", "acme", "globex"])
        await harness.seed_episode("ep-1")
        await harness.seed_episode("ep-2")
        return harness

    async def _commit_first_fact(self, harness: TemporalHarness) -> LedgerEntry:
        await harness.store.append_observations([observation("obs-1")])
        entry = ledger_for(SLOT, (0, GENESIS))
        await harness.store.commit_slot(
            SlotCommit(
                slot_key=SLOT,
                base_revision=0,
                new_facts=(version("fact-1", 1),),
                ledger=(entry,),
            )
        )
        return entry

    async def _change_object_to_globex(
        self, harness: TemporalHarness, first_entry: LedgerEntry
    ) -> None:
        first = (await harness.store.load_slot(SLOT)).facts[0]
        await harness.store.append_observations(
            [observation("obs-2", episode_id="ep-2", obj="globex")]
        )
        await harness.store.commit_slot(
            SlotCommit(
                slot_key=SLOT,
                base_revision=1,
                new_facts=(
                    version(
                        "fact-1",
                        2,
                        obj="globex",
                        recorded=RECORDED_2,
                        valid_from=at(2022),
                    ),
                ),
                expired_facts=(replace(first, expired_at=RECORDED_2),),
                supersessions=(REPLACEMENT,),
                ledger=(ledger_for(SLOT, (first_entry.seq, first_entry.checksum)),),
            )
        )

    async def test_commit_then_reload_shows_the_fact_and_ledger(self, make_harness):
        """A committed fact and its ledger entry are read back after a reload."""
        harness = await self._seeded(make_harness)
        await harness.store.append_observations([observation("obs-1")])

        before = await harness.store.load_slot(SLOT)
        revision = await harness.store.commit_slot(
            SlotCommit(
                slot_key=SLOT,
                base_revision=0,
                new_facts=(version("fact-1", 1),),
                ledger=(ledger_for(SLOT, (0, GENESIS)),),
            )
        )
        after = await harness.store.load_slot(SLOT)
        entries = await harness.store.ledger_entries(SLOT)

        assert before.revision == 0
        assert before.facts == ()
        assert revision == 1
        assert after.revision == 1
        assert [(fact.fact_id, fact.version) for fact in after.facts] == [("fact-1", 1)]
        assert after.ledger_seq == 1
        assert after.ledger_head == entries[0].checksum
        assert verify(entries) == []

    async def test_stale_base_raises_revision_conflict_and_writes_nothing(
        self, make_harness
    ):
        """A stale base is refused and leaves the slot as it was."""
        harness = await self._seeded(make_harness)
        await self._commit_first_fact(harness)
        raw_before = (
            await harness.raw_slot_properties(SLOT)
            if harness.raw_slot_properties is not None
            else None
        )

        with pytest.raises(RevisionConflict):
            await harness.store.commit_slot(
                SlotCommit(
                    slot_key=SLOT,
                    base_revision=0,
                    new_facts=(version("fact-2", 1, obj="globex"),),
                    supersessions=(REPLACEMENT,),
                    ledger=(ledger_for(SLOT, (0, GENESIS), writer="late"),),
                )
            )

        state = await harness.store.load_slot(SLOT)
        assert state.revision == 1
        assert [fact.fact_id for fact in state.facts] == ["fact-1"]
        entries = await harness.store.ledger_entries(SLOT)
        assert [entry.actor for entry in entries] == ["engine"]
        assert await harness.store.supersessions_of_slot(SLOT) == []
        if raw_before is not None and harness.raw_slot_properties is not None:
            assert await harness.raw_slot_properties(SLOT) == raw_before

    async def test_missing_entity_raises_and_writes_nothing(self, make_harness):
        """A fact that names an unknown entity is refused and writes nothing."""
        harness = await self._seeded(make_harness)
        await harness.store.append_observations([observation("obs-1")])

        with pytest.raises(MissingEntityError):
            await harness.store.commit_slot(
                SlotCommit(
                    slot_key=SLOT,
                    base_revision=0,
                    new_facts=(version("fact-1", 1, obj="nobody"),),
                    ledger=(ledger_for(SLOT, (0, GENESIS)),),
                )
            )

        state = await harness.store.load_slot(SLOT)
        assert state.revision == 0
        assert state.facts == ()
        assert await harness.store.ledger_entries(SLOT) == []

    async def test_refused_first_commit_does_not_create_a_slot(self, make_harness):
        """A missing endpoint must not leave an empty slot behind."""
        harness = await self._seeded(make_harness)

        with pytest.raises(MissingEntityError):
            await harness.store.commit_slot(
                SlotCommit(
                    slot_key=SLOT,
                    base_revision=0,
                    new_facts=(version("fact-1", 1, obj="nobody"),),
                )
            )

        assert SLOT not in await harness.store.slot_keys()

    async def test_expiring_an_unknown_version_is_refused_and_writes_nothing(
        self, make_harness
    ):
        """A commit that expires a missing version names it and writes nothing."""
        harness = await self._seeded(make_harness)
        first_entry = await self._commit_first_fact(harness)

        with pytest.raises(MissingFactVersionError, match="fact-404"):
            await harness.store.commit_slot(
                SlotCommit(
                    slot_key=SLOT,
                    base_revision=1,
                    new_facts=(version("fact-2", 1, obj="globex"),),
                    expired_facts=(
                        replace(version("fact-404", 1), expired_at=RECORDED_2),
                    ),
                    ledger=(ledger_for(SLOT, (first_entry.seq, first_entry.checksum)),),
                )
            )

        state = await harness.store.load_slot(SLOT)
        assert state.revision == 1
        assert [fact.fact_id for fact in state.facts] == ["fact-1"]

    async def test_a_missing_version_refuses_the_whole_commit(self, make_harness):
        """A commit that expires one missing version expires none of them."""
        harness = await self._seeded(make_harness)
        first_entry = await self._commit_first_fact(harness)
        first = (await harness.store.load_slot(SLOT)).facts[0]

        with pytest.raises(MissingFactVersionError, match="fact-404"):
            await harness.store.commit_slot(
                SlotCommit(
                    slot_key=SLOT,
                    base_revision=1,
                    new_facts=(version("fact-2", 1, obj="globex"),),
                    expired_facts=(
                        replace(first, expired_at=RECORDED_2),
                        replace(version("fact-404", 1), expired_at=RECORDED_2),
                    ),
                    ledger=(ledger_for(SLOT, (first_entry.seq, first_entry.checksum)),),
                )
            )

        state = await harness.store.load_slot(SLOT)
        assert state.revision == 1
        assert [(fact.fact_id, fact.expired_at) for fact in state.facts] == [
            ("fact-1", None)
        ]
        assert await harness.store.ledger_entries(SLOT) == [first_entry]

    async def test_concurrent_commits_with_one_base_have_one_winner(self, make_harness):
        """Thirty commits with one base produce exactly one winner."""
        harness = await self._seeded(make_harness)
        await harness.store.append_observations([observation("obs-1")])

        def commit(index: int) -> Awaitable[int]:
            return harness.store.commit_slot(
                SlotCommit(
                    slot_key=SLOT,
                    base_revision=0,
                    new_facts=(version("fact-1", 1),),
                    ledger=(ledger_for(SLOT, (0, GENESIS), writer=f"w{index}"),),
                )
            )

        wins, losses = await race(commit, WRITERS)

        assert wins == [1]
        assert len(losses) == WRITERS - 1
        assert all(isinstance(error, RevisionConflict) for error in losses)
        state = await harness.store.load_slot(SLOT)
        assert [(fact.fact_id, fact.version) for fact in state.facts] == [("fact-1", 1)]
        assert len(await harness.store.ledger_entries(SLOT)) == 1

    async def test_concurrent_commits_win_once_in_every_tenant(self, make_harness):
        """Concurrent races have one winner in each of many tenants."""
        harnesses = [await self._seeded(make_harness) for _ in range(TENANTS)]
        for harness in harnesses:
            await harness.store.append_observations([observation("obs-1")])

        def commit_in(harness: TemporalHarness) -> Callable[[int], Awaitable[int]]:
            def commit(index: int) -> Awaitable[int]:
                return harness.store.commit_slot(
                    SlotCommit(
                        slot_key=SLOT,
                        base_revision=0,
                        new_facts=(version("fact-1", 1),),
                        ledger=(ledger_for(SLOT, (0, GENESIS), writer=f"w{index}"),),
                    )
                )

            return commit

        results = await asyncio.gather(
            *(race(commit_in(harness), WRITERS) for harness in harnesses)
        )

        assert all(wins == [1] for wins, _ in results)
        for harness in harnesses:
            state = await harness.store.load_slot(SLOT)
            assert len(state.facts) == 1
            assert len(await harness.store.ledger_entries(SLOT)) == 1

    async def test_changed_fact_adds_a_version_and_expires_the_old_one(
        self, make_harness
    ):
        """A changed value adds a new version and expires the old one."""
        harness = await self._seeded(make_harness)
        first_entry = await self._commit_first_fact(harness)

        await self._change_object_to_globex(harness, first_entry)

        state = await harness.store.load_slot(SLOT)
        by_version = {fact.version: fact for fact in state.facts}
        assert set(by_version) == {1, 2}
        assert by_version[1].expired_at == RECORDED_2
        assert by_version[2].expired_at is None
        assert by_version[2].recorded_at > by_version[1].recorded_at
        assert await harness.store.supersessions_of_slot(SLOT) == [REPLACEMENT]

    async def test_unchanged_commit_writes_nothing(self, make_harness):
        """An empty commit leaves the slot unchanged."""
        harness = await self._seeded(make_harness)
        await self._commit_first_fact(harness)
        before = await harness.store.load_slot(SLOT)

        revision = await harness.store.commit_slot(
            SlotCommit(slot_key=SLOT, base_revision=before.revision)
        )

        after = await harness.store.load_slot(SLOT)
        assert revision == before.revision
        assert after == before

    async def test_query_as_of_and_known_at_return_the_history(self, make_harness):
        """Queries by valid time and by belief time return the stated history."""
        harness = await self._seeded(make_harness)
        first_entry = await self._commit_first_fact(harness)
        await self._change_object_to_globex(harness, first_entry)

        current = await harness.store.query(TemporalQuery(now=NOW, valid_at=at(2023)))
        known_between = await harness.store.query(
            TemporalQuery(now=NOW, known_at=KNOWN_BETWEEN, valid_at=at(2023))
        )
        before_change = await harness.store.query(
            TemporalQuery(now=NOW, valid_at=at(2021))
        )

        assert [(view.version, view.object_id) for view in current] == [(2, "globex")]
        assert [(view.version, view.object_id) for view in known_between] == [
            (1, "acme")
        ]
        assert before_change == []

    async def test_facts_of_an_episode_are_found_through_its_observations(
        self, make_harness
    ):
        """An episode's facts are the facts of the slots its observations touch."""
        harness = await self._seeded(make_harness)
        await self._commit_first_fact(harness)
        await harness.store.append_observations(
            [observation("obs-9", episode_id="ep-2", slot_key=OTHER_SLOT)]
        )
        await harness.store.commit_slot(
            SlotCommit(
                slot_key=OTHER_SLOT,
                base_revision=0,
                new_facts=(version("fact-9", 1, slot_key=OTHER_SLOT),),
                ledger=(ledger_for(OTHER_SLOT, (0, GENESIS)),),
            )
        )

        first_episode = await harness.store.facts_of_episode("ep-1")
        second_episode = await harness.store.facts_of_episode("ep-2")
        unknown = await harness.store.facts_of_episode("ep-missing")
        observations = await harness.store.observations_of_episode("ep-1")

        assert [fact.fact_id for fact in first_episode] == ["fact-1"]
        assert [fact.fact_id for fact in second_episode] == ["fact-9"]
        assert unknown == []
        assert [o.obs_id for o in observations] == ["obs-1"]

    async def test_evidence_on_a_fact_is_the_list_of_observation_ids(
        self, make_harness
    ):
        """A fact's evidence is the list of its observation ids."""
        harness = await self._seeded(make_harness)
        await harness.store.append_observations(
            [observation("obs-1"), observation("obs-2", episode_id="ep-2")]
        )
        await harness.store.commit_slot(
            SlotCommit(
                slot_key=SLOT,
                base_revision=0,
                new_facts=(version("fact-1", 1, evidence=("obs-1", "obs-2")),),
                ledger=(ledger_for(SLOT, (0, GENESIS)),),
            )
        )

        views = await harness.store.query(TemporalQuery(now=NOW))

        assert [view.evidence for view in views] == [("obs-1", "obs-2")]

    async def test_episode_state_survives_a_later_ingestion_write(self, make_harness):
        """A later ingestion write keeps the episode's temporal state."""
        harness = await self._seeded(make_harness)
        assert await harness.store.episode_state("ep-1") is None

        await harness.store.set_episode_pending("ep-1")
        pending = await harness.store.episode_state("ep-1")
        await harness.store.mark_committed("ep-1", "hash-1")
        committed = await harness.store.episode_state("ep-1")
        await harness.rewrite_episode("ep-1")
        after_rewrite = await harness.store.episode_state("ep-1")

        assert pending == EpisodeState(TemporalStatus.PENDING, None)
        assert committed == EpisodeState(TemporalStatus.COMMITTED, "hash-1")
        assert after_rewrite == committed

    async def test_episode_state_of_a_missing_episode_is_refused(self, make_harness):
        """Temporal state cannot be set on an episode that does not exist."""
        harness = await self._seeded(make_harness)

        with pytest.raises(MissingEpisodeError):
            await harness.store.set_episode_pending("ep-missing")
        with pytest.raises(MissingEpisodeError):
            await harness.store.mark_committed("ep-missing", "hash-x")

    async def test_episode_has_observations_once_they_are_appended(self, make_harness):
        """An episode has observations only after they are appended."""
        harness = await self._seeded(make_harness)
        before = await harness.store.episode_has_observations("ep-1")

        inserted = await harness.store.append_observations([observation("obs-1")])
        again = await harness.store.append_observations([observation("obs-1")])

        assert before is False
        assert inserted == 1
        assert again == 0
        assert await harness.store.episode_has_observations("ep-1") is True

    async def test_tenants_do_not_see_each_others_slots(self, make_harness):
        """Two tenants never see each other's slots or facts."""
        first = await self._seeded(make_harness)
        second = await self._seeded(make_harness)
        first_entry = await self._commit_first_fact(first)
        await self._change_object_to_globex(first, first_entry)
        await first.store.append_ledger([ledger_for("tenant-chain", (0, GENESIS))], 0)
        await first.store.mark_committed("ep-1", "hash-1")

        other = await second.store.load_slot(SLOT)
        views = await second.store.query(TemporalQuery(now=NOW))

        assert other.revision == 0
        assert other.facts == ()
        assert views == []
        assert await second.store.slot_keys() == []
        assert await second.store.observations_of_episode("ep-1") == []
        assert await second.store.facts_of_episode("ep-1") == []
        assert await second.store.ledger_entries(SLOT) == []
        assert await second.store.supersessions_of_slot(SLOT) == []
        assert await second.store.episode_has_observations("ep-1") is False
        assert await second.store.ledger_head("tenant-chain") == (0, GENESIS)
        assert await second.store.ledger_entries("tenant-chain") == []
        assert await second.store.episode_state("ep-1") is None

    async def test_append_ledger_refuses_a_stale_head(self, make_harness):
        """A ledger append with a stale head is refused and writes nothing."""
        harness = await self._seeded(make_harness)
        chain = "tenant-chain"
        first = ledger_for(chain, (0, GENESIS))

        appended = await harness.store.append_ledger([first], 0)
        stale = await harness.store.append_ledger(
            [ledger_for(chain, (0, GENESIS), writer="late")], 0
        )

        assert appended is True
        assert stale is False
        assert await harness.store.ledger_head(chain) == (1, first.checksum)
        assert verify(await harness.store.ledger_entries(chain)) == []

    async def test_concurrent_ledger_appends_have_one_winner_and_stay_linked(
        self, make_harness
    ):
        """Thirty ledger appends on one head give one winner and a linked chain."""
        harness = await self._seeded(make_harness)
        chain = "tenant-chain"

        def append(index: int) -> Awaitable[bool]:
            return harness.store.append_ledger(
                [ledger_for(chain, (0, GENESIS), writer=f"w{index}")], 0
            )

        outcomes = await asyncio.gather(*(append(i) for i in range(WRITERS)))

        assert outcomes.count(True) == 1
        entries = await harness.store.ledger_entries(chain)
        assert [entry.seq for entry in entries] == [1]
        assert verify(entries) == []

    async def test_standalone_chain_starts_at_genesis(self, make_harness):
        """A chain with no entries has the genesis head."""
        harness = await self._seeded(make_harness)

        assert await harness.store.ledger_head("never-written") == (0, GENESIS)
        assert await harness.store.ledger_entries("never-written") == []

    async def test_slot_keys_lists_only_written_slots(self, make_harness):
        """Slot keys list the slots that have been written."""
        harness = await self._seeded(make_harness)
        await harness.store.append_observations(
            [observation("obs-1"), observation("obs-3", slot_key=OTHER_SLOT)]
        )

        assert await harness.store.slot_keys() == sorted([SLOT, OTHER_SLOT])

    async def test_mark_committed_records_the_hash_in_the_state(self, make_harness):
        """Marking an episode committed stores its content hash."""
        harness = await self._seeded(make_harness)
        await harness.store.mark_committed("ep-2", "hash-2")

        state = await harness.store.episode_state("ep-2")
        assert state == EpisodeState(TemporalStatus.COMMITTED, "hash-2")

    async def test_put_verdicts_upserts_and_finds_a_verdict_by_either_side(
        self, make_harness
    ):
        """A verdict is found from both facts, and writing it twice keeps one row."""
        harness = await self._seeded(make_harness)
        verdict = contradiction("fact-1", "fact-2", "hash-1")

        await harness.store.put_verdicts([verdict])
        await harness.store.put_verdicts([verdict])

        assert await harness.store.verdicts_for(["fact-2"]) == [verdict]
        assert await harness.store.verdicts_for(["fact-1"]) == [verdict]
        assert await harness.store.verdicts_for(["fact-9"]) == []

    async def test_a_verdict_for_changed_inputs_is_stored_beside_the_old_one(
        self, make_harness
    ):
        """A changed inputs hash is a new verdict; the old one stays for history."""
        harness = await self._seeded(make_harness)
        old = contradiction("fact-1", "fact-2", "hash-1")
        new = contradiction("fact-1", "fact-2", "hash-2")

        await harness.store.put_verdicts([old, new])

        stored = await harness.store.verdicts_for(["fact-1"])
        assert sorted(v.inputs_hash for v in stored) == ["hash-1", "hash-2"]
        assert [v.inputs_hash for v in stored if v.valid_for("hash-2", "judge-1")] == [
            "hash-2"
        ]

    async def test_a_decision_is_found_by_any_of_its_fact_ids(self, make_harness):
        """A decision that names two facts is found from either one."""
        harness = await self._seeded(make_harness)
        decision = choice("decision-1")

        await harness.store.put_decision(decision)
        await harness.store.put_decision(decision)

        assert await harness.store.decisions_for(["fact-2"]) == [decision]
        assert await harness.store.decisions_for(["fact-1"]) == [decision]
        assert await harness.store.decisions_for(["fact-9"]) == []

    async def test_conflicts_are_found_by_slot_or_fact_and_replaced_by_a_commit(
        self, make_harness
    ):
        """A commit that flags a change replaces the slot's conflicts."""
        harness = await self._seeded(make_harness)
        await harness.store.append_observations([observation("obs-1")])
        conflict = pair_conflict("fact-1", "fact-2")

        await harness.store.commit_slot(
            SlotCommit(
                slot_key=SLOT,
                base_revision=0,
                new_facts=(version("fact-1", 1),),
                ledger=(ledger_for(SLOT, (0, GENESIS)),),
                conflicts=(conflict,),
                conflicts_changed=True,
            )
        )

        assert await harness.store.conflicts_for([SLOT], []) == [conflict]
        assert await harness.store.conflicts_for([], ["fact-1"]) == [conflict]
        assert await harness.store.conflicts_for([OTHER_SLOT], ["fact-9"]) == []

        await harness.store.commit_slot(
            SlotCommit(
                slot_key=SLOT,
                base_revision=1,
                conflicts=(),
                conflicts_changed=True,
            )
        )

        assert await harness.store.conflicts_for([SLOT], []) == []
        assert (await harness.store.load_slot(SLOT)).conflicts == ()

    async def test_a_commit_that_leaves_the_conflicts_alone_keeps_them(
        self, make_harness
    ):
        """A commit that does not flag a change keeps the stored conflicts."""
        harness = await self._seeded(make_harness)
        await harness.store.append_observations([observation("obs-1")])
        conflict = pair_conflict("fact-1", "fact-2")

        await harness.store.commit_slot(
            SlotCommit(
                slot_key=SLOT,
                base_revision=0,
                new_facts=(version("fact-1", 1),),
                ledger=(ledger_for(SLOT, (0, GENESIS)),),
                conflicts=(conflict,),
                conflicts_changed=True,
            )
        )

        await harness.store.commit_slot(
            SlotCommit(
                slot_key=SLOT,
                base_revision=1,
                new_facts=(version("fact-1", 2),),
            )
        )

        assert await harness.store.conflicts_for([SLOT], []) == [conflict]
        assert (await harness.store.load_slot(SLOT)).conflicts == (conflict,)

        await harness.store.commit_slot(
            SlotCommit(
                slot_key=SLOT,
                base_revision=2,
                new_facts=(version("fact-1", 3),),
                conflicts=(conflict,),
            )
        )

        assert await harness.store.conflicts_for([SLOT], []) == [conflict]
        assert (await harness.store.load_slot(SLOT)).conflicts == (conflict,)

    async def test_a_commit_keeps_the_verdicts_and_decisions_of_its_facts(
        self, make_harness
    ):
        """Verdicts and decisions are inputs, so a slot commit does not remove them."""
        harness = await self._seeded(make_harness)
        first_entry = await self._commit_first_fact(harness)
        await harness.store.put_verdicts([contradiction("fact-1", "fact-2", "h")])
        await harness.store.put_decision(choice("decision-1"))

        await self._change_object_to_globex(harness, first_entry)

        assert len(await harness.store.verdicts_for(["fact-1"])) == 1
        assert len(await harness.store.decisions_for(["fact-1"])) == 1

    async def test_pending_slots_are_derived_from_the_facts_and_the_verdicts(
        self, make_harness
    ):
        """A slot with an unjudged neighbour is pending until it is judged."""
        harness = await self._seeded(make_harness)
        other = await self._seeded(make_harness)
        await harness.store.append_observations(
            [observation("obs-1"), observation("obs-9", slot_key=OTHER_SLOT)]
        )
        await harness.store.commit_slot(
            SlotCommit(
                slot_key=SLOT,
                base_revision=0,
                new_facts=(version("fact-1", 1),),
            )
        )
        await harness.store.commit_slot(
            SlotCommit(
                slot_key=OTHER_SLOT,
                base_revision=0,
                new_facts=(version("fact-9", 1, slot_key=OTHER_SLOT),),
            )
        )

        assert await harness.store.pending_slots() == sorted([SLOT, OTHER_SLOT])
        assert await other.store.pending_slots() == []

        await harness.store.put_verdicts([contradiction("fact-1", "fact-9", "hash-1")])

        assert await harness.store.pending_slots() == []

    async def test_current_facts_and_facts_touching_return_only_open_versions(
        self, make_harness
    ):
        """Expired versions are left out of both current views."""
        harness = await self._seeded(make_harness)
        first_entry = await self._commit_first_fact(harness)
        await self._change_object_to_globex(harness, first_entry)

        current = await harness.store.current_facts()
        touching_acme = await harness.store.facts_touching(["acme"])
        touching_globex = await harness.store.facts_touching(["globex"])

        assert [(f.fact_id, f.version, f.object_id) for f in current] == [
            ("fact-1", 2, "globex")
        ]
        assert touching_acme == []
        assert [(f.fact_id, f.version) for f in touching_globex] == [("fact-1", 2)]

    async def test_withdrawn_tombstone_leaves_current_views_and_stays_in_history(
        self, make_harness
    ):
        """A withdrawn version is out of the current views and kept in history."""
        harness = await self._seeded(make_harness)
        first_entry = await self._commit_first_fact(harness)
        first = (await harness.store.load_slot(SLOT)).facts[0]
        tombstone = replace(
            first,
            version=2,
            recorded_at=RECORDED_2,
            expired_at=None,
            status=FactStatus.WITHDRAWN,
            evidence=(),
        )
        await harness.store.commit_slot(
            SlotCommit(
                slot_key=SLOT,
                base_revision=1,
                new_facts=(tombstone,),
                expired_facts=(replace(first, expired_at=RECORDED_2),),
                ledger=(ledger_for(SLOT, (first_entry.seq, first_entry.checksum)),),
            )
        )

        assert await harness.store.current_facts() == []
        assert await harness.store.facts_touching(["acme"]) == []
        history = await harness.store.query(
            TemporalQuery(now=NOW, include_history=True)
        )
        assert [(v.version, v.status) for v in history] == [
            (1, FactStatus.ACTIVE),
            (2, FactStatus.WITHDRAWN),
        ]

    async def test_remove_episode_observations_returns_its_slots(self, make_harness):
        """Removing an episode deletes its observations and reports their slots."""
        harness = await self._seeded(make_harness)
        await harness.store.append_observations(
            [observation("obs-1", episode_id="ep-1")]
        )

        removed = await harness.store.remove_episode_observations("ep-1")

        assert removed == {SLOT}
        assert await harness.store.observations_of_episode("ep-1") == []
        assert await harness.store.remove_episode_observations("ep-1") == set()

    async def test_scrub_blanks_the_text_of_verdicts_that_cite_the_episode(
        self, make_harness
    ):
        """Copied text of a verdict is blanked; the verdict itself stays."""
        harness = await self._seeded(make_harness)
        verdict = replace(
            contradiction("fact-1", "fact-2", "hash-1"),
            reason_text="copied text",
            quote_a="copied",
            source_episode_ids=("ep-1",),
        )
        await harness.store.put_verdicts([verdict])

        changed = await harness.store.scrub_episode("ep-1")

        (stored,) = await harness.store.verdicts_for(["fact-1"])
        assert changed == 1
        assert (stored.reason_text, stored.quote_a) == ("", "")

    async def test_retracted_state_keeps_the_hash_and_clearing_removes_it(
        self, make_harness
    ):
        """Retraction keeps the committed hash; clearing removes the state."""
        harness = await self._seeded(make_harness)
        await harness.store.mark_committed("ep-1", "hash-1")

        await harness.store.mark_retracted("ep-1")
        assert await harness.store.episode_state("ep-1") == EpisodeState(
            TemporalStatus.RETRACTED, "hash-1"
        )

        await harness.store.clear_episode_state("ep-1")
        assert await harness.store.episode_state("ep-1") is None

    async def test_incomplete_episodes_and_source_selection(self, make_harness):
        """Pending episodes are listed; sources are matched by system and ref."""
        harness = await self._seeded(make_harness)
        await harness.store.set_episode_pending("ep-1")

        pending = await harness.store.incomplete_episodes()

        assert [p.episode_id for p in pending] == ["ep-1"]
        assert await harness.store.episodes_matching(SourceSelector("chat")) == [
            "ep-1",
            "ep-2",
        ]
        assert await harness.store.episodes_matching(
            SourceSelector("chat", ref="ref-ep-2")
        ) == ["ep-2"]
        assert await harness.store.episodes_matching(SourceSelector("mail")) == []

    async def test_snapshot_lists_every_version_observation_and_slot(
        self, make_harness
    ):
        """The snapshot holds all fact versions, observations and slot keys."""
        harness = await self._seeded(make_harness)
        first_entry = await self._commit_first_fact(harness)
        await self._change_object_to_globex(harness, first_entry)

        snap = await harness.store.snapshot()

        assert [(f.version, f.status) for f in snap.facts] == [
            (1, FactStatus.ACTIVE),
            (2, FactStatus.ACTIVE),
        ]
        assert [o.obs_id for o in snap.observations] == ["obs-1", "obs-2"]
        assert snap.slot_keys == (SLOT,)

    async def test_facts_by_ids_return_every_version_of_a_fact(self, make_harness):
        """A fact id returns all of its versions, whatever their status."""
        harness = await self._seeded(make_harness)
        await harness.store.append_observations([observation("obs-1")])
        active = version("fact-1", 1)
        closed = replace(active, version=2, status=FactStatus.CLOSED, valid_to=at(2021))
        withdrawn = replace(active, version=3, status=FactStatus.WITHDRAWN, evidence=())
        corrected = replace(active, version=4, status=FactStatus.CORRECTED)
        await harness.store.commit_slot(
            SlotCommit(
                slot_key=SLOT,
                base_revision=0,
                new_facts=(
                    active,
                    closed,
                    withdrawn,
                    corrected,
                    version("fact-2", 1, obj="globex"),
                ),
                ledger=(ledger_for(SLOT, (0, GENESIS)),),
            )
        )

        facts = await harness.store.facts_by_ids(["fact-2", "fact-1"])

        assert [(f.fact_id, f.version, f.status) for f in facts] == [
            ("fact-1", 1, FactStatus.ACTIVE),
            ("fact-1", 2, FactStatus.CLOSED),
            ("fact-1", 3, FactStatus.WITHDRAWN),
            ("fact-1", 4, FactStatus.CORRECTED),
            ("fact-2", 1, FactStatus.ACTIVE),
        ]

    async def test_facts_by_ids_leave_out_unknown_and_repeated_ids(self, make_harness):
        """Unknown ids are left out, an empty list returns nothing, a repeat is one."""
        harness = await self._seeded(make_harness)
        await self._commit_first_fact(harness)

        assert await harness.store.facts_by_ids([]) == []
        assert await harness.store.facts_by_ids(["fact-404"]) == []
        assert [
            (f.fact_id, f.version)
            for f in await harness.store.facts_by_ids(["fact-1", "fact-1"])
        ] == [("fact-1", 1)]

    async def test_facts_of_entities_return_facts_with_the_entity_on_either_side(
        self, make_harness
    ):
        """An entity id returns the facts that name it on either side."""
        harness = await self._seeded(make_harness)
        await harness.store.append_observations([observation("obs-1")])
        subject_side = version("fact-1", 1)
        object_side = version("fact-2", 1, subject="acme", obj="ana")
        closed = replace(
            subject_side, version=2, status=FactStatus.CLOSED, valid_to=at(2021)
        )
        withdrawn = replace(
            subject_side, version=3, status=FactStatus.WITHDRAWN, evidence=()
        )
        await harness.store.commit_slot(
            SlotCommit(
                slot_key=SLOT,
                base_revision=0,
                new_facts=(subject_side, object_side, closed, withdrawn),
                ledger=(ledger_for(SLOT, (0, GENESIS)),),
            )
        )

        facts = await harness.store.facts_of_entities(["ana"])

        assert [(f.fact_id, f.version, f.status) for f in facts] == [
            ("fact-1", 1, FactStatus.ACTIVE),
            ("fact-1", 2, FactStatus.CLOSED),
            ("fact-1", 3, FactStatus.WITHDRAWN),
            ("fact-2", 1, FactStatus.ACTIVE),
        ]

    async def test_facts_of_entities_return_the_right_rows_for_sixty_ids(
        self, make_harness
    ):
        """Sixty entity ids return their own rows and nothing else."""
        objects = [f"obj-{index:03d}" for index in range(LONG_LIST)]
        harness = await self._seeded(make_harness)
        await harness.seed_entities(objects)
        await harness.store.append_observations(
            [
                observation(f"obs-{index:03d}", obj=obj)
                for index, obj in enumerate(objects)
            ]
        )
        await harness.store.commit_slot(
            SlotCommit(
                slot_key=SLOT,
                base_revision=0,
                new_facts=(
                    *(
                        version(
                            f"fact-{index:03d}",
                            1,
                            obj=obj,
                            evidence=(f"obs-{index:03d}",),
                        )
                        for index, obj in enumerate(objects)
                    ),
                    version("fact-999", 1, obj="acme"),
                ),
                ledger=(ledger_for(SLOT, (0, GENESIS)),),
            )
        )

        facts = await harness.store.facts_of_entities(objects)

        assert [f.object_id for f in facts] == objects

    async def test_observations_by_ids_keep_their_text_episode_and_slot(
        self, make_harness
    ):
        """The observations come back with their text, episode and slot."""
        harness = await self._seeded(make_harness)
        await harness.store.append_observations(
            [
                observation(
                    "obs-1",
                    statement="Ana works at Acme.",
                    quote="Ana works at Acme since 2020.",
                ),
                observation(
                    "obs-2",
                    episode_id="ep-2",
                    slot_key=OTHER_SLOT,
                    statement="Ana works at Globex.",
                    quote="Ana now works at Globex.",
                ),
            ]
        )

        found = await harness.store.observations_by_ids(["obs-2", "obs-1"])

        assert [
            (o.obs_id, o.statement, o.quote, o.episode_id, o.slot_key) for o in found
        ] == [
            (
                "obs-1",
                "Ana works at Acme.",
                "Ana works at Acme since 2020.",
                "ep-1",
                SLOT,
            ),
            (
                "obs-2",
                "Ana works at Globex.",
                "Ana now works at Globex.",
                "ep-2",
                OTHER_SLOT,
            ),
        ]
        assert await harness.store.observations_by_ids(["obs-404"]) == []
        assert await harness.store.observations_by_ids([]) == []

    async def test_tenants_do_not_see_each_other_through_the_read_verbs(
        self, make_harness
    ):
        """Two tenants read only their own facts and observations."""
        first = await self._seeded(make_harness)
        second = await self._seeded(make_harness)
        await first.store.append_observations(
            [observation("obs-1", statement="Ana works at Acme.")]
        )
        await first.store.commit_slot(
            SlotCommit(
                slot_key=SLOT,
                base_revision=0,
                new_facts=(version("fact-1", 1),),
                ledger=(ledger_for(SLOT, (0, GENESIS)),),
            )
        )
        await second.store.append_observations(
            [observation("obs-1", obj="globex", statement="Ana works at Globex.")]
        )
        await second.store.commit_slot(
            SlotCommit(
                slot_key=SLOT,
                base_revision=0,
                new_facts=(version("fact-1", 1, obj="globex"),),
                ledger=(ledger_for(SLOT, (0, GENESIS)),),
            )
        )

        assert [f.object_id for f in await first.store.facts_by_ids(["fact-1"])] == [
            "acme"
        ]
        assert [f.object_id for f in await second.store.facts_by_ids(["fact-1"])] == [
            "globex"
        ]
        assert await first.store.facts_of_entities(["globex"]) == []
        assert await second.store.facts_of_entities(["acme"]) == []
        assert [
            f.object_id for f in await second.store.facts_of_entities(["globex"])
        ] == ["globex"]
        assert [
            o.statement for o in await first.store.observations_by_ids(["obs-1"])
        ] == ["Ana works at Acme."]
        assert [
            o.statement for o in await second.store.observations_by_ids(["obs-1"])
        ] == ["Ana works at Globex."]

    async def test_facts_touching_returns_only_the_open_versions(self, make_harness):
        """The old verb still leaves the expired and withdrawn versions out."""
        harness = await self._seeded(make_harness)
        await harness.store.append_observations([observation("obs-1")])
        await harness.store.commit_slot(
            SlotCommit(
                slot_key=SLOT,
                base_revision=0,
                new_facts=(version("fact-1", 1), version("fact-2", 1, obj="globex")),
                ledger=(ledger_for(SLOT, (0, GENESIS)),),
            )
        )
        first = (await harness.store.load_slot(SLOT)).facts[0]
        tombstone = replace(
            first,
            version=2,
            recorded_at=RECORDED_2,
            expired_at=None,
            status=FactStatus.WITHDRAWN,
            evidence=(),
        )
        await harness.store.commit_slot(
            SlotCommit(
                slot_key=SLOT,
                base_revision=1,
                new_facts=(tombstone,),
                expired_facts=(replace(first, expired_at=RECORDED_2),),
            )
        )

        touching = await harness.store.facts_touching(["ana"])

        assert [(f.fact_id, f.version) for f in touching] == [("fact-2", 1)]


def contradiction(fact_a: str, fact_b: str, inputs: str) -> FactVerdict:
    """A judge verdict that two facts contradict each other."""
    return FactVerdict(
        pair_key=pair_key(fact_a, fact_b),
        fact_a_id=fact_a,
        fact_b_id=fact_b,
        verdict=VerdictKind.CONTRADICTS,
        inputs_hash=inputs,
        prompt_version="judge-1",
        source=VerdictSource.LLM,
        decided_at=RECORDED_1,
    )


def choice(decision_id: str) -> ConflictDecision:
    """A person's choice between two facts."""
    return ConflictDecision(
        decision_id=decision_id,
        conflict_id="conflict-1",
        fact_ids=("fact-1", "fact-2"),
        action=DecisionAction.CHOOSE,
        decided_by=VerdictSource.USER,
        at=RECORDED_2,
        chosen_fact_id="fact-1",
        actor="reviewer",
    )


def pair_conflict(fact_a: str, fact_b: str) -> Conflict:
    """A dispute between two facts of the slot, as the handler reports it."""
    return Conflict(
        conflict_id="conflict-1",
        slot_key=SLOT,
        kind=ConflictKind.PAIR,
        obs_ids=("obs-1",),
        episode_ids=("ep-1",),
        reason="concurrent statements",
        cls=ConflictClass.DISPUTE,
        relation_to_time=RelationToTime.OVERLAPPING,
        disposition=Disposition.COEXIST,
        fact_ids=(fact_a, fact_b),
        pair_key=pair_key(fact_a, fact_b),
    )
