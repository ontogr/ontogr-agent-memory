"""Engine behavior on a live graph tenant: ingest, settle, record and recover."""

import itertools
from collections.abc import Mapping
from dataclasses import dataclass, replace
from datetime import UTC, datetime

import pytest

from ontogr_mem.data_models.episode import TemporalStatus
from ontogr_mem.data_models.fact_versions import SupersessionKind
from ontogr_mem.data_models.facts import (
    ExtractedFact,
    FactKind,
    Method,
    Modality,
    Polarity,
    Precision,
    TimeBasis,
    TypedFact,
)
from ontogr_mem.data_models.observation import Observation
from ontogr_mem.graphstore import GraphHandle
from ontogr_mem.graphstore.protocol import GraphStore
from ontogr_mem.graphstore.temporal.store import GraphTemporalStore
from ontogr_mem.temporal.cardinality import (
    Cardinality,
    DictRelationPolicy,
    RelationRules,
)
from ontogr_mem.temporal.engine import TemporalEngine
from ontogr_mem.temporal.errors import EpisodeAlreadyIngestedError, RevisionConflict
from ontogr_mem.temporal.ledger import TENANT_CHAIN, verify
from ontogr_mem.temporal.query import TemporalQuery
from ontogr_mem.temporal.report import Outcome
from ontogr_mem.temporal.temporal_store import EpisodeState, SlotCommit, TemporalStore
from tests.support.clock import FixedClock
from tests.support.episodes import make_episode
from tests.support.graph_backend import GraphBackend
from tests.support.temporal import open_temporal
from tests.unit.temporal.helpers import LocalRefResolver, at, obs


ONE = DictRelationPolicy({1: RelationRules(Cardinality.ONE)})
SLOT = "slot"
NOW = datetime(2026, 10, 22, tzinfo=UTC)


class CrashingStore(GraphTemporalStore):
    """A store that fails when an episode is marked committed."""

    async def mark_committed(self, episode_id: str, content_hash: str) -> None:
        """Simulate a crash between the ledger entry and the status write."""
        raise RuntimeError("crash before the episode is marked committed")


class ConflictingStore(GraphTemporalStore):
    """A store that refuses some commits with a revision conflict.

    Args:
        store: The graph store that holds the tenant.
        handle: The opened tenant.
        tenant_keyed: Whether the graph keys nodes by tenant.
        refusals: How many commits to refuse. Every commit is refused when the
            count is negative.
    """

    def __init__(
        self,
        store: GraphStore,
        handle: GraphHandle,
        *,
        tenant_keyed: bool,
        refusals: int,
    ) -> None:
        """Create a store that refuses its first commits."""
        super().__init__(store, handle, tenant_keyed=tenant_keyed)
        self.refusals = refusals
        self.attempts = 0

    async def commit_slot(self, commit: SlotCommit) -> int:
        """Refuse the commit while the refusal budget lasts, then write it."""
        self.attempts += 1
        if self.refusals < 0 or self.attempts <= self.refusals:
            raise RevisionConflict(commit.slot_key, commit.base_revision)
        return await super().commit_slot(commit)


@dataclass(frozen=True)
class Call:
    """One recorded call of a commit observer.

    Attributes:
        name: The method that was called.
        commit: The commit the observer received.
        revision: The revision of the slot when the call was made.
        observations: The slot's observations keyed by observation id. It is
            empty for the call made before the commit.
    """

    name: str
    commit: SlotCommit
    revision: int
    observations: Mapping[str, Observation]


class Recorder:
    """A commit observer that records the calls it receives.

    Args:
        store: The temporal store the engine writes to. The recorder reads the
            revision of the slot at each call, so a test can tell a call made
            before the commit from one made after it.
    """

    def __init__(self, store: TemporalStore) -> None:
        """Start with no recorded calls."""
        self._store = store
        self.calls: list[Call] = []

    async def before_slot_commit(self, commit: SlotCommit) -> None:
        """Record the commit that is about to be written."""
        self.calls.append(await self._call("before", commit, {}))

    async def after_slot_commit(
        self, commit: SlotCommit, observations: Mapping[str, Observation]
    ) -> None:
        """Record the commit that was written and the slot's observations."""
        self.calls.append(await self._call("after", commit, observations))

    async def _call(
        self,
        name: str,
        commit: SlotCommit,
        observations: Mapping[str, Observation],
    ) -> Call:
        """Build the record of one call with the slot's revision at that moment."""
        state = await self._store.load_slot(commit.slot_key)
        return Call(name, commit, state.revision, observations)


def make_engine(store: TemporalStore) -> TemporalEngine:
    """Build an engine on a store with the single-valued relation policy."""
    return TemporalEngine(store, ONE, LocalRefResolver(), FixedClock().now)


def extracted(**kw: object) -> ExtractedFact:
    """Build an extracted fact that says Ana lives in Paris from 2020."""
    base: dict[str, object] = {
        "fact_id": "f1",
        "text": "Ana lives in Paris",
        "entity_refs": ("ana", "paris"),
        "relation_type_id": 1,
        "relation_label": None,
        "fact_kind": FactKind.STATE,
        "polarity": Polarity.AFFIRMED,
        "modality": Modality.ACTUAL,
        "asserted_by": None,
        "valid_from": at(2020),
        "valid_to": None,
        "precision": Precision.YEAR,
        "time_expression": "in 2020",
        "time_basis": TimeBasis.STATED,
        "evidence_quote": "Ana lives in Paris",
        "evidence_span": None,
        "confidence": 1.0,
        "method": Method.LLM,
        "episode_id": "ep1",
    }
    base.update(kw)
    return ExtractedFact(**base)  # type: ignore[arg-type]


def typed(**kw: object) -> TypedFact:
    """Wrap an extracted fact whose relation is stable type 1."""
    return TypedFact(fact=extracted(**kw), relation_type=1, catalog_version="v1")


class TestTemporalEngine:
    """Ingest, reconcile and record episodes through the engine."""

    async def test_ingest_commits_the_episode_and_writes_its_fact(
        self, graph_backend: GraphBackend
    ):
        """A committed episode has its state, its ledger entry and its fact."""
        episodes, store = await open_temporal(
            graph_backend,
            (
                "ana",
                "paris",
            ),
        )
        await episodes.write_episodes([make_episode("ep1")])
        engine = make_engine(store)

        report = await engine.ingest([typed()], make_episode("ep1"))

        assert report.ok
        assert report.counts[Outcome.EPISODE_COMMITTED.value] == 1
        assert await store.episode_state("ep1") == EpisodeState(
            TemporalStatus.COMMITTED, "hash-ep1"
        )
        facts = await store.facts_of_episode("ep1")
        assert [(f.subject_id, f.object_id, f.version) for f in facts] == [
            ("ana", "paris", 1)
        ]
        assert (await store.ledger_head(TENANT_CHAIN))[0] == 1

    async def test_reingest_of_a_committed_episode_with_the_same_hash_is_skipped(
        self,
        graph_backend: GraphBackend,
    ):
        """The second ingest writes nothing and reports the skip."""
        episodes, store = await open_temporal(
            graph_backend,
            (
                "ana",
                "paris",
            ),
        )
        await episodes.write_episodes([make_episode("ep1")])
        engine = make_engine(store)
        await engine.ingest([typed()], make_episode("ep1"))
        head = await store.ledger_head(TENANT_CHAIN)

        report = await engine.ingest([typed()], make_episode("ep1"))

        assert report.counts[Outcome.EPISODE_SKIPPED.value] == 1
        assert await store.ledger_head(TENANT_CHAIN) == head

    async def test_ingest_refuses_an_episode_that_has_observations(
        self, graph_backend: GraphBackend
    ):
        """A changed episode is refused and the error names the episode."""
        episodes, store = await open_temporal(
            graph_backend,
            (
                "ana",
                "paris",
            ),
        )
        await episodes.write_episodes([make_episode("ep1")])
        engine = make_engine(store)
        await engine.ingest([typed()], make_episode("ep1"))
        changed = replace(make_episode("ep1"), content_hash="hash-changed")

        with pytest.raises(EpisodeAlreadyIngestedError, match="'ep1'"):
            await engine.ingest([typed()], changed)

    async def test_crash_after_observations_leaves_the_episode_pending(
        self, graph_backend: GraphBackend
    ):
        """Observations and the pending state survive a crash before commit."""
        episodes, store = await open_temporal(
            graph_backend, ("ana", "paris"), CrashingStore
        )
        await episodes.write_episodes([make_episode("ep1")])
        engine = make_engine(store)

        with pytest.raises(RuntimeError, match="crash"):
            await engine.ingest([typed()], make_episode("ep1"))

        state = await store.episode_state("ep1")
        assert state == EpisodeState(TemporalStatus.PENDING, None)
        assert await store.episode_has_observations("ep1")
        assert [f.subject_id for f in await store.facts_of_episode("ep1")] == ["ana"]

    async def test_commit_hook_runs_before_the_slot_is_written(
        self, graph_backend: GraphBackend
    ):
        """The hook sees the slot at its old revision, then the commit happens."""
        episodes, store = await open_temporal(
            graph_backend,
            (
                "ana",
                "paris",
            ),
        )
        await episodes.write_episodes([make_episode("ep1")])
        recorder = Recorder(store)
        engine = TemporalEngine(
            store,
            ONE,
            LocalRefResolver(),
            FixedClock().now,
            hooks=[recorder],
        )

        await engine.ingest([typed()], make_episode("ep1"))

        before = recorder.calls[0]
        assert before.name == "before"
        assert before.revision == 0
        assert len(before.commit.new_facts) == 1
        assert (await store.load_slot(before.commit.slot_key)).revision == 1

    async def test_the_after_hook_runs_once_with_the_slot_observations(
        self, graph_backend: GraphBackend
    ):
        """The after hook runs after the commit and gets the slot's observations."""
        _, store = await open_temporal(
            graph_backend,
            (
                "ana",
                "acme",
            ),
        )
        recorder = Recorder(store)
        engine = TemporalEngine(
            store,
            ONE,
            LocalRefResolver(),
            FixedClock().now,
            hooks=[recorder],
        )

        await engine.ingest_observations([obs("o1", "acme", at(2020))])

        assert [call.name for call in recorder.calls] == ["before", "after"]
        assert [call.revision for call in recorder.calls] == [0, 1]
        after = recorder.calls[1]
        assert after.commit is recorder.calls[0].commit
        assert set(after.observations) == {"o1"}
        assert after.observations["o1"].obs_id == "o1"

    async def test_an_empty_commit_calls_neither_hook(
        self, graph_backend: GraphBackend
    ):
        """A slot that changes nothing writes no commit, so no hook runs."""
        _, store = await open_temporal(
            graph_backend,
            (
                "ana",
                "acme",
            ),
        )
        await make_engine(store).ingest_observations([obs("o1", "acme", at(2020))])
        recorder = Recorder(store)
        engine = TemporalEngine(
            store,
            ONE,
            LocalRefResolver(),
            FixedClock().now,
            hooks=[recorder],
        )

        report = await engine.reconcile_slot(SLOT)

        assert report.counts[Outcome.SLOT_UNCHANGED.value] == 1
        assert recorder.calls == []

    async def test_a_lost_race_calls_the_after_hook_once(
        self, graph_backend: GraphBackend
    ):
        """The slot that retries calls the after hook once, for the commit that won."""
        episodes, store = await open_temporal(
            graph_backend, ("ana", "acme"), ConflictingStore, refusals=1
        )
        recorder = Recorder(store)
        engine = TemporalEngine(
            store,
            ONE,
            LocalRefResolver(),
            FixedClock().now,
            hooks=[recorder],
        )

        report = await engine.ingest_observations([obs("o1", "acme", at(2020))])

        assert report.counts[Outcome.RETRY.value] == 1
        assert report.counts[Outcome.SLOT_COMMITTED.value] == 1
        assert store.attempts == 2
        assert [call.name for call in recorder.calls] == ["before", "before", "after"]
        assert recorder.calls[-1].revision == 1

    async def test_a_slot_that_never_commits_calls_the_after_hook_never(
        self, graph_backend: GraphBackend
    ):
        """The slot that loses every race writes nothing, so the hook never runs."""
        episodes, store = await open_temporal(
            graph_backend, ("ana", "acme"), ConflictingStore, refusals=-1
        )
        recorder = Recorder(store)
        engine = TemporalEngine(
            store,
            ONE,
            LocalRefResolver(),
            FixedClock().now,
            hooks=[recorder],
        )

        report = await engine.ingest_observations([obs("o1", "acme", at(2020))])

        assert report.counts[Outcome.COMMIT_FAILED.value] == 1
        assert store.attempts == 2
        assert [call.name for call in recorder.calls] == ["before", "before"]

    async def test_late_observation_replaces_the_earlier_value(
        self, graph_backend: GraphBackend
    ):
        """A later stated value closes the earlier one and writes a supersession."""
        _, store = await open_temporal(
            graph_backend,
            (
                "ana",
                "acme",
                "globex",
            ),
        )
        engine = make_engine(store)
        await engine.ingest_observations([obs("o1", "acme", at(2020))])

        report = await engine.ingest_observations([obs("o2", "globex", at(2022))])

        assert report.counts[Outcome.SLOT_COMMITTED.value] == 1
        current = await store.query(TemporalQuery(now=NOW))
        assert [(v.object_id, v.valid_to) for v in current] == [("globex", None)]
        closed = await store.query(TemporalQuery(now=NOW, valid_at=at(2021)))
        assert [(v.object_id, v.valid_to) for v in closed] == [("acme", at(2022))]
        (row,) = await store.supersessions_of_slot(SLOT)
        assert (row.kind, row.source_obs_id, row.applied_valid_to) == (
            SupersessionKind.REPLACEMENT,
            "o2",
            at(2022),
        )

    @pytest.mark.parametrize("order", list(itertools.permutations(range(3))))
    async def test_timeline_does_not_depend_on_observation_order(
        self, graph_backend: GraphBackend, order
    ):
        """Every order of the same observations gives the same stored timeline."""
        observations = [
            obs("o1", "acme", at(2020)),
            obs("o2", "globex", at(2022)),
            obs("o3", "initech", at(2024)),
        ]
        _, reference = await open_temporal(
            graph_backend,
            (
                "ana",
                "acme",
                "globex",
                "initech",
            ),
        )
        await make_engine(reference).ingest_observations(observations)
        _, shuffled = await open_temporal(
            graph_backend,
            (
                "ana",
                "acme",
                "globex",
                "initech",
            ),
        )

        await make_engine(shuffled).ingest_observations(
            [observations[i] for i in order]
        )

        ours = await shuffled.load_slot(SLOT)
        theirs = await reference.load_slot(SLOT)
        assert [
            (f.fact_id, f.version, f.object_id, f.valid_from, f.valid_to)
            for f in ours.facts
        ] == [
            (f.fact_id, f.version, f.object_id, f.valid_from, f.valid_to)
            for f in theirs.facts
        ]

    async def test_ingest_observations_stores_each_observation_once(
        self, graph_backend: GraphBackend
    ):
        """Repeating an observation is not written twice and reports no change."""
        _, store = await open_temporal(
            graph_backend,
            (
                "ana",
                "acme",
            ),
        )
        engine = make_engine(store)
        first = obs("o1", "acme", at(2020))
        await engine.ingest_observations([first])

        second = await engine.ingest_observations([first])

        assert second.counts[Outcome.SLOT_UNCHANGED.value] == 1
        assert len(await store.observations_of_slot(SLOT)) == 1

    async def test_reconcile_all_reports_every_settled_slot_unchanged(
        self, graph_backend: GraphBackend
    ):
        """A sweep over settled slots commits nothing."""
        _, store = await open_temporal(
            graph_backend,
            (
                "ana",
                "acme",
                "globex",
            ),
        )
        engine = make_engine(store)
        await engine.ingest_observations([obs("o1", "acme", at(2020))])
        ledger_before = await store.ledger_entries(SLOT)

        report = await engine.reconcile_all()

        assert report.counts[Outcome.SLOT_UNCHANGED.value] == 1
        assert await store.ledger_entries(SLOT) == ledger_before
        assert verify(ledger_before) == []
