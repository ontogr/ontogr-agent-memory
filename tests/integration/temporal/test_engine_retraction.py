"""Retraction, reset and recovery through the engine on a live graph tenant."""

import contextlib
from datetime import timedelta

from ontogr_mem.data_models.episode import TemporalStatus
from ontogr_mem.data_models.fact_versions import FactStatus
from ontogr_mem.temporal.audit import audit
from ontogr_mem.temporal.ledger import TENANT_CHAIN, slot_chain, verify
from ontogr_mem.temporal.query import TemporalQuery
from ontogr_mem.temporal.report import Outcome
from ontogr_mem.temporal.temporal_store import EpisodeState, SourceSelector
from tests.integration.temporal.test_engine import make_engine, typed
from tests.support.clock import FixedClock
from tests.support.episodes import make_episode
from tests.support.graph_backend import GraphBackend
from tests.support.temporal import open_temporal


class FakeMentions:
    """Records which episodes had their mentions removed."""

    def __init__(self) -> None:
        """Start with no removals."""
        self.removed: list[str] = []

    async def remove_episode_mentions(self, episode_id: str) -> list[str]:
        """Record the episode and delete nothing."""
        self.removed.append(episode_id)
        return []


class FlakyStore:
    """Wraps a store and fails the first commit marker, as a crash would."""

    def __init__(self, inner, crash_once: bool = True) -> None:
        """Wrap `inner` and fail once when `crash_once` is set."""
        self._inner = inner
        self._crash = crash_once

    def __getattr__(self, name: str):
        """Delegate every other store verb to the wrapped store."""
        return getattr(self._inner, name)

    async def mark_committed(self, episode_id: str, content_hash: str) -> None:
        """Fail once, then delegate."""
        if self._crash:
            self._crash = False
            raise RuntimeError("crash before the commit marker")
        await self._inner.mark_committed(episode_id, content_hash)


async def _ingest(episodes, store, episode_id: str = "ep1"):
    engine = make_engine(store)
    await episodes.write_episodes([make_episode(episode_id)])
    return await engine.ingest([typed()], make_episode(episode_id))


class TestTemporalRetraction:
    """Removing an episode withdraws its facts and records the retraction."""

    async def test_removed_episode_withdraws_its_fact_and_keeps_history(
        self, graph_backend: GraphBackend
    ):
        """The fact becomes a tombstone that is out of the current view."""
        episodes, store = await open_temporal(
            graph_backend,
            (
                "ana",
                "paris",
            ),
        )
        await _ingest(episodes, store)
        engine = make_engine(store)

        await engine.remove_episode("ep1")

        assert await store.current_facts() == []
        history = await store.query(
            TemporalQuery(now=FixedClock().now(), include_history=True)
        )
        assert [v.status for v in history] == [FactStatus.ACTIVE, FactStatus.WITHDRAWN]

    async def test_retraction_writes_one_entry_and_marks_the_episode(
        self, graph_backend: GraphBackend
    ):
        """The tenant chain gets one retraction entry; the episode node stays."""
        episodes, store = await open_temporal(
            graph_backend,
            (
                "ana",
                "paris",
            ),
        )
        await _ingest(episodes, store)
        engine = make_engine(store)

        report = await engine.remove_episode("ep1")
        await engine.remove_episode("ep1")

        ops = [e.op.value for e in await store.ledger_entries(TENANT_CHAIN)]
        assert ops == ["episode_committed", "episode_retracted"]
        state = await store.episode_state("ep1")
        assert state == EpisodeState(TemporalStatus.RETRACTED, "hash-ep1")
        assert report.counts[Outcome.EPISODE_RETRACTED.value] == 1

    async def test_ledger_and_audit_verify_after_retraction(
        self, graph_backend: GraphBackend
    ):
        """Every chain verifies and the audit reports no violation."""
        episodes, store = await open_temporal(
            graph_backend,
            (
                "ana",
                "paris",
            ),
        )
        await _ingest(episodes, store)
        await make_engine(store).remove_episode("ep1")

        assert verify(await store.ledger_entries(TENANT_CHAIN)) == []
        for slot_key in await store.slot_keys():
            assert verify(await store.ledger_entries(slot_chain(slot_key))) == []
        assert await audit(store) == []

    async def test_retract_source_retracts_every_matching_episode(
        self, graph_backend: GraphBackend
    ):
        """Every episode of the selected source is removed, others are kept."""
        episodes, store = await open_temporal(
            graph_backend,
            (
                "ana",
                "paris",
            ),
        )
        await _ingest(episodes, store)
        await make_engine(store).retract_source(SourceSelector(system="chat"))

        assert await store.current_facts() == []
        state = await store.episode_state("ep1")
        assert state is not None and state.status is TemporalStatus.RETRACTED


class TestTemporalReset:
    """Reset clears an episode so it can be ingested again."""

    async def test_reset_withdraws_clears_state_and_removes_mentions(
        self, graph_backend: GraphBackend
    ):
        """No retraction entry is written; mentions are removed once."""
        episodes, store = await open_temporal(
            graph_backend,
            (
                "ana",
                "paris",
            ),
        )
        await _ingest(episodes, store)
        mentions = FakeMentions()

        await make_engine(store).reset_episode("ep1", mentions)

        assert mentions.removed == ["ep1"]
        assert await store.episode_state("ep1") is None
        ops = [e.op.value for e in await store.ledger_entries(TENANT_CHAIN)]
        assert "episode_retracted" not in ops
        assert await store.current_facts() == []

    async def test_episode_can_be_ingested_again_after_reset(
        self, graph_backend: GraphBackend
    ):
        """A reset episode ingests again and the new fact is live."""
        episodes, store = await open_temporal(
            graph_backend,
            (
                "ana",
                "paris",
            ),
        )
        await _ingest(episodes, store)
        await make_engine(store).reset_episode("ep1", FakeMentions())

        await make_engine(store).ingest([typed()], make_episode("ep1"))

        live = await store.current_facts()
        assert [f.status for f in live] == [FactStatus.ACTIVE]
        assert await audit(store) == []

    async def test_reset_of_an_episode_without_observations_succeeds(
        self, graph_backend: GraphBackend
    ):
        """Reset is idempotent: a second call finds nothing to remove."""
        episodes, store = await open_temporal(
            graph_backend,
            (
                "ana",
                "paris",
            ),
        )
        await _ingest(episodes, store)
        engine = make_engine(store)
        await engine.reset_episode("ep1", FakeMentions())

        await engine.reset_episode("ep1", FakeMentions())

        assert await store.current_facts() == []


class TestTemporalRecovery:
    """Recovery finishes an episode whose commit marker is missing."""

    async def test_recover_commits_an_episode_with_observations(
        self, graph_backend: GraphBackend
    ):
        """A crash before the marker leaves PENDING; recover commits it."""
        episodes, inner = await open_temporal(
            graph_backend,
            (
                "ana",
                "paris",
            ),
        )
        store = FlakyStore(inner)
        await episodes.write_episodes([make_episode("ep1")])
        await episodes.record_content_hash("ep1", "hash-ep1")
        with contextlib.suppress(RuntimeError):
            await make_engine(store).ingest([typed()], make_episode("ep1"))
        assert (await inner.episode_state("ep1")).status is TemporalStatus.PENDING

        report = await make_engine(inner).recover()

        assert (await inner.episode_state("ep1")).status is TemporalStatus.COMMITTED
        assert report.counts[Outcome.EPISODE_RECOVERED.value] == 1
        assert await audit(inner) == []

    async def test_recover_leaves_an_episode_without_observations_pending(
        self, graph_backend: GraphBackend
    ):
        """An episode with no observations is reported and stays pending."""
        episodes, store = await open_temporal(
            graph_backend,
            (
                "ana",
                "paris",
            ),
        )
        await episodes.write_episodes([make_episode("ep1")])
        await episodes.record_content_hash("ep1", "hash-ep1")
        await store.set_episode_pending("ep1")

        report = await make_engine(store).recover()

        state = await store.episode_state("ep1")
        assert state is not None and state.status is TemporalStatus.PENDING
        assert report.counts[Outcome.EPISODE_NOT_RECOVERABLE.value] == 1

    async def test_audit_reports_an_episode_pending_past_its_ttl(
        self, graph_backend: GraphBackend
    ):
        """A pending episode older than the TTL is a violation."""
        episodes, store = await open_temporal(
            graph_backend,
            (
                "ana",
                "paris",
            ),
        )
        await episodes.write_episodes([make_episode("ep1")])
        await episodes.record_content_hash("ep1", "hash-ep1")
        await store.set_episode_pending("ep1")
        now = make_episode("ep1").ingested_at + timedelta(days=2)

        violations = await audit(store, now=now, episode_ttl=timedelta(hours=1))

        assert [v.code for v in violations] == ["episode_not_committed"]
