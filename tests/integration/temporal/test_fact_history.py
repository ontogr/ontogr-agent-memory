"""Fact history reads against a live graph tenant."""

from datetime import timedelta

from ontogr_mem.data_models.fact_versions import FactStatus, SupersessionKind
from ontogr_mem.temporal.lineage import fact_history
from ontogr_mem.temporal.reader import FactReader
from tests.support.graph_backend import GraphBackend
from tests.support.p9i import NOW, add_fact, make_world
from tests.unit.temporal.helpers import at


class TestFactHistory:
    """Read a fact slot's world and system history."""

    async def test_history_orders_versions_and_separates_both_ends(
        self, graph_backend: GraphBackend
    ):
        """A replacement closes the old world interval and system version."""
        episodes, _, store, clock, engine, _ = await make_world(
            graph_backend, ("ana", "berlin", "munich")
        )
        await add_fact(
            episodes,
            engine,
            observation_id="o-berlin",
            episode_id="ep-berlin",
            subject="ana",
            object_id="berlin",
            slot="residence",
            valid_from=at(2020),
            statement="Ana lives in Berlin.",
        )
        fact_id = (await store.load_slot("residence")).facts[0].fact_id
        clock.value += timedelta(days=1)
        await add_fact(
            episodes,
            engine,
            observation_id="o-munich",
            episode_id="ep-munich",
            subject="ana",
            object_id="munich",
            slot="residence",
            valid_from=at(2025),
            statement="Ana lives in Munich.",
        )

        result = await fact_history(store, FactReader(store), fact_id, now=clock.value)

        assert result is not None
        old, current, new_claim = result.versions
        assert [(item.version, item.status) for item in result.versions] == [
            (1, FactStatus.ACTIVE),
            (2, FactStatus.CLOSED),
            (1, FactStatus.ACTIVE),
        ]
        assert old.statement == "Ana lives in Berlin."
        assert current.statement == "Ana lives in Berlin."
        assert old.world_end is None
        assert old.superseded_at == current.recorded_at
        assert old.superseded_by_version == 2
        assert current.world_end is not None
        assert current.world_end.kind is SupersessionKind.REPLACEMENT
        assert current.world_end.episode_id == "ep-munich"
        assert current.superseded_at is None
        assert new_claim.statement == "Ana lives in Munich."
        assert new_claim.fact_id != old.fact_id
        assert new_claim.world_end is None

    async def test_time_filters_return_what_was_believed_then(
        self, graph_backend: GraphBackend
    ):
        """Known time limits versions independently of valid time."""
        episodes, _, store, clock, engine, _ = await make_world(
            graph_backend, ("ana", "berlin", "munich")
        )
        await add_fact(
            episodes,
            engine,
            observation_id="o-berlin",
            episode_id="ep-berlin",
            subject="ana",
            object_id="berlin",
            slot="residence",
            valid_from=at(2020),
            statement="Ana lives in Berlin.",
        )
        before_update = clock.value
        fact_id = (await store.load_slot("residence")).facts[0].fact_id
        clock.value += timedelta(days=1)
        await add_fact(
            episodes,
            engine,
            observation_id="o-munich",
            episode_id="ep-munich",
            subject="ana",
            object_id="munich",
            slot="residence",
            valid_from=at(2025),
            statement="Ana lives in Munich.",
        )

        before = await fact_history(
            store,
            FactReader(store),
            fact_id,
            now=clock.value,
            valid_at=at(2024),
            known_at=before_update,
        )
        after = await fact_history(
            store,
            FactReader(store),
            fact_id,
            now=clock.value,
            valid_at=at(2024),
        )
        known_after_update = await fact_history(
            store,
            FactReader(store),
            fact_id,
            now=clock.value,
            valid_at=at(2024),
            known_at=clock.value,
        )

        assert before is not None
        assert [item.version for item in before.versions] == [1]
        assert after is not None
        assert [item.version for item in after.versions] == [1, 2]
        assert known_after_update is not None
        assert [item.version for item in known_after_update.versions] == [2]

    async def test_withdrawn_tombstone_has_no_world_end_or_evidence(
        self, graph_backend: GraphBackend
    ):
        """Retracting the source leaves an auditable tombstone in history."""
        episodes, _, store, clock, engine, _ = await make_world(
            graph_backend, ("ana", "berlin")
        )
        await add_fact(
            episodes,
            engine,
            observation_id="o-berlin",
            episode_id="ep-berlin",
            subject="ana",
            object_id="berlin",
            slot="residence",
            valid_from=at(2020),
            statement="Ana lives in Berlin.",
        )
        fact_id = (await store.load_slot("residence")).facts[0].fact_id
        clock.value += timedelta(days=1)
        await engine.retract_episodes(["ep-berlin"])

        result = await fact_history(store, FactReader(store), fact_id, now=clock.value)

        assert result is not None
        tombstone = result.versions[-1]
        assert tombstone.status is FactStatus.WITHDRAWN
        assert tombstone.world_end is None
        assert tombstone.evidence == ()
        assert tombstone.episode_ids == ()
        assert result.versions[0].superseded_by_version == tombstone.version

    async def test_unknown_fact_returns_none(self, graph_backend: GraphBackend):
        """A fact id outside the tenant's store has no history."""
        _, _, store, _, _, _ = await make_world(graph_backend, ("ana",))

        result = await fact_history(store, FactReader(store), "missing", now=NOW)

        assert result is None
