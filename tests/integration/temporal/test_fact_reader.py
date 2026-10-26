"""FactReader queries over engine-built temporal history and conflict state."""

from dataclasses import replace
from datetime import timedelta

from ontogr_mem.data_models.fact_versions import FactStatus
from ontogr_mem.data_models.observation import ObservationKind
from ontogr_mem.temporal.cardinality import (
    Cardinality,
    DictRelationPolicy,
    RelationRules,
)
from ontogr_mem.temporal.engine import TemporalEngine
from ontogr_mem.temporal.query import TemporalQuery
from ontogr_mem.temporal.reader import FactReader, ObservationText
from ontogr_mem.temporal.verdicts import (
    ConflictDecision,
    DecisionAction,
    VerdictKind,
    VerdictSource,
)
from tests.integration.temporal.test_engine_contradictions import ScriptedJudge
from tests.support.clock import FixedClock
from tests.support.episodes import make_episode
from tests.support.graph_backend import GraphBackend
from tests.support.temporal import open_temporal
from tests.unit.temporal.helpers import LocalRefResolver, at, obs


NOW = FixedClock().now()
ONE_AND_MANY = DictRelationPolicy(
    {
        1: RelationRules(Cardinality.ONE),
        2: RelationRules(Cardinality.MANY),
    }
)


async def make_reader_world(graph_backend: GraphBackend):
    """Build facts with replacements, correction and reset through the engine."""
    episodes, store = await open_temporal(
        graph_backend, ("ana", "acme", "globex", "paris", "rome")
    )
    clock = FixedClock(NOW)
    engine = TemporalEngine(store, ONE_AND_MANY, LocalRefResolver(), clock.now)
    await episodes.write_episodes(
        [make_episode("reset-episode"), make_episode("kept-episode")]
    )
    await engine.ingest_observations(
        [
            replace(
                obs("stable-1", "paris", at(2000), slot="stable"),
                statement="Ana lives in Paris.",
                quote="I live in Paris.",
                source_system="test-source",
            ),
            obs("closed-old", "acme", at(2020), slot="closed"),
            obs("closed-new", "globex", at(2022), slot="closed"),
            obs("correction-old", "acme", at(2020), slot="correction"),
            obs(
                "reset-observation",
                "rome",
                at(2020),
                slot="withdrawn",
                episode="reset-episode",
            ),
            obs(
                "kept-observation",
                "paris",
                at(2020),
                slot="kept",
                episode="kept-episode",
            ),
            obs(
                "inverse-observation",
                "paris",
                at(2020),
                slot="inverse",
                inverse=True,
                holder="ana",
            ),
        ]
    )
    stable = replace(
        obs("stable-2", "paris", at(2000), slot="stable"),
        statement="Ana lives in Paris.",
        quote="I still live in Paris.",
        source_system="test-source",
    )
    await engine.ingest_observations([stable])

    correction_time = NOW + timedelta(days=1)
    clock.value = correction_time
    await engine.ingest_observations(
        [
            obs(
                "correction-new",
                "globex",
                at(2022),
                kind=ObservationKind.CORRECTION,
                slot="correction",
            )
        ]
    )
    await engine.reset_episode("reset-episode", _NoopMentions())
    return engine, store, clock, FactReader(store)


class _NoopMentions:
    """Keep entity mentions while resetting the temporal episode."""

    async def remove_episode_mentions(self, episode_id: str) -> list[str]:
        """Leave entities in place because another episode still uses them."""
        return []


async def make_dispute_world(graph_backend: GraphBackend):
    """Build two cross-slot facts that remain in an open dispute."""
    _, store = await open_temporal(graph_backend, ("ana", "acme", "globex"))
    clock = FixedClock(NOW)
    engine = TemporalEngine(store, ONE_AND_MANY, LocalRefResolver(), clock.now)
    await engine.ingest_observations(
        [
            obs("dispute-a", "acme", at(2020), slot="dispute-a", rel=2),
            obs("dispute-b", "globex", at(2020), slot="dispute-b", rel=2),
        ]
    )
    await engine.judge_pass(ScriptedJudge(VerdictKind.UNCERTAIN))
    return engine, store, clock, FactReader(store)


class TestFactReader:
    """Read fact versions through temporal and evidence queries."""

    async def test_queries_apply_time_visibility_and_annotate_evidence(
        self, graph_backend: GraphBackend
    ):
        """Every query preserves temporal rules and adds observation text."""
        _, store, clock, reader = await make_reader_world(graph_backend)
        before_correction = clock.value - timedelta(microseconds=1)
        correction_slot = await store.load_slot("correction")
        original = next(
            f for f in correction_slot.facts if f.status is FactStatus.CORRECTED
        )
        closed_slot = await store.load_slot("closed")
        closed = next(f for f in closed_slot.facts if f.status is FactStatus.CLOSED)
        successor = next(f for f in closed_slot.facts if f.status is FactStatus.ACTIVE)
        withdrawn_slot = await store.load_slot("withdrawn")
        withdrawn = next(
            f for f in withdrawn_slot.facts if f.status is FactStatus.WITHDRAWN
        )

        known_then = await reader.by_ids(
            [original.fact_id],
            TemporalQuery(
                now=clock.value,
                valid_at=at(2021),
                known_at=before_correction,
            ),
        )
        at_past = await reader.by_ids(
            [closed.fact_id], TemporalQuery(now=clock.value, valid_at=at(2021))
        )
        current = await reader.by_ids(
            [original.fact_id, withdrawn.fact_id], TemporalQuery(now=clock.value)
        )
        missing = await reader.by_ids(["unknown"], TemporalQuery(now=clock.value))
        history = await reader.by_ids(
            [original.fact_id, withdrawn.fact_id],
            TemporalQuery(now=clock.value, include_history=True),
        )
        by_observation = await reader.by_observations(
            ["stable-1"], TemporalQuery(now=clock.value, include_history=True)
        )
        reader_query = TemporalQuery(now=clock.value, valid_at=at(2021))
        read = await reader.read(reader_query)
        stored_query = await store.query(reader_query)

        assert [(v.fact_id, v.version, v.status) for v in known_then] == [
            (original.fact_id, 1, FactStatus.ACTIVE)
        ]
        assert [(v.fact_id, v.version) for v in at_past] == [
            (closed.fact_id, closed.version)
        ]
        assert current == []
        assert missing == []
        assert {v.status for v in history} == {
            FactStatus.CORRECTED,
            FactStatus.WITHDRAWN,
            FactStatus.ACTIVE,
        }
        assert [(v.version, v.evidence) for v in by_observation] == [
            (1, ("stable-1",)),
            (2, ("stable-1", "stable-2")),
        ]
        assert [(v.fact_id, v.version) for v in read] == [
            (v.fact_id, v.version) for v in stored_query
        ]
        assert any(v.fact_id == closed.fact_id for v in read)
        assert all(v.fact_id != successor.fact_id for v in read)
        stable_view = next(v for v in read if v.slot_key == "stable")
        assert (stable_view.statement, stable_view.source) == (
            "Ana lives in Paris.",
            "test-source",
        )
        assert (
            await reader.by_observations(
                ["reset-observation"], TemporalQuery(now=clock.value)
            )
            == []
        )

    async def test_touching_matches_either_entity_side(
        self, graph_backend: GraphBackend
    ):
        """Entity lookup returns annotated facts with that entity on either side."""
        _, _, _, reader = await make_reader_world(graph_backend)

        touching = await reader.touching(["paris"], TemporalQuery(now=NOW))

        assert {view.slot_key for view in touching} == {"stable", "kept", "inverse"}
        assert all(
            view.statement or view.slot_key in {"kept", "inverse"} for view in touching
        )

    async def test_subset_of_open_dispute_keeps_its_counterpart_annotation(
        self, graph_backend: GraphBackend
    ):
        """A single requested side still names the other disputed fact."""
        _, store, clock, reader = await make_dispute_world(graph_backend)
        facts = await store.facts_of_entities(["ana"])
        first, second = sorted(facts, key=lambda fact: fact.slot_key)

        [view] = await reader.by_ids([first.fact_id], TemporalQuery(now=clock.value))

        assert view.disputed is True
        assert view.counterpart_ids == (second.fact_id,)
        assert view.dispute_reason == "the judge could not decide"

    async def test_person_decision_clears_dispute_and_overrides_the_loser(
        self, graph_backend: GraphBackend
    ):
        """Choosing one side settles the dispute and leaves the loser visible."""
        engine, store, clock, reader = await make_dispute_world(graph_backend)
        facts = sorted(await store.facts_of_entities(["ana"]), key=lambda f: f.slot_key)
        chosen, loser = facts
        decision = ConflictDecision(
            decision_id="decision-1",
            conflict_id="conflict-1",
            fact_ids=(chosen.fact_id, loser.fact_id),
            action=DecisionAction.CHOOSE,
            decided_by=VerdictSource.USER,
            at=NOW,
            chosen_fact_id=chosen.fact_id,
        )

        clock.value = NOW + timedelta(days=1)
        await engine.record_decision(decision)
        views = await reader.by_ids(
            [chosen.fact_id, loser.fact_id], TemporalQuery(now=clock.value)
        )

        assert {view.fact_id for view in views} == {chosen.fact_id, loser.fact_id}
        assert all(not view.disputed for view in views)
        assert next(view for view in views if view.fact_id == loser.fact_id).status is (
            FactStatus.OVERRIDDEN
        )

    async def test_observation_texts_include_episode_and_omit_unknown_ids(
        self, graph_backend: GraphBackend
    ):
        """Evidence text carries its episode id and ignores missing observations."""
        _, _, _, reader = await make_reader_world(graph_backend)

        texts = await reader.observation_texts(["stable-1", "missing"])

        assert texts == {
            "stable-1": ObservationText(
                statement="Ana lives in Paris.",
                quote="I live in Paris.",
                episode_id="ep-stable-1",
            )
        }
