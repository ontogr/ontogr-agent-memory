"""Contradiction scenarios through the engine on FalkorDB and Neo4j.

Each scenario ingests observations, runs the judge or a person decision where
the scenario needs one, and checks the current view and the history with literal
values. The judge is scripted, so no model is called.
"""

from datetime import UTC, datetime

import pytest

from ontogr_mem.data_models.fact_versions import FactStatus
from ontogr_mem.data_models.observation import ObservationKind
from ontogr_mem.graphstore.falkordb import FalkorDBGraphStore
from ontogr_mem.graphstore.neo4j import Neo4jGraphStore
from ontogr_mem.graphstore.schemas import EPISODES, RESOLUTION, TEMPORAL
from ontogr_mem.graphstore.temporal.store import GraphTemporalStore
from ontogr_mem.temporal.cardinality import (
    Cardinality,
    DictRelationPolicy,
    RelationRules,
)
from ontogr_mem.temporal.engine import TemporalEngine
from ontogr_mem.temporal.judge import JudgeOutcome
from ontogr_mem.temporal.query import TemporalQuery
from ontogr_mem.temporal.verdicts import (
    ConflictDecision,
    DecisionAction,
    FactVerdict,
    VerdictKind,
    VerdictSource,
)
from tests.support.clock import FixedClock
from tests.support.temporal import graph_temporal_harness
from tests.support.tenants import new_tenant
from tests.unit.temporal.helpers import LocalRefResolver, at, obs


NOW = datetime(2026, 10, 22, tzinfo=UTC)
RULES = DictRelationPolicy(
    {
        1: RelationRules(Cardinality.ONE),
        2: RelationRules(Cardinality.MANY),
    }
)
ENTITIES = ["ana", "acme", "globex"]


class ScriptedJudge:
    """A judge that answers every pair with one verdict."""

    prompt_version = "scripted-1"

    def __init__(self, verdict: VerdictKind) -> None:
        """Answer every pair with ``verdict``."""
        self._verdict = verdict

    async def judge_pairs(self, pairs, *, now: datetime) -> JudgeOutcome:
        """Return the scripted verdict for each pair."""
        outcome = JudgeOutcome()
        for pair in pairs:
            outcome.verdicts.append(
                FactVerdict(
                    pair_key=pair.pair_key,
                    fact_a_id=pair.a.fact_id,
                    fact_b_id=pair.b.fact_id,
                    verdict=self._verdict,
                    inputs_hash=pair.inputs_hash,
                    prompt_version=self.prompt_version,
                    source=VerdictSource.LLM,
                    decided_at=now,
                )
            )
        return outcome


class ScenarioContract:
    """Scenarios every graph backend must show. Subclasses provide `make_store`."""

    async def _engine(self, make_store) -> tuple[TemporalEngine, object]:
        store = await make_store()
        await store.seed_entities(ENTITIES)
        engine = TemporalEngine(
            store.store, RULES, LocalRefResolver(), FixedClock().now
        )
        return engine, store.store

    async def test_a_correction_leaves_the_wrong_value_corrected_in_history(
        self, make_store
    ):
        """The corrected employer is out of the current view and kept as history."""
        engine, store = await self._engine(make_store)

        await engine.ingest_observations(
            [
                obs("o1", "acme", at(2020)),
                obs("o2", "globex", at(2022), kind=ObservationKind.CORRECTION),
            ]
        )

        current = await store.query(TemporalQuery(now=NOW))
        history = await store.query(TemporalQuery(now=NOW, include_history=True))
        assert [v.object_id for v in current] == ["globex"]
        assert sorted((v.object_id, v.status) for v in history) == [
            ("acme", FactStatus.CORRECTED),
            ("globex", FactStatus.ACTIVE),
        ]

    async def test_a_judged_contradiction_closes_the_earlier_fact(self, make_store):
        """A contradiction across slots ends the earlier fact at the later start."""
        engine, store = await self._engine(make_store)
        await engine.ingest_observations(
            [
                obs("m1", "acme", at(2010)),
                obs("p1", "globex", at(2015), slot="other", rel=2),
            ]
        )

        await engine.judge_pass(ScriptedJudge(VerdictKind.CONTRADICTS))

        current = await store.query(TemporalQuery(now=NOW))
        assert [v.object_id for v in current] == ["globex"]
        assert await store.pending_slots() == []

    async def test_a_person_choice_survives_a_judged_contradiction(self, make_store):
        """A CHOOSE decision keeps the chosen fact open after the judge answers."""
        engine, store = await self._engine(make_store)
        await engine.ingest_observations(
            [
                obs("m1", "acme", at(2010)),
                obs("p1", "globex", at(2015), slot="other", rel=2),
            ]
        )
        facts = {f.object_id: f.fact_id for f in await store.current_facts()}
        await engine.record_decision(
            ConflictDecision(
                decision_id="decision-1",
                conflict_id="conflict-1",
                fact_ids=(facts["acme"], facts["globex"]),
                action=DecisionAction.CHOOSE,
                decided_by=VerdictSource.USER,
                at=datetime(2026, 10, 1, tzinfo=UTC),
                chosen_fact_id=facts["acme"],
            )
        )

        await engine.judge_pass(ScriptedJudge(VerdictKind.CONTRADICTS))
        await engine.reconcile_all()

        current = await store.query(TemporalQuery(now=NOW))
        assert sorted(v.object_id for v in current) == ["acme", "globex"]


class TestFalkorDBScenarios(ScenarioContract):
    """Scenarios on FalkorDB."""

    @pytest.fixture
    async def make_store(self, falkordb_store: FalkorDBGraphStore):
        """Return a factory of fresh tenant stores, erased afterwards."""
        tenants: list[str] = []

        async def make():
            handle = await falkordb_store.open(
                new_tenant(), [EPISODES, RESOLUTION, TEMPORAL]
            )
            tenants.append(handle.tenant)
            return graph_temporal_harness(
                GraphTemporalStore(falkordb_store, handle),
                falkordb_store,
                handle,
                tenant_keyed=False,
            )

        yield make
        for tenant in tenants:
            await falkordb_store.erase_tenant(tenant)


class TestNeo4jScenarios(ScenarioContract):
    """Scenarios on Neo4j."""

    @pytest.fixture
    async def make_store(self, neo4j_store: Neo4jGraphStore):
        """Return a factory of fresh tenant stores, erased afterwards."""
        tenants: list[str] = []

        async def make():
            handle = await neo4j_store.open(
                new_tenant(), [EPISODES, RESOLUTION, TEMPORAL]
            )
            tenants.append(handle.tenant)
            return graph_temporal_harness(
                GraphTemporalStore(neo4j_store, handle),
                neo4j_store,
                handle,
                tenant_keyed=True,
            )

        yield make
        for tenant in tenants:
            await neo4j_store.erase_tenant(tenant)
