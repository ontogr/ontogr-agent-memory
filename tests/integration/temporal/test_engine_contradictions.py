"""Contradiction handling through the engine, on a live graph tenant.

Each test ingests observations, then checks the stored history and the current
view that a reader gets. The judge is a scripted pair judge.
"""

from datetime import UTC, datetime

from ontogr_mem.data_models.fact_versions import FactStatus
from ontogr_mem.data_models.observation import ObservationKind
from ontogr_mem.graphstore.temporal.store import GraphTemporalStore
from ontogr_mem.temporal.cardinality import (
    Cardinality,
    DictRelationPolicy,
    RelationRules,
)
from ontogr_mem.temporal.engine import TemporalEngine
from ontogr_mem.temporal.query import TemporalQuery
from ontogr_mem.temporal.report import Outcome
from ontogr_mem.temporal.verdicts import (
    ConflictDecision,
    DecisionAction,
    VerdictKind,
    VerdictSource,
)
from tests.support.clock import FixedClock
from tests.support.graph_backend import GraphBackend
from tests.support.temporal import open_temporal
from tests.support.verdicts import verdict
from tests.unit.temporal.helpers import LocalRefResolver, at, obs


NOW = datetime(2026, 10, 22, tzinfo=UTC)
RULES = DictRelationPolicy(
    {
        1: RelationRules(Cardinality.ONE),
        2: RelationRules(Cardinality.MANY),
    }
)


async def build(
    graph_backend: GraphBackend,
) -> tuple[TemporalEngine, GraphTemporalStore]:
    """An engine over a fresh tenant that knows the three entities."""
    _, store = await open_temporal(graph_backend, ("ana", "acme", "globex"))
    engine = TemporalEngine(store, RULES, LocalRefResolver(), FixedClock().now)
    return engine, store


async def ingest_city_and_employer(engine: TemporalEngine) -> None:
    """Ana worked at acme from 2010 (one-valued slot) and globex from 2015 (many)."""
    await engine.ingest_observations(
        [
            obs("m1", "acme", at(2010)),
            obs("p1", "globex", at(2015), slot="other", rel=2),
        ]
    )


async def fact_id_of(store: GraphTemporalStore, obj: str) -> str:
    """The id of the current fact that names ``obj``."""
    [fact] = [f for f in await store.current_facts() if f.object_id == obj]
    return fact.fact_id


async def current_objects(store: GraphTemporalStore) -> list[str]:
    """Objects of the facts a reader sees today."""
    views = await store.query(TemporalQuery(now=NOW))
    return sorted(view.object_id for view in views)


async def test_a_correction_keeps_the_wrong_value_in_history_as_corrected(
    graph_backend: GraphBackend,
):
    """The corrected value leaves the current view and stays in the history."""
    engine, store = await build(graph_backend)

    await engine.ingest_observations(
        [
            obs("o1", "acme", at(2020)),
            obs("o2", "globex", at(2022), kind=ObservationKind.CORRECTION),
        ]
    )

    assert await current_objects(store) == ["globex"]
    history = await store.query(TemporalQuery(now=NOW, include_history=True))
    statuses = sorted((v.object_id, v.status) for v in history)
    assert statuses == [
        ("acme", FactStatus.CORRECTED),
        ("globex", FactStatus.ACTIVE),
    ]


async def test_record_verdicts_then_reconcile_applies_them(graph_backend: GraphBackend):
    """A stored contradiction closes the local fact where the other one starts."""
    engine, store = await build(graph_backend)
    await ingest_city_and_employer(engine)
    assert await current_objects(store) == ["acme", "globex"]

    facts = await store.current_facts()
    report = await engine.record_verdicts(
        [await verdict(store, facts[0], facts[1], VerdictKind.CONTRADICTS, lambda: NOW)]
    )

    assert report.counts[Outcome.JUDGE_CONTRADICTS.value] == 1
    assert await current_objects(store) == ["globex"]
    believed = TemporalQuery(now=NOW, include_history=True, known_at=NOW)
    closed = [v for v in await store.query(believed) if v.object_id == "acme"]
    assert [(v.valid_to, v.status) for v in closed] == [(at(2015), FactStatus.CLOSED)]


async def test_person_decision_overrides_rules_and_survives_rerun(
    graph_backend: GraphBackend,
):
    """A person's choice keeps the chosen fact open, even after a reconcile."""
    engine, store = await build(graph_backend)
    await ingest_city_and_employer(engine)
    decision = ConflictDecision(
        decision_id="d1",
        conflict_id="c1",
        fact_ids=(await fact_id_of(store, "acme"), await fact_id_of(store, "globex")),
        action=DecisionAction.CHOOSE,
        decided_by=VerdictSource.USER,
        at=datetime(2026, 10, 1, tzinfo=UTC),
        chosen_fact_id=await fact_id_of(store, "acme"),
    )

    await engine.record_decision(decision)
    facts = await store.current_facts()
    await engine.record_verdicts(
        [await verdict(store, facts[0], facts[1], VerdictKind.CONTRADICTS, lambda: NOW)]
    )
    rerun = await engine.reconcile_all()

    assert await current_objects(store) == ["acme", "globex"]
    assert rerun.counts[Outcome.SLOT_COMMITTED.value] == 0
