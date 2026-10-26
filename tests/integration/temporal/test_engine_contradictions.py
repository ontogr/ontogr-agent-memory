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
from ontogr_mem.temporal.judge import JudgeOutcome
from ontogr_mem.temporal.query import TemporalQuery
from ontogr_mem.temporal.report import Outcome
from ontogr_mem.temporal.verdicts import (
    ConflictDecision,
    DecisionAction,
    FactVerdict,
    VerdictKind,
    VerdictSource,
)
from tests.support.clock import FixedClock
from tests.support.graph_backend import GraphBackend
from tests.support.temporal import open_temporal
from tests.unit.temporal.helpers import LocalRefResolver, at, obs


NOW = datetime(2026, 10, 22, tzinfo=UTC)
RULES = DictRelationPolicy(
    {
        1: RelationRules(Cardinality.ONE),
        2: RelationRules(Cardinality.MANY),
    }
)


class ScriptedJudge:
    """A judge that gives one verdict to every pair it is asked about."""

    prompt_version = "scripted-1"

    def __init__(self, verdict: VerdictKind | None) -> None:
        """Answer every pair with ``verdict``; ``None`` answers nothing."""
        self._verdict = verdict
        self.asked: list[str] = []

    async def judge_pairs(self, pairs, *, now: datetime) -> JudgeOutcome:
        """Record the pairs and return the scripted verdicts."""
        self.asked.extend(pair.pair_key for pair in pairs)
        outcome = JudgeOutcome()
        if self._verdict is None:
            return outcome
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

    report = await engine.judge_pass(ScriptedJudge(VerdictKind.CONTRADICTS))

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
    await engine.judge_pass(ScriptedJudge(VerdictKind.CONTRADICTS))
    rerun = await engine.reconcile_all()

    assert await current_objects(store) == ["acme", "globex"]
    assert rerun.counts[Outcome.SLOT_COMMITTED.value] == 0


async def test_judge_pass_judges_only_pending_slots_and_clears_the_flag(
    graph_backend: GraphBackend,
):
    """A pending slot is judged once; a second pass has nothing to ask."""
    engine, store = await build(graph_backend)
    await ingest_city_and_employer(engine)
    judge = ScriptedJudge(VerdictKind.COMPATIBLE)

    await engine.judge_pass(judge)
    second = await engine.judge_pass(judge)

    assert len(judge.asked) == 1
    assert await store.pending_slots() == []
    assert second.counts == {}


async def test_unjudged_partner_leaves_the_slot_pending_until_judged(
    graph_backend: GraphBackend,
):
    """A pass that answers nothing leaves the slots with a neighbour pending."""
    engine, store = await build(graph_backend)
    await ingest_city_and_employer(engine)

    await engine.judge_pass(ScriptedJudge(None))

    assert await store.pending_slots() == ["other", "slot"]


async def test_two_single_valued_slots_are_not_sent_to_the_judge(
    graph_backend: GraphBackend,
):
    """The rules settle one-per-subject relations, so the judge is not asked."""
    engine, store = await build(graph_backend)
    judge = ScriptedJudge(VerdictKind.COMPATIBLE)
    await engine.ingest_observations(
        [
            obs("m1", "acme", at(2010)),
            obs("p1", "globex", at(2015), slot="other"),
        ]
    )

    report = await engine.judge_pass(judge)

    assert judge.asked == []
    assert report.counts == {}
