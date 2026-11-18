"""Integration coverage for temporal closure behavior."""

from datetime import UTC, datetime
from types import SimpleNamespace

from ontogr_mem.data_models.facts import FactKind, Modality, Precision, TimeBasis
from ontogr_mem.data_models.observation import Observation, ObservationKind
from ontogr_mem.temporal.cardinality import (
    Cardinality,
    DictRelationPolicy,
    RelationRules,
)
from ontogr_mem.temporal.engine import TemporalEngine
from ontogr_mem.temporal.report import Outcome
from ontogr_mem.temporal.verdicts import VerdictKind
from tests.support.clock import FixedClock
from tests.support.graph_backend import GraphBackend
from tests.support.temporal import open_temporal
from tests.unit.temporal.helpers import LocalRefResolver, at, obs


class ScriptedClosureJudge:
    """Return a fixed contradiction result and retain the judged inputs."""

    prompt_version = "closure-test-1"

    def __init__(self, contradicted: tuple[str, ...] = ()) -> None:
        """Initialize the judge with fact ids to contradict."""
        self.contradicted = contradicted
        self.calls: list[tuple[str, list[tuple[str, str]]]] = []

    async def judge(self, new_text, existing):
        """Record the pair and return a successful scripted verdict."""
        self.calls.append((new_text, list(existing)))
        return SimpleNamespace(
            contradicted=self.contradicted,
            invalid_ids=0,
            failed=False,
            counts={},
            input_tokens=0,
            output_tokens=0,
        )


async def test_closure_pass_records_contradiction_and_closes_earlier_fact(
    graph_backend: GraphBackend,
) -> None:
    """Store a contradiction verdict and close the replaced fact."""
    _, store = await open_temporal(graph_backend, ("ana", "one", "two"))
    engine = TemporalEngine(
        store,
        DictRelationPolicy({2: RelationRules(Cardinality.ONE, judged=True)}),
        LocalRefResolver(),
        FixedClock(datetime(2026, 11, 18, tzinfo=UTC)).now,
    )

    await engine.ingest_observations([obs("old", "one", at(2010), slot="old", rel=2)])
    old_fact = next(f for f in await store.current_facts() if f.object_id == "one")
    await engine.ingest_observations(
        [
            Observation(
                obs_id="new",
                slot_key="new",
                subject_id="ana",
                object_id="two",
                relation_type_id=2,
                kind=ObservationKind.ASSERTION,
                episode_id="ep-new",
                occurred_at=at(2015),
                ingested_at=at(2015),
                valid_from=at(2015),
                valid_from_precision=Precision.YEAR,
                time_basis=TimeBasis.STATED,
                fact_kind=FactKind.STATE,
                modality=Modality.ACTUAL,
                relation_label="employer",
                statement="Ana works for Two",
            )
        ]
    )
    new_fact = next(f for f in await store.current_facts() if f.object_id == "two")
    judge = ScriptedClosureJudge((old_fact.fact_id,))

    report = await engine.closure_pass(judge, [new_fact.fact_id], max_candidates=10)

    assert len(judge.calls) == 1
    assert report.counts[Outcome.JUDGE_ASKED.value] == 1
    verdicts = await store.verdicts_for([old_fact.fact_id, new_fact.fact_id])
    assert VerdictKind.CONTRADICTS in {verdict.verdict for verdict in verdicts}
    assert judge.calls[0][0] == "Ana works for Two"
