"""The report stage on the in-memory twins: what a run found but did not act on.

Disputes are opened through the temporal engine with the same scripted judge and
two-valued rules that the temporal contradiction tests use, so the report reads
them the way a reader would see them.
"""

from ontogr_mem.consolidation.errors import StoreError
from ontogr_mem.consolidation.models import (
    Proposal,
    ProposalStatus,
    StageResult,
    StageStatus,
)
from ontogr_mem.consolidation.stage import StageContext
from ontogr_mem.consolidation.stages.report import ReportStage
from ontogr_mem.consolidation.store import M
from ontogr_mem.graphstore.memory import MemoryConsStore
from ontogr_mem.temporal.engine import TemporalEngine
from ontogr_mem.temporal.verdicts import VerdictKind
from tests.support.consolidation import ConsWorld, new_world
from tests.unit.temporal.helpers import at
from tests.unit.temporal.test_engine_contradictions import (
    RULES,
    ScriptedJudge,
    ingest_city_and_employer,
)


class FailingFindStore(MemoryConsStore):
    """A consolidation store whose lookups fail, as a broken backend would."""

    async def find(self, model: type[M], field: str, value: object) -> list[M]:
        """Raise a store error for every lookup."""
        raise StoreError(f"lookup of {model.__name__} failed")


def ctx_for(
    world: ConsWorld,
    *,
    results: list[StageResult] | None = None,
    store: MemoryConsStore | None = None,
) -> StageContext:
    """Build a stage context over the world, optionally with another store."""
    return StageContext(
        run_id="r1",
        clock=world.clock,
        store=store or world.store,
        settings=world.settings,
        resolution=world.resolution,
        temporal=world.temporal,
        results=results or [],
    )


def proposal(proposal_id: str, status: ProposalStatus) -> Proposal:
    """A reflection proposal in the given status."""
    return Proposal(
        proposal_id=proposal_id,
        session_id="s1",
        text="Ana prefers tea.",
        source_entry_ids=["e1"],
        status=status,
        created_at=at(2030),
    )


async def test_reports_both_facts_of_an_open_dispute() -> None:
    """A judge that cannot decide leaves both facts open and reported as disputed."""
    world = await new_world()
    people_and_places = [("ana", "Ana"), ("acme", "Acme"), ("globex", "Globex")]
    for entity_id, name in people_and_places:
        await world.add_entity(entity_id, name)
    await world.ensure_entities([entity_id for entity_id, _ in people_and_places])
    engine = TemporalEngine(
        world.temporal_store,
        RULES,
        world.resolution.store,
        world.clock,
    )
    await ingest_city_and_employer(engine)
    await engine.judge_pass(ScriptedJudge(VerdictKind.UNCERTAIN))

    result = await ReportStage().run(ctx_for(world))

    assert result.status is StageStatus.OK
    assert result.counts["disputed_facts"] == 2


async def test_reports_a_pending_proposal_and_not_a_decided_one() -> None:
    """Only proposals still awaiting a decision are counted as pending."""
    world = await new_world()
    await world.store.put(proposal("p1", ProposalStatus.PENDING))
    await world.store.put(proposal("p2", ProposalStatus.ACCEPTED))

    result = await ReportStage().run(ctx_for(world))

    assert result.status is StageStatus.OK
    assert result.counts["pending_proposals"] == 1


async def test_empty_tenant_reports_zeros_and_status_ok() -> None:
    """A tenant with nothing to report gets zero counts and no notes."""
    world = await new_world()

    result = await ReportStage().run(ctx_for(world))

    assert result.status is StageStatus.OK
    assert result.counts == {
        "near_misses": 0,
        "stale_summaries": 0,
        "disputed_facts": 0,
        "pending_proposals": 0,
    }
    assert result.notes == []


async def test_a_failing_store_read_returns_partial_without_raising() -> None:
    """A store failure is recorded on the result instead of aborting the run."""
    world = await new_world()

    result = await ReportStage().run(ctx_for(world, store=FailingFindStore()))

    assert result.status is StageStatus.PARTIAL
    assert result.errors
    assert result.stage == "report"


async def test_counts_rejected_pairs_of_deduplication_as_near_misses() -> None:
    """Pairs the deduplication stage judged distinct are near misses."""
    world = await new_world()
    earlier = [
        StageResult(
            stage="deduplication",
            status=StageStatus.OK,
            counts={"pairs_considered": 3, "pairs_asked": 2, "pairs_rejected": 2},
        )
    ]

    result = await ReportStage().run(ctx_for(world, results=earlier))

    assert result.counts["near_misses"] == 2


async def test_lists_stages_that_failed_or_stopped_partway() -> None:
    """The notes name every earlier stage that failed or ended partial."""
    world = await new_world()
    earlier = [
        StageResult(stage="deduplication", status=StageStatus.FAILED),
        StageResult(stage="feedback_weights", status=StageStatus.PARTIAL),
        StageResult(stage="ingest_cleanup", status=StageStatus.OK),
    ]

    result = await ReportStage().run(ctx_for(world, results=earlier))

    assert result.notes == [
        "failed stages: deduplication",
        "partial stages: feedback_weights",
    ]
