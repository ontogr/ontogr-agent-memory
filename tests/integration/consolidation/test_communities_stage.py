"""The communities stage on live graph tenants, with a scripted model.

Most tests run the stage through the runner, so the fingerprint watermark and the
partial status behave as they do in production. The content-hash reuse test calls
the stage directly, because a second runner pass is skipped by the gate before it
reaches the reuse path.
"""

from datetime import timedelta
from typing import Any
from uuid import uuid4

import pytest

from ontogr_mem.consolidation.archive import archive_item
from ontogr_mem.consolidation.config import CommunitySettings, ConsolidationSettings
from ontogr_mem.consolidation.models import (
    Bucket,
    Community,
    RunStatus,
    StageResult,
    StageStatus,
)
from ontogr_mem.consolidation.runner import ConsolidationRunner
from ontogr_mem.consolidation.stage import StageContext
from ontogr_mem.consolidation.stages.communities import CommunitiesStage
from ontogr_mem.consolidation.stages.ingest_cleanup import IngestCleanupStage
from ontogr_mem.data_models.facts import Precision, TimeBasis
from ontogr_mem.data_models.observation import Observation, ObservationKind
from ontogr_mem.llm import LLMClient, LLMGate
from ontogr_mem.llm.errors import LLMCallError
from tests.support.consolidation import ConsWorld, new_world
from tests.support.graph_backend import GraphBackend
from tests.support.llm import ScriptedLLM


def merged_llm() -> ScriptedLLM:
    """Return a model that answers every summary and description request."""
    return ScriptedLLM(
        answer=lambda _: {"summary": "Merged.", "description": "A group."}
    )


def _gate(llm: LLMClient) -> LLMGate:
    """Build a gate around a client."""
    return LLMGate(llm, max_input_tokens=10_000)


def _calls(result: StageResult) -> int:
    """Return the number of model calls a stage made."""
    return sum(bucket.calls for bucket in result.usage.buckets.values())


def runner_for(world: ConsWorld, llm: LLMClient | None) -> ConsolidationRunner:
    """Run the communities stage after the fatal cleanup stage."""
    return ConsolidationRunner(
        [IngestCleanupStage(), CommunitiesStage()],
        world.store,
        world.clock,
        world.settings,
        resolution=world.resolution,
        temporal=world.temporal,
        llm=_gate(llm) if llm is not None else None,
    )


def stage_context(world: ConsWorld, llm: LLMClient | None) -> StageContext:
    """Build a stage context over the world with the given model."""
    return StageContext(
        run_id="r1",
        clock=world.clock,
        store=world.store,
        settings=world.settings,
        resolution=world.resolution,
        temporal=world.temporal,
        llm=_gate(llm) if llm is not None else None,
    )


async def add_described_entity(world: ConsWorld, entity_id: str) -> None:
    """Save an active entity with one description of its own."""
    entity = await world.add_entity(entity_id, entity_id.upper())
    described = entity.model_copy(update={"descriptions": [f"{entity_id} text"]})
    await world.resolution.save_entity(described, bump_revision=False)


async def add_fact(world: ConsWorld, subject: str, obj: str) -> None:
    """Ingest one current fact from ``subject`` to ``obj`` into the temporal store."""
    now = world.clock()
    await world.ensure_entities([subject, obj])
    await world.temporal.ingest_observations(
        [
            Observation(
                obs_id=uuid4().hex,
                slot_key=f"{subject}->{obj}",
                subject_id=subject,
                object_id=obj,
                relation_type_id=1,
                kind=ObservationKind.ASSERTION,
                episode_id=f"ep-{uuid4().hex}",
                occurred_at=now - timedelta(days=1),
                ingested_at=now,
                valid_from=now - timedelta(days=1),
                valid_from_precision=Precision.DAY,
                valid_to=None,
                valid_to_precision=None,
                time_basis=TimeBasis.STATED,
                inverse=False,
            )
        ]
    )


async def add_triangle(world: ConsWorld, ids: str) -> None:
    """Add three described entities that are all connected to each other."""
    for entity_id in ids:
        await add_described_entity(world, entity_id)
    for subject, obj in [(0, 1), (1, 2), (0, 2)]:
        await add_fact(world, ids[subject], ids[obj])


async def two_clusters(world: ConsWorld) -> None:
    """Add two triangles with no fact between them."""
    await add_triangle(world, "abc")
    await add_triangle(world, "xyz")


async def stored_communities(world: ConsWorld) -> list[Community]:
    """Return every stored community, ordered by id."""
    return await world.store.list(Community)


async def test_two_clusters_become_summarised_communities_under_one_bucket(
    graph_backend: GraphBackend,
) -> None:
    """Each cluster is one community with a summary, and one bucket holds both."""
    world = await new_world(
        graph_backend,
    )
    await two_clusters(world)

    result = await runner_for(world, merged_llm()).run()

    communities = await stored_communities(world)
    assert sorted(c.member_ids for c in communities) == [
        ["a", "b", "c"],
        ["x", "y", "z"],
    ]
    assert all(c.summary == "Merged." for c in communities)
    assert all(c.description == "A group." for c in communities)
    assert result.stage("communities").status is StageStatus.OK
    buckets = await world.store.list(Bucket)
    assert len(buckets) == 1
    assert sorted(buckets[0].child_ids) == sorted(c.community_id for c in communities)


async def test_second_build_with_unchanged_content_makes_no_model_call(
    graph_backend: GraphBackend,
) -> None:
    """Rebuilding unchanged communities makes no model call and changes nothing."""
    llm = merged_llm()
    world = await new_world(
        graph_backend,
    )
    await two_clusters(world)
    await CommunitiesStage().run(stage_context(world, llm))
    communities = await stored_communities(world)
    buckets = await world.store.list(Bucket)
    calls_after_first_build = len(llm.requests)

    result = await CommunitiesStage().run(stage_context(world, llm))

    assert len(llm.requests) == calls_after_first_build
    assert _calls(result) == 0
    assert result.status is StageStatus.OK
    assert await stored_communities(world) == communities
    assert await world.store.list(Bucket) == buckets


async def test_unchanged_graph_is_skipped_by_the_fingerprint(
    graph_backend: GraphBackend,
) -> None:
    """A second run over an unchanged graph is skipped and calls no model."""
    llm = merged_llm()
    world = await new_world(
        graph_backend,
    )
    await two_clusters(world)
    runner = runner_for(world, llm)
    await runner.run()
    calls_after_first_run = len(llm.requests)

    result = await runner.run()

    assert result.stage("communities").status is StageStatus.SKIPPED
    assert len(llm.requests) == calls_after_first_run


async def test_without_a_model_communities_are_stored_unbuilt_with_a_note(
    graph_backend: GraphBackend,
) -> None:
    """With no model, communities are stored with empty summaries and a note."""
    world = await new_world(
        graph_backend,
    )
    await two_clusters(world)

    result = await runner_for(world, llm=None).run()

    stage = result.stage("communities")
    assert stage.status is StageStatus.PARTIAL
    assert stage.notes == ["no llm: communities stored without summaries"]
    communities = await stored_communities(world)
    assert sorted(c.member_ids for c in communities) == [
        ["a", "b", "c"],
        ["x", "y", "z"],
    ]
    assert all(c.summary == "" for c in communities)


async def test_more_communities_than_a_bucket_holds_build_a_second_level(
    graph_backend: GraphBackend,
) -> None:
    """Three communities with a bucket size of two make a two-level hierarchy."""
    settings = ConsolidationSettings(community=CommunitySettings(max_bucket_size=2))
    world = await new_world(graph_backend, settings)
    await add_triangle(world, "abc")
    await add_triangle(world, "def")
    await add_triangle(world, "ghi")

    await runner_for(world, merged_llm()).run()

    buckets = await world.store.list(Bucket)
    root = max(buckets, key=lambda b: b.level)
    assert len(buckets) == 3
    assert root.child_ids == ["bucket-1-0", "bucket-1-1"]
    assert all(b.summary == "Merged." for b in buckets)


@pytest.mark.parametrize(
    "outage",
    [LLMCallError("model is down"), "this is prose, not json"],
    ids=["call_fails", "malformed_answer"],
)
async def test_failed_build_leaves_the_community_unbuilt_and_retries(
    graph_backend: GraphBackend,
    outage: Any,
) -> None:
    """A failed or malformed model answer is PARTIAL, and the next run builds it."""
    world = await new_world(
        graph_backend,
    )
    await two_clusters(world)

    failed = await runner_for(world, ScriptedLLM(answer=lambda _: outage)).run()

    assert failed.status is RunStatus.PARTIAL
    assert failed.stage("communities").status is StageStatus.PARTIAL
    assert failed.stage("communities").errors
    assert all(c.summary == "" for c in await stored_communities(world))

    retried = await runner_for(world, merged_llm()).run()

    assert retried.stage("communities").status is StageStatus.OK
    assert all(c.summary == "Merged." for c in await stored_communities(world))


async def test_communities_whose_members_vanished_are_deleted(
    graph_backend: GraphBackend,
) -> None:
    """A stored community with no current counterpart is removed with its bucket."""
    world = await new_world(
        graph_backend,
    )
    await two_clusters(world)
    runner = runner_for(world, merged_llm())
    await runner.run()
    ctx = stage_context(world, merged_llm())
    for entity_id in "xyz":
        await archive_item(ctx, entity_id, "entity", "gone")

    result = await runner.run()

    assert [c.member_ids for c in await stored_communities(world)] == [["a", "b", "c"]]
    assert result.stage("communities").counts["removed"] == 1
    assert await world.store.list(Bucket) == []
