"""Deduplication behavior on a live graph tenant."""

import asyncio
from collections.abc import Sequence

import pytest

from ontogr_mem.data_models.entities import Alias, AliasSource, EntityStatus
from ontogr_mem.resolution.config import Budget
from ontogr_mem.resolution.keys import entity_text
from ontogr_mem.resolution.resolver_store import ResolverStore
from ontogr_mem.vectorstore import (
    EmbeddingFailedError,
    EmbeddingPurpose,
    PermanentEmbeddingError,
    VectorKind,
)
from tests.contracts.resolver_store import alias, entity
from tests.integration.resolution.resolver_helpers import make
from tests.support.embedder import MockEmbedder, RecordingEmbedder
from tests.support.graph_backend import GraphBackend
from tests.unit.resolution.helpers import oracle_models


class SameVectorEmbedder(MockEmbedder):
    """Give every entity the same vector so ANN finds their pair."""

    async def embed_batch(
        self, texts: Sequence[str], *, purpose: EmbeddingPurpose
    ) -> list[list[float]]:
        """Return identical vectors for the requested texts."""
        return [[1.0, 0.0, 0.0] for _ in texts]


async def _duplicate_store(graph_backend: GraphBackend) -> ResolverStore:
    store = (await graph_backend.open_stores()).resolver_store
    await store.write_entity_batch(
        [
            entity("e1", name="Acme"),
            entity("e2", name="acme").model_copy(update={"strict_key": "acme"}),
        ],
        [],
    )
    return store


async def test_rule_same_pairs_are_merged_without_a_model(graph_backend: GraphBackend):
    """A merge leaves its survivor dirty until a later pass checks it."""
    store = await _duplicate_store(graph_backend)
    resolver = await make(store)

    report = await resolver.deduplicate()

    assert report.pairs_rule_same == 1
    assert report.pairs_merged == 1
    statuses = {e.entity_id: e.status for e in await store.get_entities(["e1", "e2"])}
    assert sorted(statuses.values()) == [EntityStatus.ACTIVE, EntityStatus.MERGED]
    survivor_id = next(
        entity_id
        for entity_id, status in statuses.items()
        if status is EntityStatus.ACTIVE
    )
    survivor = (await store.get_entities([survivor_id]))[0]
    assert survivor.last_checked_version == 0
    assert survivor.revision > survivor.last_checked_version

    settled = await resolver.deduplicate()

    assert settled.selected == 1
    survivor = (await store.get_entities([survivor_id]))[0]
    assert survivor.last_checked_version == survivor.revision
    assert (await resolver.deduplicate()).selected == 0


async def test_plan_only_reports_the_plan_without_writing(graph_backend: GraphBackend):
    """Repeated plans leave entity, alias, checkpoint and verdict state unchanged."""
    store = (await graph_backend.open_stores()).resolver_store
    await store.write_entity_batch(
        [entity("e1", name="Acme"), entity("e2", name="acme")],
        [alias("e1", "Acme Group")],
    )
    before_entities = await store.all_entities()
    before_aliases = await store.all_aliases()
    before_verdict = await store.live_verdict("e1", "e2")
    resolver = await make(store)

    reports = [await resolver.deduplicate(plan_only=True) for _ in range(2)]

    assert reports[0].plan
    assert reports[1].plan == reports[0].plan
    assert await store.all_entities() == before_entities
    assert await store.all_aliases() == before_aliases
    assert await store.live_verdict("e1", "e2") == before_verdict


async def test_selected_entities_are_marked_checked(graph_backend: GraphBackend):
    """A completed run records the revision it checked."""
    store = (await graph_backend.open_stores()).resolver_store
    await store.write_entity_batch([entity("e1", name="Acme")], [])
    resolver = await make(store)

    await resolver.deduplicate()

    survivor = (await store.get_entities(["e1"]))[0]
    assert survivor.last_checked_version == survivor.revision


async def test_completed_default_pass_does_not_reselect_checked_entities(
    graph_backend: GraphBackend,
):
    """A stable entity is not selected or embedded again on the next pass."""
    store = (await graph_backend.open_stores()).resolver_store
    await store.write_entity_batch([entity("e1", name="Amber Maritime")], [])
    embedder = RecordingEmbedder()
    resolver = await make(store, embedder=embedder)
    embedder.embedded.clear()

    first = await resolver.deduplicate()
    selected = (await store.get_entities(["e1"]))[0]
    second = await resolver.deduplicate()

    assert first.selected == 1
    assert selected.last_checked_version == selected.revision
    assert second.selected == 0
    assert embedder.embedded == [entity_text(selected)]


@pytest.mark.parametrize(
    "alias_on_selected",
    [True, False],
    ids=["selected-alias", "counterpart-alias"],
)
async def test_alias_matches_in_either_direction_without_stored_vectors(
    graph_backend: GraphBackend, alias_on_selected: bool
):
    """Blocking and rule classification find aliases whichever side owns one."""
    store = (await graph_backend.open_stores()).resolver_store
    selected = entity("e1", name="Amber Maritime Consortium")
    counterpart = entity("e2", name="Violet Quantum Laboratory")
    selected = (await store.write_entity_batch([selected], []))[0]
    counterpart = (await store.write_entity_batch([counterpart], []))[0]
    counterpart = await store.save_entity(
        counterpart.model_copy(update={"last_checked_version": counterpart.revision}),
        bump_revision=False,
    )
    alias_entity = selected if alias_on_selected else counterpart
    alias_text = (
        counterpart.canonical_name if alias_on_selected else selected.canonical_name
    )
    await store.write_entity_batch(
        [],
        [
            Alias(
                alias_id="alias-either-direction",
                entity_id=alias_entity.entity_id,
                text=alias_text,
                strict_key=alias_text.casefold(),
                loose_key=alias_text.casefold(),
                source=AliasSource.USER,
            )
        ],
    )
    embedder = RecordingEmbedder()
    resolver = await make(store, embedder=embedder)
    embedder.embedded.clear()

    report = await resolver.deduplicate()

    assert report.selected == 1
    assert selected.revision < counterpart.revision
    assert counterpart.last_checked_version == counterpart.revision
    assert report.pairs_rule_same == 1
    assert report.pairs_merged == 1
    assert embedder.embedded == [entity_text(selected)]
    assert await resolver.vectors.exists(VectorKind.NODE, ["e1", "e2"]) == set()


async def test_alias_query_uses_its_key_for_jaccard_without_stored_vectors(
    graph_backend: GraphBackend,
):
    """A selected alias supplies the Jaccard score for its discovered pair."""
    store = (await graph_backend.open_stores()).resolver_store
    selected = entity("e1", name="Amber Maritime Consortium")
    candidate = entity("e2", name="Violet Quantum Laboratory")
    selected = (await store.write_entity_batch([selected], []))[0]
    candidate = (await store.write_entity_batch([candidate], []))[0]
    candidate = await store.save_entity(
        candidate.model_copy(update={"last_checked_version": candidate.revision}),
        bump_revision=False,
    )
    await store.write_entity_batch(
        [],
        [
            Alias(
                alias_id="alias-jaccard-query",
                entity_id=selected.entity_id,
                text="Violet Quantum Laboratory!",
                strict_key="violet quantum laboratory!",
                loose_key="violet quantum laboratory",
                source=AliasSource.USER,
            )
        ],
    )
    models, scripted = oracle_models(lambda _a, _b: True)
    embedder = RecordingEmbedder()
    resolver = await make(store, models=models, embedder=embedder)
    embedder.embedded.clear()

    report = await resolver.deduplicate()

    assert report.selected == 1
    assert selected.revision < candidate.revision
    assert candidate.last_checked_version == candidate.revision
    assert report.pairs_considered == 1
    assert report.pairs_asked == 1
    assert report.pairs_merged == 1
    assert "jaccard=1.00" in scripted.calls[0].prompt
    assert "cosine=" not in scripted.calls[0].prompt
    assert embedder.embedded == [entity_text(selected)]
    assert await resolver.vectors.exists(VectorKind.NODE, ["e1", "e2"]) == set()


async def test_embedding_failure_leaves_ann_candidates_for_retry(
    graph_backend: GraphBackend,
):
    """A failed query embedding leaves entities dirty for a complete retry."""

    class FailFirstQueryEmbedder(SameVectorEmbedder):
        """Fail one query batch, then return vectors that link all entities."""

        def __init__(self) -> None:
            super().__init__()
            self.fail_query = False

        async def embed_batch(
            self, texts: Sequence[str], *, purpose: EmbeddingPurpose
        ) -> list[list[float]]:
            if purpose == "query" and self.fail_query:
                self.fail_query = False
                raise PermanentEmbeddingError("temporary failure")
            return await super().embed_batch(texts, purpose=purpose)

    store = (await graph_backend.open_stores()).resolver_store
    await store.write_entity_batch(
        [
            entity("e1", name="Zephyr Quarry"),
            entity("e2", name="Mariner Orchard"),
        ],
        [],
    )
    models, _ = oracle_models(lambda _a, _b: False)
    embedder = FailFirstQueryEmbedder()
    resolver = await make(store, models=models, embedder=embedder)
    entities = await store.get_entities(["e1", "e2"])
    for item in entities:
        await resolver.embed(item)

    embedder.fail_query = True
    before = {item.entity_id: item.last_checked_version for item in entities}
    with pytest.raises(EmbeddingFailedError):
        await resolver.deduplicate()

    after_failure = await store.get_entities(["e1", "e2"])
    assert {
        item.entity_id: item.last_checked_version for item in after_failure
    } == before
    assert await resolver.store.live_verdict("e1", "e2") is None

    report = await resolver.deduplicate()

    assert report.pairs_considered == 1
    assert report.pairs_asked == 1
    assert report.pairs_rejected == 1
    recovered = await store.get_entities(["e1", "e2"])
    assert all(item.last_checked_version == item.revision for item in recovered)
    assert await resolver.vectors.exists(VectorKind.NODE, ["e1", "e2"]) == {
        "e1",
        "e2",
    }


async def test_budget_deferred_pair_stays_dirty_until_a_later_verification(
    graph_backend: GraphBackend,
):
    """Entities deferred by the verifier budget remain eligible for a retry."""
    store = (await graph_backend.open_stores()).resolver_store
    await store.write_entity_batch(
        [entity("e1", name="Zephyr Quarry"), entity("e2", name="Mariner Orchard")],
        [],
    )
    models, _ = oracle_models(lambda _a, _b: False)
    resolver = await make(store, models=models, embedder=SameVectorEmbedder())
    for item in await store.get_entities(["e1", "e2"]):
        await resolver.embed(item)

    deferred = await resolver.deduplicate(budget=Budget(max_llm_pairs=0))

    assert deferred.pairs_considered == 1
    assert deferred.pairs_asked == 0
    after_deferred = await store.get_entities(["e1", "e2"])
    assert [item.last_checked_version for item in after_deferred] == [0, 0]
    assert await store.live_verdict("e1", "e2") is None

    retried = await resolver.deduplicate(budget=Budget(max_llm_pairs=1))

    assert retried.pairs_asked == 1
    assert retried.pairs_rejected == 1
    after_retry = await store.get_entities(["e1", "e2"])
    assert all(item.last_checked_version == item.revision for item in after_retry)


async def test_concurrent_entity_update_is_not_checkpointed_from_an_old_snapshot(
    graph_backend: GraphBackend,
):
    """A write during candidate discovery stays dirty and keeps its new data."""

    class PausingQueryEmbedder(SameVectorEmbedder):
        """Hold query embedding until the test updates the selected entity."""

        def __init__(self) -> None:
            super().__init__()
            self.pause_query = False
            self.query_started = asyncio.Event()
            self.resume_query = asyncio.Event()

        async def embed_batch(
            self, texts: Sequence[str], *, purpose: EmbeddingPurpose
        ) -> list[list[float]]:
            if purpose == "query" and self.pause_query:
                self.query_started.set()
                await self.resume_query.wait()
            return await super().embed_batch(texts, purpose=purpose)

    store = (await graph_backend.open_stores()).resolver_store
    await store.write_entity_batch([entity("e1", name="Original Name")], [])
    embedder = PausingQueryEmbedder()
    resolver = await make(store, embedder=embedder)
    selected = (await store.get_entities(["e1"]))[0]
    embedder.pause_query = True
    run = asyncio.create_task(resolver.deduplicate())

    try:
        await asyncio.wait_for(embedder.query_started.wait(), timeout=5)
        concurrent = selected.model_copy(
            update={
                "canonical_name": "Updated During Discovery",
                "strict_key": "updated during discovery",
                "loose_key": "updated during discovery",
            }
        )
        written = (await store.write_entity_batch([concurrent], []))[0]
    finally:
        embedder.resume_query.set()
        await run

    current = (await store.get_entities(["e1"]))[0]
    assert current.canonical_name == "Updated During Discovery"
    assert current.revision == written.revision
    assert current.last_checked_version == selected.last_checked_version


async def test_a_blocked_pair_is_judged_on_its_jaccard_and_not_on_a_cosine(
    graph_backend: GraphBackend,
):
    """A pair found by blocking clears the Jaccard bar on one vote."""
    store = (await graph_backend.open_stores()).resolver_store
    # The two names share a Jaccard of 0.94: above the Jaccard bar of 0.92 but
    # below the cosine bar of 0.96, so only the Jaccard floor can settle them.
    await store.write_entity_batch(
        [
            entity("e1", name="Beta Solutions Group"),
            entity("e2", name="Beta Solutions Groups"),
        ],
        [],
    )
    models, scripted = oracle_models(lambda a, b: True)
    resolver = await make(store, models=models)

    report = await resolver.deduplicate()

    assert len(scripted.calls) == 1
    assert "jaccard=" in scripted.calls[0].prompt
    assert "cosine=" not in scripted.calls[0].prompt
    assert report.pairs_merged == 1
