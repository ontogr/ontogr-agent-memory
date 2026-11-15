"""Deduplication behavior on a live graph tenant."""

from collections.abc import Sequence

import pytest

from ontogr_mem.data_models.entities import EntityStatus
from ontogr_mem.resolution.config import Budget
from ontogr_mem.resolution.resolver_store import ResolverStore
from ontogr_mem.vectorstore import (
    EmbeddingFailedError,
    EmbeddingPurpose,
    PermanentEmbeddingError,
    VectorKind,
)
from tests.contracts.resolver_store import entity
from tests.integration.resolution.resolver_helpers import make
from tests.support.embedder import MockEmbedder
from tests.support.graph_backend import GraphBackend
from tests.unit.resolution.helpers import oracle_models


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
    """Two active entities with one strict key merge on the rule alone."""
    store = await _duplicate_store(graph_backend)
    resolver = await make(store)

    report = await resolver.deduplicate()

    assert report.pairs_rule_same == 1
    assert report.pairs_merged == 1
    statuses = {e.entity_id: e.status for e in await store.get_entities(["e1", "e2"])}
    assert sorted(statuses.values()) == [EntityStatus.ACTIVE, EntityStatus.MERGED]


async def test_plan_only_reports_the_plan_without_writing(graph_backend: GraphBackend):
    """A plan-only run leaves both entities active."""
    store = await _duplicate_store(graph_backend)
    resolver = await make(store)

    report = await resolver.deduplicate(plan_only=True)

    assert report.plan
    statuses = {e.entity_id: e.status for e in await store.get_entities(["e1", "e2"])}
    assert set(statuses.values()) == {EntityStatus.ACTIVE}


async def test_selected_entities_are_marked_checked(graph_backend: GraphBackend):
    """A completed run records the revision it checked."""
    store = await _duplicate_store(graph_backend)
    resolver = await make(store)

    await resolver.deduplicate(budget=Budget(sweep=10))

    survivor = (await store.get_entities(["e1"]))[0]
    assert survivor.last_checked_version == survivor.revision


async def test_embedding_failure_leaves_ann_candidates_for_retry(
    graph_backend: GraphBackend,
):
    """A failed query embedding leaves entities dirty for a complete retry."""

    class FailFirstQueryEmbedder(MockEmbedder):
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
            return [[1.0, 0.0, 0.0] for _ in texts]

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
