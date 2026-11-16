"""Deduplication behavior on a live graph tenant."""

import asyncio
from collections.abc import Sequence

import pytest

from ontogr_mem.data_models.entities import Alias, AliasSource, EntityStatus
from ontogr_mem.graphstore import QuerySpec, Rows
from ontogr_mem.graphstore.protocol import ScopedRunner
from ontogr_mem.graphstore.resolver_store import GraphResolverStore
from ontogr_mem.resolution.config import Budget
from ontogr_mem.resolution.errors import ConcurrentEntityWriteError
from ontogr_mem.resolution.keys import entity_text
from ontogr_mem.resolution.resolver_store import ResolverStore
from ontogr_mem.vectorstore import (
    EmbeddingFailedError,
    EmbeddingPurpose,
    PermanentEmbeddingError,
    VectorKind,
)
from tests.contracts.resolver_store import alias, entity, mention
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


async def test_alias_only_batch_reopens_checked_owner(graph_backend: GraphBackend):
    """Adding a matching alias makes its checked owner eligible again."""
    store = (await graph_backend.open_stores()).resolver_store
    await store.write_entity_batch(
        [
            entity("e1", name="Amber Maritime Consortium"),
            entity("e2", name="Violet Quantum Laboratory"),
        ],
        [],
    )
    await store.mark_deduplication_checked(await store.get_entities(["e1", "e2"]))

    await store.write_entity_batch([], [alias("e1", "Violet Quantum Laboratory")])

    owner = (await store.get_entities(["e1"]))[0]
    counterpart = (await store.get_entities(["e2"]))[0]
    report = await (await make(store)).deduplicate()

    assert owner.revision > owner.last_checked_version
    assert counterpart.last_checked_version == counterpart.revision
    assert report.selected == 1
    assert report.pairs_merged == 1


async def test_checkpoint_cannot_split_entity_and_alias_batch(
    graph_backend: GraphBackend,
):
    """A checkpoint cannot land between an entity update and its new alias."""
    handle = await graph_backend.open_tenant()
    runner = graph_backend.store.runner(handle)
    base_store = GraphResolverStore(runner, tenant_keyed=graph_backend.tenant_keyed)
    await base_store.write_entity_batch(
        [
            entity("e1", name="Amber Maritime Consortium"),
            entity("e2", name="Violet Quantum Laboratory"),
        ],
        [],
    )
    await base_store.mark_deduplication_checked(
        await base_store.get_entities(["e1", "e2"])
    )

    class PauseAfterWriteRunner:
        """Pause after the entity batch is committed to the graph."""

        def __init__(self, wrapped: ScopedRunner) -> None:
            self._wrapped = wrapped
            self.scope = wrapped.scope
            self.write_finished = asyncio.Event()
            self.resume = asyncio.Event()

        async def run(self, spec: QuerySpec) -> Rows:
            result = await self._wrapped.run(spec)
            if spec.tag == "resolver_entities_write":
                self.write_finished.set()
                await self.resume.wait()
            return result

    pausing_runner = PauseAfterWriteRunner(runner)
    store = GraphResolverStore(pausing_runner, tenant_keyed=graph_backend.tenant_keyed)
    resolver = await make(store)
    updated_owner = (await store.get_entities(["e1"]))[0].model_copy(
        update={"description": "Updated description"}
    )
    batch = asyncio.create_task(
        store.write_entity_batch(
            [updated_owner], [alias("e1", "Violet Quantum Laboratory")]
        )
    )

    try:
        await asyncio.wait_for(pausing_runner.write_finished.wait(), timeout=10)
        report = await resolver.deduplicate()
    finally:
        pausing_runner.resume.set()
        await batch

    assert report.selected == 1
    assert report.pairs_merged == 1


async def test_alias_writes_retry_out_of_order_revision_allocations(
    graph_backend: GraphBackend,
):
    """A late alias writer retries instead of lowering its owner's revision."""
    handle = await graph_backend.open_tenant()
    runner = graph_backend.store.runner(handle)
    base_store = GraphResolverStore(runner, tenant_keyed=graph_backend.tenant_keyed)
    owner = entity("e1", name="Amber Maritime").model_copy(
        update={"attributes": {"revision": 1}}
    )
    await base_store.write_entity_batch([owner], [])
    await base_store.mark_deduplication_checked(await base_store.get_entities(["e1"]))

    class PauseFirstRevisionRunner:
        """Hold one alias writer after it reserves a revision."""

        def __init__(self, wrapped: ScopedRunner) -> None:
            self._wrapped = wrapped
            self.scope = wrapped.scope
            self.revision_reserved = asyncio.Event()
            self.resume = asyncio.Event()
            self._paused = False

        async def run(self, spec: QuerySpec) -> Rows:
            result = await self._wrapped.run(spec)
            if spec.tag == "resolver_revision" and not self._paused:
                self._paused = True
                self.revision_reserved.set()
                await self.resume.wait()
            return result

    pausing_runner = PauseFirstRevisionRunner(runner)
    delayed_store = GraphResolverStore(
        pausing_runner, tenant_keyed=graph_backend.tenant_keyed
    )
    delayed_write = asyncio.create_task(
        delayed_store.write_entity_batch([], [alias("e1", "Delayed alias")])
    )

    try:
        await asyncio.wait_for(pausing_runner.revision_reserved.wait(), timeout=5)
        await base_store.write_entity_batch([], [alias("e1", "Concurrent alias")])
        await base_store.mark_deduplication_checked(
            await base_store.get_entities(["e1"])
        )
    finally:
        pausing_runner.resume.set()
        await delayed_write

    stored_owner = (await base_store.get_entities(["e1"]))[0]
    stored_aliases = await base_store.aliases_of("e1")

    assert stored_owner.revision > stored_owner.last_checked_version
    assert stored_owner.attributes["revision"] == 1
    assert {stored_alias.text for stored_alias in stored_aliases} == {
        "Concurrent alias",
        "Delayed alias",
    }


async def test_concurrent_alias_id_claims_keep_one_checked_owner_dirty(
    graph_backend: GraphBackend,
):
    """Only one owner can claim an alias ID in concurrent batches."""
    handle = await graph_backend.open_tenant()
    runner = graph_backend.store.runner(handle)
    base_store = GraphResolverStore(runner, tenant_keyed=graph_backend.tenant_keyed)
    await base_store.write_entity_batch(
        [entity("e1", name="Amber Maritime"), entity("e2", name="Violet Quantum")],
        [],
    )
    await base_store.mark_deduplication_checked(
        await base_store.get_entities(["e1", "e2"])
    )

    class AliasWriteBarrier:
        """Start two alias batches together after they reserve revisions."""

        def __init__(self) -> None:
            self.arrived = 0
            self.both_arrived = asyncio.Event()

    barrier = AliasWriteBarrier()

    class BarrierRunner:
        """Wait until both alias batches reach their graph write."""

        def __init__(self, wrapped: ScopedRunner) -> None:
            self._wrapped = wrapped
            self.scope = wrapped.scope

        async def run(self, spec: QuerySpec) -> Rows:
            if spec.tag == "resolver_entities_write":
                barrier.arrived += 1
                if barrier.arrived == 2:
                    barrier.both_arrived.set()
                await barrier.both_arrived.wait()
            return await self._wrapped.run(spec)

    first_store = GraphResolverStore(
        BarrierRunner(runner), tenant_keyed=graph_backend.tenant_keyed
    )
    second_store = GraphResolverStore(
        BarrierRunner(runner), tenant_keyed=graph_backend.tenant_keyed
    )
    results = await asyncio.gather(
        first_store.write_entity_batch([], [alias("e1", "Shared alias", "same-id")]),
        second_store.write_entity_batch([], [alias("e2", "Shared alias", "same-id")]),
        return_exceptions=True,
    )

    assert (
        sum(isinstance(result, ConcurrentEntityWriteError) for result in results) == 1
    )
    stored_aliases = await base_store.all_aliases()
    assert len(stored_aliases) == 1
    owner_id = stored_aliases[0].entity_id
    assert owner_id in {"e1", "e2"}
    assert [row.alias_id for row in await base_store.aliases_of(owner_id)] == [
        "same-id"
    ]
    assert await base_store.aliases_of("e2" if owner_id == "e1" else "e1") == []
    owners = {row.entity_id: row for row in await base_store.get_entities(["e1", "e2"])}
    assert owners[owner_id].revision > owners[owner_id].last_checked_version
    other_id = "e2" if owner_id == "e1" else "e1"
    assert owners[other_id].revision == owners[other_id].last_checked_version


async def test_stale_mixed_batch_leaves_no_entity_or_alias_rows(
    graph_backend: GraphBackend,
):
    """A conflicted batch removes its placeholders before retrying."""
    handle = await graph_backend.open_tenant()
    runner = graph_backend.store.runner(handle)
    base_store = GraphResolverStore(runner, tenant_keyed=graph_backend.tenant_keyed)
    await base_store.write_entity_batch([entity("e1", name="Amber Maritime")], [])

    class PauseConflictRunner:
        """Pause one batch before its snapshot-conflict retry."""

        def __init__(self, wrapped: ScopedRunner) -> None:
            self._wrapped = wrapped
            self.scope = wrapped.scope
            self.revision_reserved = asyncio.Event()
            self.conflict_found = asyncio.Event()
            self.resume_write = asyncio.Event()
            self.resume_retry = asyncio.Event()
            self._paused_revision = False

        async def run(self, spec: QuerySpec) -> Rows:
            result = await self._wrapped.run(spec)
            if spec.tag == "resolver_revision" and not self._paused_revision:
                self._paused_revision = True
                self.revision_reserved.set()
                await self.resume_write.wait()
            elif spec.tag == "resolver_entities_write" and not result.scalar(
                "applied", True
            ):
                self.conflict_found.set()
                await self.resume_retry.wait()
            return result

    pausing_runner = PauseConflictRunner(runner)
    delayed_store = GraphResolverStore(
        pausing_runner, tenant_keyed=graph_backend.tenant_keyed
    )
    delayed_batch = asyncio.create_task(
        delayed_store.write_entity_batch(
            [entity("e2", name="Violet Quantum")],
            [alias("e1", "Delayed alias")],
        )
    )

    try:
        await asyncio.wait_for(pausing_runner.revision_reserved.wait(), timeout=5)
        current = (await base_store.get_entities(["e1"]))[0]
        await base_store.save_entity(
            current.model_copy(update={"description": "Concurrent update"})
        )
        pausing_runner.resume_write.set()
        await asyncio.wait_for(pausing_runner.conflict_found.wait(), timeout=5)

        assert await base_store.get_entities(["e2"]) == []
        assert await base_store.aliases_of("e1") == []
    finally:
        pausing_runner.resume_write.set()
        pausing_runner.resume_retry.set()
        await delayed_batch

    assert (await base_store.get_entities(["e2"]))[0].canonical_name == "Violet Quantum"
    assert [item.text for item in await base_store.aliases_of("e1")] == [
        "Delayed alias"
    ]


async def test_stale_no_bump_save_cannot_restore_a_checkpoint(
    graph_backend: GraphBackend,
):
    """A stale full-row save cannot roll back alias invalidation or checking."""
    store = (await graph_backend.open_stores()).resolver_store
    await store.write_entity_batch([entity("e1", name="Amber Maritime")], [])
    stale = (await store.get_entities(["e1"]))[0]
    await store.mark_deduplication_checked([stale])
    stale = (await store.get_entities(["e1"]))[0]

    await store.write_entity_batch([], [alias("e1", "New owner alias")])
    invalidated = (await store.get_entities(["e1"]))[0]
    await store.mark_deduplication_checked([invalidated])
    checked = (await store.get_entities(["e1"]))[0]

    with pytest.raises(ConcurrentEntityWriteError):
        await store.save_entity(stale, bump_revision=False)

    after_stale_save = (await store.get_entities(["e1"]))[0]
    assert after_stale_save.revision == checked.revision
    assert after_stale_save.last_checked_version == checked.last_checked_version


async def test_pass_continues_when_a_merge_loses_a_concurrent_write(
    graph_backend: GraphBackend,
):
    """A merge that loses a write race is counted as an error and retried later."""
    handle = await graph_backend.open_tenant()
    runner = graph_backend.store.runner(handle)
    base_store = GraphResolverStore(runner, tenant_keyed=graph_backend.tenant_keyed)
    await base_store.write_entity_batch(
        [
            entity("e1", name="Acme"),
            entity("e2", name="acme").model_copy(update={"strict_key": "acme"}),
        ],
        [],
    )

    class LateAliasOnMergeRunner:
        """Write a competing alias to each entity before the first merge applies."""

        def __init__(self, wrapped: ScopedRunner) -> None:
            self._wrapped = wrapped
            self.scope = wrapped.scope
            self._raced = False

        async def run(self, spec: QuerySpec) -> Rows:
            if spec.tag == "resolver_merge_apply" and not self._raced:
                self._raced = True
                await base_store.write_entity_batch([], [alias("e1", "Late alias one")])
                await base_store.write_entity_batch([], [alias("e2", "Late alias two")])
            return await self._wrapped.run(spec)

    racing_store = GraphResolverStore(
        LateAliasOnMergeRunner(runner), tenant_keyed=graph_backend.tenant_keyed
    )
    resolver = await make(racing_store)

    report = await resolver.deduplicate()

    assert report.pairs_merged == 0
    assert report.pairs_rejected == 0
    assert report.errors == 1
    stored = await base_store.get_entities(["e1", "e2"])
    assert all(item.status is EntityStatus.ACTIVE for item in stored)
    assert any(item.revision > item.last_checked_version for item in stored)

    second = await resolver.deduplicate()

    assert second.pairs_merged == 1


async def test_mention_recount_retries_after_a_concurrent_alias_write(
    graph_backend: GraphBackend,
):
    """A mention write keeps its count and a newer revision after a write race."""
    handle = await graph_backend.open_tenant()
    runner = graph_backend.store.runner(handle)
    base_store = GraphResolverStore(runner, tenant_keyed=graph_backend.tenant_keyed)
    await base_store.write_entity_batch([entity("e1", name="Amber Maritime")], [])
    await base_store.mark_deduplication_checked(await base_store.get_entities(["e1"]))

    class LateAliasAfterCountRunner:
        """Write a competing alias after the mention recount reads the entity."""

        def __init__(self, wrapped: ScopedRunner) -> None:
            self._wrapped = wrapped
            self.scope = wrapped.scope
            self._counted = False
            self._raced = False

        async def run(self, spec: QuerySpec) -> Rows:
            result = await self._wrapped.run(spec)
            if spec.tag == "resolver_mentions_count":
                self._counted = True
            elif spec.tag == "resolver_get_entities" and self._counted:
                if not self._raced:
                    self._raced = True
                    await base_store.write_entity_batch([], [alias("e1", "Late alias")])
            return result

    racing_store = GraphResolverStore(
        LateAliasAfterCountRunner(runner), tenant_keyed=graph_backend.tenant_keyed
    )

    await racing_store.write_mentions("ep1", [mention("e1", "m1")])

    stored = (await base_store.get_entities(["e1"]))[0]
    assert stored.mention_count == 1
    assert stored.revision > stored.last_checked_version
    assert [item.text for item in await base_store.aliases_of("e1")] == ["Late alias"]


async def test_alias_batch_cannot_reassign_an_existing_alias(
    graph_backend: GraphBackend,
):
    """An alias batch cannot change ownership without checking both owners."""
    store = (await graph_backend.open_stores()).resolver_store
    await store.write_entity_batch(
        [
            entity("e1", name="Amber Maritime"),
            entity("e2", name="Violet Quantum"),
        ],
        [alias("e1", "Shared display name")],
    )
    selected = await store.get_entities(["e1", "e2"])
    await store.mark_deduplication_checked(selected)
    checked = {item.entity_id: item for item in await store.get_entities(["e1", "e2"])}
    moved = alias("e1", "Shared display name").model_copy(update={"entity_id": "e3"})

    with pytest.raises(ConcurrentEntityWriteError):
        await store.write_entity_batch([entity("e3", name="Silver Systems")], [moved])

    current = {item.entity_id: item for item in await store.get_entities(["e1", "e2"])}
    assert current["e1"].revision == checked["e1"].revision
    assert current["e1"].last_checked_version == checked["e1"].last_checked_version
    assert [item.entity_id for item in await store.aliases_of("e1")] == ["e1"]
    assert await store.get_entities(["e3"]) == []
    assert await store.aliases_of("e3") == []


async def test_alias_batch_rejects_a_conflicting_relationship_owner(
    graph_backend: GraphBackend,
):
    """An alias link to another owner blocks an ordinary alias write."""
    handle = await graph_backend.open_tenant()
    runner = graph_backend.store.runner(handle)
    store = (await graph_backend.stores_for(handle.tenant)).resolver_store
    await store.write_entity_batch(
        [
            entity("e1", name="Amber Maritime"),
            entity("e2", name="Violet Quantum"),
        ],
        [alias("e1", "Shared display name")],
    )
    selected = await store.get_entities(["e1", "e2"])
    await store.mark_deduplication_checked(selected)
    checked = (await store.get_entities(["e1"]))[0]
    scope = runner.scope
    await runner.run(
        QuerySpec(
            f"MATCH (a:Alias {scope.pattern('id: $alias_id')}) "
            f"OPTIONAL MATCH (a)-[old:ALIAS_OF {scope.edge()}]->"
            f"(:Entity {scope.pattern()}) DELETE old "
            f"WITH a MATCH (e:Entity {scope.pattern('id: $entity_id')}) "
            f"MERGE (a)-[r:ALIAS_OF {scope.edge()}]->(e) "
            f"{scope.stamp_assignment('r')}",
            {"alias_id": "a-e1-Shared display name", "entity_id": "e2"},
            idempotent=True,
            tag="test_alias_owner_mismatch",
        )
    )

    with pytest.raises(ConcurrentEntityWriteError):
        await store.write_entity_batch([], [alias("e1", "Shared display name")])

    current = (await store.get_entities(["e1"]))[0]
    assert current.revision == checked.revision
    assert current.last_checked_version == checked.last_checked_version
    assert [item.entity_id for item in await store.aliases_of("e1")] == ["e1"]


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
    await store.mark_deduplication_checked(await store.get_entities(["e1", "e2"]))
    checked = {
        stored.entity_id: stored for stored in await store.get_entities(["e1", "e2"])
    }
    alias_entity = selected if alias_on_selected else counterpart
    alias_entity_id = alias_entity.entity_id
    unchanged_entity_id = (
        counterpart.entity_id if alias_on_selected else selected.entity_id
    )
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
    after_alias = {
        stored.entity_id: stored for stored in await store.get_entities(["e1", "e2"])
    }
    embedder = RecordingEmbedder()
    resolver = await make(store, embedder=embedder)
    embedder.embedded.clear()

    report = await resolver.deduplicate()

    assert report.selected == 1
    assert after_alias[alias_entity_id].revision > checked[alias_entity_id].revision
    assert (
        after_alias[unchanged_entity_id].revision
        == checked[unchanged_entity_id].revision
    )
    assert (
        after_alias[unchanged_entity_id].last_checked_version
        == checked[unchanged_entity_id].revision
    )
    assert report.pairs_rule_same == 1
    assert report.pairs_merged == 1
    assert embedder.embedded == [entity_text(checked[alias_entity_id])]
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
