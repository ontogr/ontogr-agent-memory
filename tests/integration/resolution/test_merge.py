"""Merge, unmerge and reset-side behavior on a live graph tenant."""

import pytest

from ontogr_mem.data_models.entities import EntityStatus, MergeState
from ontogr_mem.resolution.errors import (
    EntityHasFactsError,
    SnapshotPurgedError,
    UnmergeBlockedError,
)
from ontogr_mem.resolution.ids import entity_id
from ontogr_mem.resolution.merge import pick_survivor, survivor_order
from ontogr_mem.resolution.resolver_store import ResolverStore
from ontogr_mem.vectorstore import VectorKind
from tests.contracts.resolver_store import entity
from tests.integration.resolution.resolver_helpers import ex, make
from tests.support.graph_backend import GraphBackend


async def _two_entities(
    graph_backend: GraphBackend,
) -> tuple[ResolverStore, object, str, str]:
    store = (await graph_backend.open_stores()).resolver_store
    resolver = await make(store)
    first = await resolver.resolve([ex("1", "Acme", "Company")])
    second = await resolver.resolve(
        [ex("1", "Acme Incorporated", "Company", episode_id="ep2")]
    )
    return store, resolver, first.entities[0].entity_id, second.entities[0].entity_id


def test_survivor_order_prefers_mentions_then_age_then_id():
    """The survivor choice is independent of input order."""
    bound = entity("e1", name="Zed").model_copy(
        update={"uri": "https://example.test/zed", "mention_count": 1}
    )
    busy = entity("e2", name="Acme").model_copy(update={"mention_count": 3})
    quiet = entity("e3", name="Beta")

    assert survivor_order(bound) < survivor_order(quiet)
    assert pick_survivor([quiet, busy, bound]) is bound


async def test_merge_then_unmerge_restores_the_absorbed_entity(
    graph_backend: GraphBackend,
):
    """An unmerge returns the absorbed entity, its aliases and its mentions."""
    store, resolver, survivor_id, absorbed_id = await _two_entities(graph_backend)

    record = await resolver.merge(survivor_id, absorbed_id, actor="user")
    absorbed = (await store.get_entities([absorbed_id]))[0]
    assert absorbed.status is EntityStatus.MERGED
    assert absorbed.merged_into == survivor_id
    assert await store.mentions_of(absorbed_id) == []

    restored = await resolver.unmerge(record.merge_id, actor="user")
    absorbed = (await store.get_entities([absorbed_id]))[0]
    assert restored.state is MergeState.REVERTED
    assert absorbed.status is EntityStatus.ACTIVE
    assert absorbed.merged_into is None
    assert {m.local_id for m in await store.mentions_of(absorbed_id)} == {"1"}


async def test_merge_after_a_user_unmerge_is_refused(graph_backend: GraphBackend):
    """The user DISTINCT verdict blocks an automated merge of the same pair."""
    store, resolver, survivor_id, absorbed_id = await _two_entities(graph_backend)
    record = await resolver.merge(survivor_id, absorbed_id, actor="user")
    await resolver.unmerge(record.merge_id, actor="user")

    with pytest.raises(UnmergeBlockedError):
        await resolver.merge(survivor_id, absorbed_id, actor="resolver")


async def test_purged_snapshot_blocks_unmerge(graph_backend: GraphBackend):
    """A privacy purge makes the merge irreversible."""
    store, resolver, survivor_id, absorbed_id = await _two_entities(graph_backend)
    record = await resolver.merge(survivor_id, absorbed_id, actor="user")

    assert await resolver.purge_snapshots(entity_id=survivor_id) == 1
    with pytest.raises(SnapshotPurgedError):
        await resolver.unmerge(record.merge_id, actor="user")


async def test_remove_episode_mentions_deletes_an_orphan_entity(
    graph_backend: GraphBackend,
):
    """The reset-side operation removes the mention and its entity."""
    store = (await graph_backend.open_stores()).resolver_store
    resolver = await make(store)
    await resolver.resolve([ex("1", "Acme", "Company")])

    deleted = await resolver.remove_episode_mentions("ep1")

    assert deleted == [entity_id("ep1", "1")]
    assert await store.active_entities() == []
    assert await resolver.vectors.exists(VectorKind.NODE, deleted) == set()


async def test_delete_refuses_an_entity_with_a_live_fact(graph_backend: GraphBackend):
    """A live fact is never deleted with its entity."""
    stores = await graph_backend.open_stores()
    store = stores.resolver_store
    resolver = await make(store)
    result = await resolver.resolve([ex("1", "Acme", "Company")])
    entity_id = result.entities[0].entity_id
    await graph_backend.seed_fact_edge(stores, entity_id)

    with pytest.raises(EntityHasFactsError):
        await resolver.delete(entity_id)

    assert len(await store.get_entities([entity_id])) == 1


async def test_an_expired_fact_version_does_not_block_a_delete(
    graph_backend: GraphBackend,
):
    """Only the current version of a fact counts as a live edge."""
    stores = await graph_backend.open_stores()
    store = stores.resolver_store
    resolver = await make(store)
    result = await resolver.resolve([ex("1", "Acme", "Company")])
    entity_id = result.entities[0].entity_id
    await graph_backend.seed_fact_edge(stores, entity_id, expired_at=5)
    await graph_backend.seed_fact_edge(stores, entity_id, status="withdrawn")

    await resolver.delete(entity_id)

    assert await store.get_entities([entity_id]) == []


async def test_a_new_mention_restores_an_archived_entity(graph_backend: GraphBackend):
    """A mention that matches an archived entity reuses it and makes it active."""
    store = (await graph_backend.open_stores()).resolver_store
    resolver = await make(store)
    first = await resolver.resolve([ex("1", "Acme", "Company")])
    entity_id = first.entities[0].entity_id
    await resolver.archive(entity_id)

    again = await resolver.resolve([ex("1", "Acme", "Company", episode_id="ep2")])

    assert again.entities[0].entity_id == entity_id
    stored = (await store.get_entities([entity_id]))[0]
    assert stored.status is EntityStatus.ACTIVE
    assert stored.mention_count == 2


async def test_archive_keeps_the_entity_readable(graph_backend: GraphBackend):
    """Archive changes the status without deleting the row."""
    store = (await graph_backend.open_stores()).resolver_store
    resolver = await make(store)
    result = await resolver.resolve([ex("1", "Acme", "Company")])
    entity_id = result.entities[0].entity_id

    await resolver.archive(entity_id)

    stored = (await store.get_entities([entity_id]))[0]
    assert stored.status is EntityStatus.ARCHIVED
