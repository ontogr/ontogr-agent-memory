"""Merge, unmerge and reset-side behavior on a live graph tenant."""

from collections.abc import Sequence

import pytest

from ontogr_mem.data_models import SystemClock
from ontogr_mem.data_models.entities import (
    Entity,
    EntityStatus,
    MergeRecord,
    MergeState,
    VerdictKind,
    VerdictSource,
)
from ontogr_mem.graphstore import QuerySpec
from ontogr_mem.graphstore.cypher import TenantScope
from ontogr_mem.graphstore.resolver.rows import entity_row
from ontogr_mem.resolution.errors import (
    ConcurrentEntityWriteError,
    EntityHasFactsError,
    SnapshotPurgedError,
    UnmergeBlockedError,
)
from ontogr_mem.resolution.ids import entity_id
from ontogr_mem.resolution.merge import (
    build_merge_plan,
    pick_survivor,
    survivor_order,
)
from ontogr_mem.resolution.policy import ResolutionPolicy, TypeRelations
from ontogr_mem.resolution.resolver_store import ResolverStore, merge_intent
from ontogr_mem.vectorstore import VectorKind
from tests.contracts.resolver_store import alias, entity
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
    unmerged = await resolver.unmerge(record.merge_id, actor="user")

    restored = await store.get_entities([survivor_id, absorbed_id])
    assert all(item.revision == unmerged.seq for item in restored)
    assert all(item.last_checked_version == 0 for item in restored)
    verdict = await store.live_verdict(survivor_id, absorbed_id)
    assert verdict is not None
    assert verdict.verdict is VerdictKind.DISTINCT
    assert verdict.source is VerdictSource.USER

    with pytest.raises(UnmergeBlockedError):
        await resolver.merge(survivor_id, absorbed_id, actor="resolver")


async def test_merge_rejects_aliases_added_after_planning(
    graph_backend: GraphBackend,
):
    """A stale merge plan cannot drop an alias added after it was built."""
    store = (await graph_backend.open_stores()).resolver_store
    await store.write_entity_batch(
        [
            entity("e1", name="Acme"),
            entity("e2", name="Acme Incorporated"),
        ],
        [alias("e2", "Original name")],
    )
    plan = await build_merge_plan(
        store,
        ResolutionPolicy(),
        TypeRelations(),
        "e1",
        "e2",
        clock=SystemClock(),
        actor="user",
        reason="",
    )
    await store.write_entity_batch([], [alias("e2", "Name added later")])

    with pytest.raises(ConcurrentEntityWriteError):
        await store.apply_merge(plan)

    absorbed = (await store.get_entities(["e2"]))[0]
    assert absorbed.status is EntityStatus.ACTIVE
    assert {item.text for item in await store.aliases_of("e2")} == {
        "Original name",
        "Name added later",
    }
    assert await store.aliases_of("e1") == []


@pytest.mark.parametrize("checkpoint_before_recovery", [True, False])
async def test_recovery_does_not_hide_an_alias_move_from_a_checkpoint(
    graph_backend: GraphBackend,
    checkpoint_before_recovery: bool,
):
    """A survivor checkpoint does not hide an alias move from recovery."""
    handle = await graph_backend.open_tenant()
    stores = await graph_backend.stores_for(handle.tenant)
    store = stores.resolver_store
    runner = graph_backend.store.runner(handle)
    scope = TenantScope.from_mode(graph_backend.tenant_keyed)
    await store.write_entity_batch(
        [
            entity("e1", name="Acme"),
            entity("e2", name="Acme Incorporated"),
        ],
        [alias("e2", "Original name")],
    )
    resolver = await make(store)
    record = await resolver.merge("e1", "e2", actor="user")
    applying = record.model_copy(update={"state": MergeState.APPLYING})
    await runner.run(
        QuerySpec(
            f"MATCH (m:MergeRecord {scope.pattern('id: $id')}) "
            "SET m.state = $state, m.json = $json",
            {
                "id": record.merge_id,
                "state": MergeState.APPLYING.value,
                "json": applying.model_dump_json(),
            },
            idempotent=True,
            tag="test_merge_interruption",
        )
    )
    moved_alias = (await store.aliases_of("e1"))[0]
    restored_alias = moved_alias.model_copy(update={"entity_id": "e2"})
    await runner.run(
        QuerySpec(
            f"MATCH (a:Alias {scope.pattern('id: $alias_id')}) "
            f"OPTIONAL MATCH (a)-[old:ALIAS_OF {scope.edge()}]->"
            f"(:Entity {scope.pattern()}) DELETE old "
            f"WITH a MATCH (e:Entity {scope.pattern('id: $entity_id')}) "
            "SET a += $props "
            f"MERGE (a)-[r:ALIAS_OF {scope.edge()}]->(e) "
            f"{scope.stamp_assignment('r')}",
            {
                "alias_id": restored_alias.alias_id,
                "entity_id": restored_alias.entity_id,
                "props": {
                    "entity_id": restored_alias.entity_id,
                    "strict_key": restored_alias.strict_key,
                    "loose_key": restored_alias.loose_key,
                    "text": restored_alias.text,
                    "json": restored_alias.model_dump_json(),
                },
            },
            idempotent=True,
            tag="test_restore_alias_owner",
        )
    )
    survivor = (await store.get_entities(["e1"]))[0]
    if checkpoint_before_recovery:
        await store.mark_deduplication_checked([survivor])
        assert await store.recover_merges() == []
    else:
        assert await store.recover_merges() == []
        await store.mark_deduplication_checked([survivor])

    current_record = await store.merge_record(record.merge_id)
    assert current_record is not None
    assert current_record.state is MergeState.APPLYING
    assert [item.text for item in await store.aliases_of("e2")] == ["Original name"]
    assert await store.aliases_of("e1") == []


async def test_recovery_keeps_aliases_a_finished_merge_already_moved(
    graph_backend: GraphBackend,
):
    """Recovery finishes an applying merge whose aliases already moved."""
    handle = await graph_backend.open_tenant()
    stores = await graph_backend.stores_for(handle.tenant)
    store = stores.resolver_store
    runner = graph_backend.store.runner(handle)
    scope = TenantScope.from_mode(graph_backend.tenant_keyed)
    await store.write_entity_batch(
        [
            entity("e1", name="Acme"),
            entity("e2", name="Acme Incorporated"),
        ],
        [alias("e2", "Original name")],
    )
    resolver = await make(store)
    record = await resolver.merge("e1", "e2", actor="user")
    applying = record.model_copy(update={"state": MergeState.APPLYING})
    await runner.run(
        QuerySpec(
            f"MATCH (m:MergeRecord {scope.pattern('id: $id')}) "
            "SET m.state = $state, m.json = $json",
            {
                "id": record.merge_id,
                "state": MergeState.APPLYING.value,
                "json": applying.model_dump_json(),
            },
            idempotent=True,
            tag="test_merge_interruption",
        )
    )

    finished = await store.recover_merges()

    assert [item.merge_id for item in finished] == [record.merge_id]
    current_record = await store.merge_record(record.merge_id)
    assert current_record is not None
    assert current_record.state is MergeState.APPLIED
    assert [item.text for item in await store.aliases_of("e1")] == ["Original name"]
    assert await store.aliases_of("e2") == []


async def test_recovery_refuses_aliases_owned_outside_the_intent(
    graph_backend: GraphBackend,
):
    """Recovery leaves a third owner's alias and checkpoint unchanged."""
    handle = await graph_backend.open_tenant()
    stores = await graph_backend.stores_for(handle.tenant)
    store = stores.resolver_store
    runner = graph_backend.store.runner(handle)
    scope = TenantScope.from_mode(graph_backend.tenant_keyed)
    await store.write_entity_batch(
        [
            entity("e1", name="Acme"),
            entity("e2", name="Acme Incorporated"),
            entity("e3", name="Other owner"),
        ],
        [alias("e2", "Original name")],
    )
    resolver = await make(store)
    record = await resolver.merge("e1", "e2", actor="user")
    applying = record.model_copy(update={"state": MergeState.APPLYING})
    await runner.run(
        QuerySpec(
            f"MATCH (m:MergeRecord {scope.pattern('id: $id')}) "
            "SET m.state = $state, m.json = $json",
            {
                "id": record.merge_id,
                "state": MergeState.APPLYING.value,
                "json": applying.model_dump_json(),
            },
            idempotent=True,
            tag="test_merge_interruption",
        )
    )
    moved_alias = (await store.aliases_of("e1"))[0]
    third_owner_alias = moved_alias.model_copy(update={"entity_id": "e3"})
    await runner.run(
        QuerySpec(
            f"MATCH (a:Alias {scope.pattern('id: $alias_id')}) "
            f"OPTIONAL MATCH (a)-[old:ALIAS_OF {scope.edge()}]->"
            f"(:Entity {scope.pattern()}) DELETE old "
            f"WITH a MATCH (e:Entity {scope.pattern('id: $entity_id')}) "
            "SET a += $props "
            f"MERGE (a)-[r:ALIAS_OF {scope.edge()}]->(e) "
            f"{scope.stamp_assignment('r')}",
            {
                "alias_id": third_owner_alias.alias_id,
                "entity_id": third_owner_alias.entity_id,
                "props": {
                    "entity_id": third_owner_alias.entity_id,
                    "strict_key": third_owner_alias.strict_key,
                    "loose_key": third_owner_alias.loose_key,
                    "text": third_owner_alias.text,
                    "json": third_owner_alias.model_dump_json(),
                },
            },
            idempotent=True,
            tag="test_restore_alias_owner",
        )
    )
    third_owner = (await store.get_entities(["e3"]))[0]
    await store.mark_deduplication_checked([third_owner])
    checked_third_owner = (await store.get_entities(["e3"]))[0]

    assert await store.recover_merges() == []

    current_record = await store.merge_record(record.merge_id)
    assert current_record is not None
    assert current_record.state is MergeState.APPLYING
    assert (await store.get_entities(["e3"]))[0] == checked_third_owner
    assert [item.entity_id for item in await store.aliases_of("e3")] == ["e3"]


async def test_recovery_replays_stamped_merge_and_unmerge_intents(
    graph_backend: GraphBackend,
):
    """Recovery reuses operation revisions and preserves unmerge verdicts."""
    handle = await graph_backend.open_tenant()
    stores = await graph_backend.stores_for(handle.tenant)
    store = stores.resolver_store
    runner = graph_backend.store.runner(handle)
    scope = TenantScope.from_mode(graph_backend.tenant_keyed)
    written = await store.write_entity_batch(
        [entity("e1", name="Acme"), entity("e2", name="Acme Incorporated")], []
    )
    checked = [
        await store.save_entity(
            item.model_copy(update={"last_checked_version": item.revision}),
            bump_revision=False,
        )
        for item in written
    ]
    resolver = await make(store)

    async def interrupt(record: MergeRecord) -> None:
        """Leave a saved record for the recovery path to finish."""
        applying = record.model_copy(update={"state": MergeState.APPLYING})
        await runner.run(
            QuerySpec(
                f"MATCH (m:MergeRecord {scope.pattern('id: $id')}) "
                "SET m.state = $state, m.json = $json",
                {
                    "id": record.merge_id,
                    "state": MergeState.APPLYING.value,
                    "json": applying.model_dump_json(),
                },
                idempotent=True,
                tag="test_merge_interruption",
            )
        )

    async def restore(rows: Sequence[Entity]) -> None:
        """Put entity rows back to the state before the interrupted write."""
        await runner.run(
            QuerySpec(
                "UNWIND $rows AS item "
                f"MATCH (e:Entity {scope.pattern('id: item.id')}) "
                "SET e += item.props",
                {"rows": [entity_row(item) for item in rows]},
                idempotent=True,
                tag="test_restore_entity_rows",
            )
        )

    merge = await resolver.merge("e1", "e2", actor="user")
    merge_rows = merge_intent(merge.snapshot)
    assert merge_rows is not None
    assert merge_rows.survivor.revision == merge.seq
    assert merge_rows.absorbed.revision == merge.seq
    assert [
        item.last_checked_version for item in (merge_rows.survivor, merge_rows.absorbed)
    ] == [item.last_checked_version for item in checked]

    await interrupt(merge)
    await restore(checked)
    recovered_merge = await store.recover_merges()
    assert [(item.state, item.seq) for item in recovered_merge] == [
        (MergeState.APPLIED, merge.seq)
    ]
    after_merge_recovery = await store.get_entities(["e1", "e2"])
    assert [item.revision for item in after_merge_recovery] == [merge.seq, merge.seq]
    assert [item.last_checked_version for item in after_merge_recovery] == [
        item.last_checked_version for item in checked
    ]
    assert [(item.status, item.merged_into) for item in after_merge_recovery] == [
        (EntityStatus.ACTIVE, None),
        (EntityStatus.MERGED, "e1"),
    ]

    unmerge = await resolver.unmerge(merge.merge_id, actor="user")
    unmerge_rows = merge_intent(unmerge.snapshot)
    assert unmerge_rows is not None
    assert unmerge_rows.survivor.revision == unmerge.seq
    assert unmerge_rows.absorbed.revision == unmerge.seq
    assert [
        item.last_checked_version
        for item in (unmerge_rows.survivor, unmerge_rows.absorbed)
    ] == [item.last_checked_version for item in checked]

    await interrupt(unmerge)
    await restore(after_merge_recovery)
    recovered_unmerge = await store.recover_merges()
    assert [(item.state, item.seq) for item in recovered_unmerge] == [
        (MergeState.REVERTED, unmerge.seq)
    ]
    after_unmerge_recovery = await store.get_entities(["e1", "e2"])
    assert [item.revision for item in after_unmerge_recovery] == [
        unmerge.seq,
        unmerge.seq,
    ]
    assert [item.last_checked_version for item in after_unmerge_recovery] == [
        item.last_checked_version for item in checked
    ]
    assert [(item.status, item.merged_into) for item in after_unmerge_recovery] == [
        (EntityStatus.ACTIVE, None),
        (EntityStatus.ACTIVE, None),
    ]
    verdict = await store.live_verdict("e1", "e2")
    assert verdict is not None
    assert verdict.verdict is VerdictKind.DISTINCT
    assert verdict.source is VerdictSource.USER


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
