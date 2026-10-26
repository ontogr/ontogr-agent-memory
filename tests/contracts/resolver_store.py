"""Contract every `ResolverStore` backend must satisfy.

A backend test module subclasses `ResolverStoreContract` and provides the
fixtures and hooks named in the class body. The class has no `Test` prefix, so
pytest collects only its subclasses.
"""

import asyncio
from datetime import UTC, datetime

import pytest

from ontogr_mem.data_models.entities import (
    Alias,
    AliasSource,
    DecisionMethod,
    Entity,
    EntityStatus,
    Mention,
    MergeRecord,
    MergeState,
    Outcome,
)
from ontogr_mem.resolution.errors import (
    EntityHasFactsError,
    EpisodeAlreadyResolvedError,
    MergeRefusedError,
)
from ontogr_mem.resolution.ids import alias_id
from ontogr_mem.resolution.keys import strict_key
from ontogr_mem.resolution.resolver_store import (
    MergePlan,
    ResolverStore,
    UnmergePlan,
    merge_intent,
)


NOW = datetime(2026, 10, 21, 12, 0, tzinfo=UTC)


def entity(
    entity_id: str,
    *,
    name: str | None = None,
    uri: str | None = None,
    status: EntityStatus = EntityStatus.ACTIVE,
    merged_into: str | None = None,
) -> Entity:
    """Build an active or merged entity with a fixed clock."""
    canonical_name = name or entity_id
    return Entity(
        entity_id=entity_id,
        type_label="Company",
        canonical_name=canonical_name,
        strict_key=canonical_name.casefold(),
        loose_key=canonical_name.casefold(),
        uri=uri,
        status=status,
        merged_into=merged_into,
        created_at=NOW,
        updated_at=NOW,
    )


def alias(entity_id: str, text: str, alias_id: str | None = None) -> Alias:
    """Build an alias of an entity."""
    return Alias(
        alias_id=alias_id or f"a-{entity_id}-{text}",
        entity_id=entity_id,
        text=text,
        strict_key=text.casefold(),
        loose_key=text.casefold(),
        source=AliasSource.EXTRACTED,
    )


def mention(
    entity_id: str,
    local_id: str,
    *,
    episode_id: str = "ep1",
    mention_id: str | None = None,
) -> Mention:
    """Build a mention with a decision."""
    return Mention(
        mention_id=mention_id or f"m-{episode_id}-{local_id}",
        entity_id=entity_id,
        episode_id=episode_id,
        local_id=local_id,
        surface_form=local_id,
        type_label_at_extraction="Company",
        resolver_version="1",
        decision_method=DecisionMethod.KEY,
        decision_outcome=Outcome.MATCHED,
    )


def merge_record(merge_id: str, survivor_id: str, absorbed_id: str) -> MergeRecord:
    """Build an applied merge record."""
    return MergeRecord(
        merge_id=merge_id,
        survivor_id=survivor_id,
        absorbed_id=absorbed_id,
        actor="test",
        reason="contract",
        state=MergeState.APPLIED,
        created_at=NOW,
    )


def merge_plan(
    merge_id: str,
    survivor: Entity,
    absorbed: Entity,
    *,
    mentions: tuple[Mention, ...] = (),
    aliases: tuple[Alias, ...] = (),
) -> MergePlan:
    """Build the rows one merge writes."""
    moved_mentions = tuple(
        m.model_copy(update={"entity_id": survivor.entity_id}) for m in mentions
    )
    moved_aliases = tuple(
        a.model_copy(update={"entity_id": survivor.entity_id}) for a in aliases
    )
    return MergePlan(
        record=merge_record(merge_id, survivor.entity_id, absorbed.entity_id),
        survivor=survivor.model_copy(
            update={
                "mention_count": survivor.mention_count + len(moved_mentions),
                "updated_at": NOW,
            }
        ),
        absorbed=absorbed.model_copy(
            update={
                "status": EntityStatus.MERGED,
                "merged_into": survivor.entity_id,
                "mention_count": 0,
                "updated_at": NOW,
            }
        ),
        mentions=moved_mentions,
        aliases=moved_aliases,
    )


def unmerge_plan(
    record: MergeRecord,
    survivor: Entity,
    absorbed: Entity,
    *,
    mentions: tuple[Mention, ...] = (),
    aliases: tuple[Alias, ...] = (),
) -> UnmergePlan:
    """Build the rows one unmerge writes."""
    restored_mentions = tuple(
        m.model_copy(update={"entity_id": absorbed.entity_id}) for m in mentions
    )
    restored_aliases = tuple(
        a.model_copy(update={"entity_id": absorbed.entity_id}) for a in aliases
    )
    return UnmergePlan(
        record=record.model_copy(update={"state": MergeState.REVERTED}),
        survivor=survivor.model_copy(
            update={
                "mention_count": survivor.mention_count - len(restored_mentions),
                "updated_at": NOW,
            }
        ),
        absorbed=absorbed.model_copy(
            update={
                "status": EntityStatus.ACTIVE,
                "merged_into": None,
                "mention_count": len(restored_mentions),
                "updated_at": NOW,
            }
        ),
        mentions=restored_mentions,
        aliases=restored_aliases,
    )


class ResolverStoreContract:
    """Behavior shared by every `ResolverStore`.

    Subclasses define these fixtures and hooks:

    - `store`: a store bound to a fresh tenant.
    - `other_store`: a store bound to a different fresh tenant.
    - `seed_fact_edge(store, entity_id, *, live)`: write one `FACT` edge that
      touches the entity.
    - `fact_edge_count(store, entity_id, *, live)`: count the `FACT` edges that
      touch the entity and are live or withdrawn.
    """

    async def seed_fact_edge(
        self, store: ResolverStore, entity_id: str, *, live: bool
    ) -> None:
        """Write one live or withdrawn fact edge for the entity."""
        raise NotImplementedError

    async def fact_edge_count(
        self, store: ResolverStore, entity_id: str, *, live: bool
    ) -> int:
        """Count the live or withdrawn fact edges of the entity."""
        raise NotImplementedError

    # 1. Round trip and merged pointer

    async def test_entity_write_and_read_round_trip(self, store: ResolverStore):
        """Written entities come back in input order and with their fields."""
        second = entity("e2", name="Beta")
        first = entity("e1", name="Alpha", uri="https://example.test/a")

        stored = await store.write_entity_batch([second, first], [])

        assert [e.entity_id for e in stored] == ["e2", "e1"]
        assert stored[1].uri == "https://example.test/a"
        assert [e.entity_id for e in await store.get_entities(["e1", "e2"])] == [
            "e1",
            "e2",
        ]

    async def test_has_merge_record_finds_both_entities_in_a_merge(self, store):
        """Merge history protects both its survivor and absorbed entity."""
        survivor = entity("e1")
        absorbed = entity("e2")
        await store.write_entity_batch([survivor, absorbed], [])
        await store.apply_merge(merge_plan("merge-1", survivor, absorbed))

        assert await store.has_merge_record("e1")
        assert await store.has_merge_record("e2")
        assert not await store.has_merge_record("unrelated")

    async def test_mentions_of_episode_is_sorted_and_tenant_scoped(
        self, store, other_store
    ):
        """Episode lookup is ordered, filtered, and scoped to one tenant."""
        first = mention("entity-1", "local-1", episode_id="ep1", mention_id="m2")
        second = mention("entity-2", "local-2", episode_id="ep1", mention_id="m1")
        other = mention("entity-1", "local-3", episode_id="ep2", mention_id="m3")
        await store.write_entity_batch([entity("entity-1"), entity("entity-2")], [])
        await store.write_mentions("ep1", [first, second])
        await store.write_mentions("ep2", [other])
        foreign = mention(
            "other-entity", "local-1", episode_id="ep1", mention_id="foreign"
        )
        await other_store.write_entity_batch([entity("other-entity")], [])
        await other_store.write_mentions("ep1", [foreign])

        stored = await store.mentions_of_episode("ep1")
        assert [(row.mention_id, row.episode_id) for row in stored] == [
            ("m1", "ep1"),
            ("m2", "ep1"),
        ]
        assert await store.mentions_of_episode("missing") == []
        other_mentions = await other_store.mentions_of_episode("ep1")
        assert [row.mention_id for row in other_mentions] == ["foreign"]

    async def test_get_entities_follows_a_merged_pointer(self, store: ResolverStore):
        """A merged entity is replaced by its survivor when asked."""
        survivor = entity("e1")
        absorbed = entity(
            "e2",
            status=EntityStatus.MERGED,
            merged_into=survivor.entity_id,
        )
        await store.write_entity_batch([survivor, absorbed], [])

        direct = (await store.get_entities(["e2"]))[0]
        followed = (await store.get_entities(["e2"], follow_merged=True))[0]
        assert direct.status is EntityStatus.MERGED
        assert direct.merged_into == "e1"
        assert followed.entity_id == "e1"

    # 2. Lookups

    async def test_loose_key_candidates_follow_merges_without_changing_identity(
        self, store: ResolverStore
    ):
        """Accent-insensitive candidates remain separate entities until merged."""
        first = entity("e1", name="Zürich")
        second = entity("e2", name="Zurich")
        names = [
            alias("e1", "Zürich").model_copy(update={"loose_key": "zurich"}),
            alias("e2", "Zurich"),
        ]
        await store.write_entity_batch([first, second], names)
        assert [e.entity_id for e in await store.entities_by_loose_key("zurich")] == [
            "e1",
            "e2",
        ]
        assert [e.entity_id for e in await store.entities_by_strict_key("zurich")] == [
            "e2"
        ]
        await store.apply_merge(
            merge_plan("loose-merge", first, second, aliases=(names[1],))
        )
        assert [e.entity_id for e in await store.entities_by_loose_key("zurich")] == [
            "e1"
        ]
        assert await store.entities_by_loose_key("unknown") == []

    async def test_lookup_by_uri_and_by_alias_strict_key(self, store: ResolverStore):
        """Both lookup paths return the entity the row points at."""
        acme = entity("e1", name="Acme", uri="https://example.test/acme")
        await store.write_entity_batch([acme], [alias("e1", "Acme Incorporated")])

        assert [
            e.entity_id
            for e in await store.entities_by_uri("https://example.test/acme")
        ] == ["e1"]
        assert [
            e.entity_id for e in await store.entities_by_strict_key("acme incorporated")
        ] == ["e1"]

    # 3. One resolution run per episode

    async def test_mention_write_is_refused_when_the_episode_has_mentions(
        self, store: ResolverStore
    ):
        """A second run over one episode is refused and names the episode."""
        await store.write_entity_batch([entity("e1")], [])
        await store.write_mentions("ep1", [mention("e1", "m1")])

        with pytest.raises(EpisodeAlreadyResolvedError) as raised:
            await store.write_mentions("ep1", [mention("e1", "m2")])

        assert raised.value.episode_id == "ep1"

    # 4. Idempotent mention write

    async def test_mention_write_is_idempotent_and_recounts(self, store: ResolverStore):
        """Writing the same ids again changes nothing and keeps the count exact."""
        await store.write_entity_batch([entity("e1")], [])
        batch = [mention("e1", "m1"), mention("e1", "m2")]

        await store.write_mentions("ep1", batch)
        await store.write_mentions("ep1", batch)

        mentions = await store.mentions_of("e1")
        stored = (await store.get_entities(["e1"]))[0]
        assert len(mentions) == 2
        assert stored.mention_count == 2

    # 5. Removing an episode

    async def test_remove_episode_mentions_returns_the_touched_entities(
        self, store: ResolverStore
    ):
        """The returned ids are exactly the entities that lost a mention."""
        await store.write_entity_batch([entity("e1"), entity("e2"), entity("e3")], [])
        await store.write_mentions("ep1", [mention("e1", "m1"), mention("e2", "m2")])
        await store.write_mentions("ep2", [mention("e3", "m3")])

        touched = await store.remove_episode_mentions("ep1")

        assert touched == ["e1", "e2"]
        assert await store.mentions_of("e1") == []
        assert (await store.get_entities(["e1"]))[0].mention_count == 0
        assert len(await store.mentions_of("e3")) == 1

    # 5a. The revision of a mention write

    async def test_local_reference_is_scoped_to_its_episode(self, store: ResolverStore):
        """The same local id can name different entities in different episodes."""
        await store.write_entity_batch([entity("e1"), entity("e2")], [])
        await store.write_mentions("ep1", [mention("e1", "local")])
        await store.write_mentions("ep2", [mention("e2", "local", episode_id="ep2")])

        assert await store.entity_id_for_ref("ep1", "local") == "e1"
        assert await store.entity_id_for_ref("ep2", "local") == "e2"
        assert await store.entity_id_for_ref("ep3", "local") is None
        assert await store.entity_id_for_ref("ep1", "missing") is None

    async def test_mentions_carry_the_revision_of_their_write(
        self, store: ResolverStore
    ):
        """A mention keeps the store revision of the write that stored it."""
        await store.write_entity_batch([entity("e1")], [])

        await store.write_mentions("ep1", [mention("e1", "m1")])

        stored = await store.mentions_of("e1")
        assert stored[0].seq > 0

    async def test_a_later_mention_write_gets_a_higher_revision(
        self, store: ResolverStore
    ):
        """Two writes of one episode stamp two different revisions."""
        await store.write_entity_batch([entity("e1")], [])

        await store.write_mentions("ep1", [mention("e1", "m1")])
        await store.write_mentions("ep2", [mention("e1", "m2")])

        stored = await store.mentions_of("e1")
        assert stored[0].seq < stored[1].seq

    async def test_writing_the_same_mentions_again_keeps_the_revision(
        self, store: ResolverStore
    ):
        """Writing the same ids again changes nothing, not even the revision."""
        await store.write_entity_batch([entity("e1")], [])
        batch = [mention("e1", "m1")]

        await store.write_mentions("ep1", batch)
        first = (await store.mentions_of("e1"))[0].seq
        await store.write_mentions("ep1", batch)

        assert (await store.mentions_of("e1"))[0].seq == first

    # 6. Orphan deletion

    async def test_delete_orphans_removes_a_safe_entity_and_its_tombstones(
        self, store: ResolverStore
    ):
        """No mentions, no live fact and no merge record means deletion."""
        await store.write_entity_batch([entity("e1")], [])
        await self.seed_fact_edge(store, "e1", live=False)

        deleted = await store.delete_orphans(["e1"])

        assert deleted == ["e1"]
        assert await store.get_entities(["e1"]) == []
        assert await self.fact_edge_count(store, "e1", live=False) == 0

    async def test_delete_orphans_keeps_an_entity_named_by_a_merge_record(
        self, store: ResolverStore
    ):
        """A merge record still needs its survivor, so it blocks deletion."""
        survivor = entity("e1")
        absorbed = entity("e2")
        await store.write_entity_batch([survivor, absorbed], [])
        await store.write_mentions("ep1", [mention("e1", "m1")])
        await store.apply_merge(merge_plan("mg1", survivor, absorbed))
        await store.remove_episode_mentions("ep1")

        assert await store.delete_orphans(["e1"]) == []
        assert [e.entity_id for e in await store.get_entities(["e1"])] == ["e1"]

    async def test_delete_orphans_keeps_an_entity_with_a_live_fact(
        self, store: ResolverStore
    ):
        """A live fact from another episode protects the entity."""
        await store.write_entity_batch([entity("e1")], [])
        await self.seed_fact_edge(store, "e1", live=True)

        assert await store.delete_orphans(["e1"]) == []
        assert await self.fact_edge_count(store, "e1", live=True) == 1

    # 7. Reversible merge

    async def test_merge_then_unmerge_restores_the_absorbed_entity(
        self, store: ResolverStore
    ):
        """The absorbed entity, its aliases and its mentions come back."""
        survivor = entity("e1", name="Acme")
        absorbed = entity("e2", name="Acme Inc")
        survivor_alias = alias("e1", "Acme")
        absorbed_alias = alias("e2", "Acme Inc")
        survivor_mention = mention("e1", "m1")
        absorbed_mention = mention("e2", "m2")
        await store.write_entity_batch(
            [survivor, absorbed], [survivor_alias, absorbed_alias]
        )
        await store.write_mentions("ep1", [survivor_mention, absorbed_mention])

        plan = merge_plan(
            "mg1",
            survivor.model_copy(update={"mention_count": 1}),
            absorbed.model_copy(update={"mention_count": 1}),
            mentions=(absorbed_mention,),
            aliases=(absorbed_alias,),
        )
        record = await store.apply_merge(plan)

        merged = (await store.get_entities(["e2"]))[0]
        assert merged.status is EntityStatus.MERGED
        assert merged.merged_into == "e1"
        assert await store.mentions_of("e2") == []
        assert len(await store.mentions_of("e1")) == 2
        assert {a.alias_id for a in await store.aliases_of("e1")} == {
            survivor_alias.alias_id,
            absorbed_alias.alias_id,
        }

        restored = await store.apply_unmerge(
            unmerge_plan(
                record,
                plan.survivor,
                plan.absorbed,
                mentions=(absorbed_mention,),
                aliases=(absorbed_alias,),
            )
        )

        absorbed_again = (await store.get_entities(["e2"]))[0]
        assert restored.state is MergeState.REVERTED
        assert absorbed_again.status is EntityStatus.ACTIVE
        assert absorbed_again.merged_into is None
        assert [m.local_id for m in await store.mentions_of("e2")] == ["m2"]
        assert {a.alias_id for a in await store.aliases_of("e2")} == {
            absorbed_alias.alias_id
        }

    # 7a. The rows the merge keeps

    async def test_the_merge_record_keeps_the_rows_of_the_merge(
        self, store: ResolverStore
    ):
        """The stored record holds the rows, so an interrupted merge can finish."""
        survivor = entity("e1", name="Acme")
        absorbed = entity("e2", name="Acme Inc")
        await store.write_entity_batch([survivor, absorbed], [])
        plan = merge_plan("mg1", survivor, absorbed)

        record = await store.apply_merge(plan)

        rows = merge_intent(record.snapshot)
        assert rows is not None
        assert rows.state is MergeState.APPLIED
        assert rows.survivor.entity_id == survivor.entity_id
        assert rows.absorbed.entity_id == absorbed.entity_id
        assert rows.absorbed.status is EntityStatus.MERGED
        assert rows.absorbed.merged_into == survivor.entity_id

    # 8. Concurrent merge

    async def test_concurrent_merges_of_one_pair_yield_one_applied_record(
        self, store: ResolverStore
    ):
        """Only the first of twenty callers merges; the rest are refused."""
        survivor = entity("e1")
        absorbed = entity("e2")
        await store.write_entity_batch([survivor, absorbed], [])
        plans = [merge_plan("mg1", survivor, absorbed) for _ in range(20)]

        results = await asyncio.gather(
            *(store.apply_merge(p) for p in plans), return_exceptions=True
        )

        assert sum(not isinstance(r, Exception) for r in results) == 1
        assert all(
            isinstance(r, MergeRefusedError)
            for r in results
            if isinstance(r, Exception)
        )
        record = await store.merge_record("mg1")
        assert record is not None and record.state is MergeState.APPLIED

    async def test_competing_merge_ids_cannot_merge_the_same_entities(
        self, store: ResolverStore
    ):
        """Only one merge id can apply to a pair of active entities."""
        survivor = entity("e1")
        absorbed = entity("e2")
        await store.write_entity_batch([survivor, absorbed], [])
        plans = [merge_plan(f"mg{index}", survivor, absorbed) for index in range(2)]

        results = await asyncio.gather(
            *(store.apply_merge(plan) for plan in plans), return_exceptions=True
        )

        applied = [result for result in results if not isinstance(result, Exception)]
        refused = [result for result in results if isinstance(result, Exception)]
        assert len(applied) == 1
        assert len(refused) == 1 and isinstance(refused[0], MergeRefusedError)
        records = [await store.merge_record(plan.record.merge_id) for plan in plans]
        assert sum(record is not None for record in records) == 1
        merged = (await store.get_entities(["e2"]))[0]
        assert merged.status is EntityStatus.MERGED
        assert merged.merged_into == "e1"

    # 9. Tenant isolation

    async def test_the_same_ids_in_two_tenants_never_read_across(
        self, store: ResolverStore, other_store: ResolverStore
    ):
        """Tenant is fixed at construction, so ids do not leak between stores."""
        await store.write_entity_batch([entity("e1", name="First tenant")], [])
        await other_store.write_entity_batch([entity("e1", name="Second tenant")], [])

        assert (await store.get_entities(["e1"]))[0].canonical_name == "First tenant"
        assert (await other_store.get_entities(["e1"]))[
            0
        ].canonical_name == "Second tenant"

    async def test_all_entities_returns_every_status_in_id_order(
        self, store: ResolverStore, other_store: ResolverStore
    ):
        """Entity listing includes active, merged and archived rows."""
        rows = [
            entity("e3", status=EntityStatus.ARCHIVED),
            entity("e2", status=EntityStatus.MERGED, merged_into="e1"),
            entity("e1"),
        ]
        await store.write_entity_batch(rows, [])
        await other_store.write_entity_batch([entity("e4")], [])

        result = await store.all_entities()

        assert [(item.entity_id, item.status) for item in result] == [
            ("e1", EntityStatus.ACTIVE),
            ("e2", EntityStatus.MERGED),
            ("e3", EntityStatus.ARCHIVED),
        ]
        assert [item.entity_id for item in await other_store.all_entities()] == ["e4"]

    async def test_all_aliases_returns_tenant_aliases_in_id_order(
        self, store: ResolverStore, other_store: ResolverStore
    ):
        """Alias listing is stable and stays within its tenant."""
        await store.write_entity_batch(
            [entity("e1"), entity("e2")],
            [
                alias("e2", "second", "a-2"),
                alias("e1", "first", "a-1"),
            ],
        )
        await other_store.write_entity_batch(
            [entity("e3")], [alias("e3", "foreign", "a-foreign")]
        )

        result = await store.all_aliases()

        assert [item.alias_id for item in result] == ["a-1", "a-2"]
        assert [item.alias_id for item in await other_store.all_aliases()] == [
            "a-foreign"
        ]

    async def test_all_entities_and_aliases_are_empty_for_a_new_tenant(
        self, store: ResolverStore
    ):
        """Empty tenants have no entity or alias rows to list."""
        assert await store.all_entities() == []
        assert await store.all_aliases() == []

    async def test_prune_entity_text_removes_unsupported_aliases_and_blanks_text(
        self, store: ResolverStore
    ):
        """Only mention-supported aliases remain when entity text is pruned."""
        stored_entity = entity("e1", name="Old Name").model_copy(
            update={
                "summary": "Sensitive summary",
                "descriptions": ["Sensitive description"],
                "attributes": {"sensitive": "value"},
            }
        )
        await store.write_entity_batch(
            [stored_entity],
            [
                alias("e1", "Old Name"),
                alias("e1", "Current Name"),
                alias("e1", "Unsupported Name"),
            ],
        )
        mentions = [
            mention("e1", "old", mention_id="m-old").model_copy(
                update={"surface_form": "Old Name"}
            ),
            mention(
                "e1", "current", episode_id="ep2", mention_id="m-current"
            ).model_copy(update={"surface_form": "Current Name"}),
        ]
        await store.write_mentions("ep1", [mentions[0]])
        await store.write_mentions("ep2", [mentions[1]])

        pruned = await store.prune_entity_text(["missing", "e1", "e1"])

        assert pruned == ["e1"]
        stored = (await store.get_entities(["e1"]))[0]
        assert stored.canonical_name == "Old Name"
        assert stored.summary == ""
        assert stored.descriptions == []
        assert stored.attributes == {}
        assert stored.summary_stale
        assert {row.text for row in await store.aliases_of("e1")} == {
            "Old Name",
            "Current Name",
        }

    async def test_prune_entity_text_replaces_unmentioned_canonical_name(
        self, store: ResolverStore
    ):
        """Canonical replacement uses count, then the lowest mention id."""
        await store.write_entity_batch(
            [entity("e1", name="Retired Name")],
            [
                alias("e1", "Retired Name"),
                alias("e1", "Unsupported Name"),
            ],
        )
        mentions = [
            mention("e1", "common-1", mention_id="z-common-1").model_copy(
                update={"surface_form": "Common Name"}
            ),
            mention(
                "e1", "common-2", episode_id="ep2", mention_id="z-common-2"
            ).model_copy(update={"surface_form": "Common Name"}),
            mention("e1", "tie-1", episode_id="ep3", mention_id="a-tie-1").model_copy(
                update={"surface_form": "Tie Name"}
            ),
            mention("e1", "tie-2", episode_id="ep4", mention_id="a-tie-2").model_copy(
                update={"surface_form": "Tie Name"}
            ),
        ]
        for row in mentions:
            await store.write_mentions(row.episode_id, [row])

        assert await store.prune_entity_text(["e1"]) == ["e1"]

        updated = (await store.get_entities(["e1"]))[0]
        assert updated.canonical_name == "Tie Name"
        assert updated.strict_key == "tie name"
        assert updated.loose_key == "tie name"
        aliases = await store.aliases_of("e1")
        assert [(row.text, row.strict_key, row.loose_key) for row in aliases] == [
            ("Tie Name", "tie name", "tie name")
        ]
        assert aliases[0].alias_id == alias_id("e1", strict_key("Tie Name"))

        assert await store.prune_entity_text(["e1"]) == ["e1"]
        assert (await store.get_entities(["e1"]))[0] == updated
        assert await store.aliases_of("e1") == aliases

    async def test_prune_entity_text_preserves_merge_records(
        self, store: ResolverStore
    ):
        """Pruning a surviving entity does not alter its merge history."""
        survivor = entity("e1", name="Survivor")
        absorbed = entity("e2", name="Absorbed")
        await store.write_entity_batch([survivor, absorbed], [])
        record = await store.apply_merge(merge_plan("merge-1", survivor, absorbed))
        surviving_mention = mention(
            "e1", "remaining", episode_id="ep2", mention_id="m-remaining"
        ).model_copy(update={"surface_form": "Remaining Name"})
        await store.write_mentions("ep2", [surviving_mention])

        await store.prune_entity_text(["e1"])

        assert await store.merge_record(record.merge_id) == record

    # 10. Deduplication order

    async def test_list_for_deduplication_is_ordered_by_revision_then_id(
        self, store: ResolverStore
    ):
        """Two calls over an unchanged store return the same stable order."""
        await store.write_entity_batch([entity("e3"), entity("e1"), entity("e2")], [])

        first = await store.list_for_deduplication(0, False, 0)
        second = await store.list_for_deduplication(0, False, 0)

        assert [e.entity_id for e in first] == [e.entity_id for e in second]
        assert [(e.revision, e.entity_id) for e in first] == sorted(
            (e.revision, e.entity_id) for e in first
        )

    # 11. Delete and revision

    async def test_delete_entity_cascades_aliases_mentions_and_tombstones(
        self, store: ResolverStore
    ):
        """Deleting an entity removes its rows and leaves no tombstone."""
        await store.write_entity_batch([entity("e1")], [alias("e1", "Acme")])
        await store.write_mentions("ep1", [mention("e1", "m1")])
        await self.seed_fact_edge(store, "e1", live=False)

        await store.delete_entity("e1")

        assert await store.get_entities(["e1"]) == []
        assert await store.aliases_of("e1") == []
        assert await store.mentions_of("e1") == []
        assert await self.fact_edge_count(store, "e1", live=False) == 0

    async def test_delete_entity_refuses_a_live_fact(self, store: ResolverStore):
        """A live fact is never deleted with its entity."""
        await store.write_entity_batch([entity("e1")], [])
        await self.seed_fact_edge(store, "e1", live=True)

        with pytest.raises(EntityHasFactsError):
            await store.delete_entity("e1")

        assert len(await store.get_entities(["e1"])) == 1

    async def test_save_entity_without_a_bump_keeps_the_revision(
        self, store: ResolverStore
    ):
        """The caller can change fields without claiming a new revision."""
        await store.write_entity_batch([entity("e1", name="Before")], [])
        before = (await store.get_entities(["e1"]))[0]

        after = await store.save_entity(
            before.model_copy(update={"canonical_name": "After"}),
            bump_revision=False,
        )

        assert after.revision == before.revision
        assert (await store.get_entities(["e1"]))[0].canonical_name == "After"

    async def test_blocking_rows_list_archived_entities_only_on_request(
        self, store: ResolverStore
    ):
        """Archived entities are findable by a new mention but not by default."""
        await store.write_entity_batch(
            [
                entity("e1", name="Acme"),
                entity("e2", name="Beta", status=EntityStatus.ARCHIVED),
                entity(
                    "e3", name="Gamma", status=EntityStatus.MERGED, merged_into="e1"
                ),
            ],
            [],
        )

        default = [r.entity_id for r in await store.blocking_rows()]
        with_archived = [
            r.entity_id for r in await store.blocking_rows(include_archived=True)
        ]

        assert default == ["e1"]
        assert with_archived == ["e1", "e2"]
