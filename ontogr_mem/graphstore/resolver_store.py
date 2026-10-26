"""Resolver store on the graph backends.

The adapter keeps the same labels and edges on both servers: `Entity`, `Alias`,
`Mention`, `EntityVerdict`, `MergeRecord` and `ResolverCounter`, with
`ALIAS_OF`, `IN` and `OF` edges. Model data round-trips through one JSON
property. Indexed scalar properties serve the lookups.
"""

import asyncio
from collections.abc import Sequence

from ontogr_mem.data_models.entities import (
    Alias,
    Entity,
    EntityStatus,
    EntityVerdict,
    Mention,
    MergeRecord,
)
from ontogr_mem.graphstore.cypher import TenantScope
from ontogr_mem.graphstore.protocol import ScopedRunner
from ontogr_mem.graphstore.resolver.entities import ResolverEntities
from ontogr_mem.graphstore.resolver.merges import ResolverMerges
from ontogr_mem.graphstore.resolver.prune import ResolverPrune
from ontogr_mem.resolution.blocking import BlockingRow
from ontogr_mem.resolution.resolver_store import MergePlan, UnmergePlan


class GraphResolverStore:
    """`ResolverStore` over a tenant's graph.

    Args:
        runner: Runner bound to the tenant's graph.
        tenant_keyed: Override the runner's backend scope when supplied.
    """

    def __init__(
        self, runner: ScopedRunner, *, tenant_keyed: bool | None = None
    ) -> None:
        """Bind the store to a tenant's graph."""
        scope = (
            TenantScope.from_mode(tenant_keyed)
            if tenant_keyed is not None
            else runner.scope
        )
        self.resolve_lock = asyncio.Lock()
        self._entities = ResolverEntities(runner, scope)
        self._merges = ResolverMerges(runner, scope, self._entities, self.resolve_lock)
        self._prune = ResolverPrune(runner, scope, self._entities, self._merges)

    async def episode_resolution_state(self, episode_id: str) -> bool:
        """Return True when resolution already wrote mentions for the episode."""
        return await self._entities.episode_resolution_state(episode_id)

    async def entity_id_for_ref(self, episode_id: str, local_id: str) -> str | None:
        """Return the entity named by an episode-local reference, or None."""
        return await self._entities.entity_id_for_ref(episode_id, local_id)

    async def has_entity(self, entity_id: str) -> bool:
        """Return whether this tenant contains the entity id."""
        return await self._entities.has_entity(entity_id)

    async def entities_by_uri(self, uri: str) -> list[Entity]:
        """Return the entities carrying `uri`, in any status, ordered by id."""
        return await self._entities.entities_by_uri(uri)

    async def entities_by_strict_key(self, strict_key: str) -> list[Entity]:
        """Return the entities an alias with `strict_key` resolves to."""
        return await self._entities.entities_by_strict_key(strict_key)

    async def entities_by_loose_key(self, loose_key: str) -> list[Entity]:
        """Return candidate entities an alias's indexed loose key resolves to."""
        return await self._entities.entities_by_loose_key(loose_key)

    async def get_entities(
        self, ids: Sequence[str], *, follow_merged: bool = False
    ) -> list[Entity]:
        """Return the entities with these ids, ordered by id."""
        return await self._entities.get_entities(ids, follow_merged=follow_merged)

    async def blocking_rows(
        self, *, include_archived: bool = False
    ) -> list[BlockingRow]:
        """Return the blocking rows of the active entities, ordered by id."""
        return await self._entities.blocking_rows(include_archived=include_archived)

    async def write_entity_batch(
        self, entities: Sequence[Entity], aliases: Sequence[Alias]
    ) -> list[Entity]:
        """Write entities and aliases and stamp each entity with a new revision."""
        return await self._entities.write_entity_batch(entities, aliases)

    async def write_mentions(
        self, episode_id: str, mentions: Sequence[Mention]
    ) -> None:
        """Write an episode's mentions and recount the touched entities."""
        return await self._entities.write_mentions(episode_id, mentions)

    async def live_verdict(self, a_id: str, b_id: str) -> EntityVerdict | None:
        """Return the verdict for the pair unless a merge superseded it."""
        return await self._entities.live_verdict(a_id, b_id)

    async def write_verdicts(self, verdicts: Sequence[EntityVerdict]) -> None:
        """Write verdicts, replacing rows with the same id."""
        return await self._entities.write_verdicts(verdicts)

    async def list_for_deduplication(
        self, after_revision: int, provisional: bool, sweep_limit: int
    ) -> list[Entity]:
        """Return the active entities a deduplication run should consider."""
        return await self._entities.list_for_deduplication(
            after_revision, provisional, sweep_limit
        )

    async def apply_merge(self, plan: MergePlan) -> MergeRecord:
        """Apply one merge atomically and return its record.

        Raises:
            MergeRefusedError: If either entity is no longer active, or the
                merge was already applied.
        """
        return await self._merges.apply_merge(plan)

    async def apply_unmerge(self, plan: UnmergePlan) -> MergeRecord:
        """Apply one unmerge atomically and return its record.

        Raises:
            MergeRefusedError: If the merge record is missing or is not applied.
        """
        return await self._merges.apply_unmerge(plan)

    async def merge_record(self, merge_id: str) -> MergeRecord | None:
        """Return the merge record, or None."""
        return await self._merges.merge_record(merge_id)

    async def recover_merges(self) -> list[MergeRecord]:
        """Finish every interrupted merge and return the finished records.

        A record left ``APPLYING`` is reapplied from the rows it keeps. Those
        rows are checked before the record is settled. A record whose rows did
        not move stays ``APPLYING`` and is not returned.
        """
        return await self._merges.recover_merges()

    async def remove_episode_mentions(self, episode_id: str) -> list[str]:
        """Remove an episode's mentions and recount their entities."""
        return await self._entities.remove_episode_mentions(episode_id)

    async def mentions_of_episode(self, episode_id: str) -> list[Mention]:
        """Return one episode's mentions, ordered by mention id."""
        return await self._entities.mentions_of_episode(episode_id)

    async def has_merge_record(self, entity_id: str) -> bool:
        """Return whether any merge record names this entity."""
        return await self._entities.has_merge_record(entity_id)

    async def delete_orphans(self, entity_ids: Sequence[str]) -> list[str]:
        """Delete the given entities that are safe to delete."""
        return await self._entities.delete_orphans(entity_ids)

    async def set_entity_status(self, entity_id: str, status: EntityStatus) -> None:
        """Set the status without bumping the revision or changing `updated_at`."""
        return await self._entities.set_entity_status(entity_id, status)

    async def delete_entity(self, entity_id: str) -> None:
        """Delete an entity with its aliases, mentions and tombstone edges."""
        return await self._entities.delete_entity(entity_id)

    async def purge_snapshots(
        self, *, entity_id: str | None = None, before_seq: int | None = None
    ) -> int:
        """Blank merge snapshots and return how many records changed."""
        return await self._prune.purge_snapshots(
            entity_id=entity_id, before_seq=before_seq
        )

    async def next_revision(self) -> int:
        """Return the next value of the tenant's revision counter.

        The counter is read and written in one statement, so the value is
        never handed out twice. The statement is not idempotent: a replay
        would count the same revision twice, so the store refuses to replay
        it and raises `DatabaseUnavailableError` instead.
        """
        return await self._entities.next_revision()

    async def active_entities(self) -> list[Entity]:
        """Return the active entities, ordered by id."""
        return await self._entities.active_entities()

    async def all_entities(self) -> list[Entity]:
        """Return every entity status, ordered by id."""
        return await self._entities.all_entities()

    async def save_entity(
        self, entity: Entity, *, bump_revision: bool = True
    ) -> Entity:
        """Write one entity and return the stored copy."""
        return await self._entities.save_entity(entity, bump_revision=bump_revision)

    async def aliases_of(self, entity_id: str) -> list[Alias]:
        """Return the aliases of `entity_id`, ordered by id."""
        return await self._entities.aliases_of(entity_id)

    async def all_aliases(self) -> list[Alias]:
        """Return every alias in this tenant, ordered by alias id."""
        return await self._entities.all_aliases()

    async def mentions_of(self, entity_id: str) -> list[Mention]:
        """Return the mentions of `entity_id`, ordered by id."""
        return await self._entities.mentions_of(entity_id)

    async def prune_entity_text(self, entity_ids: Sequence[str]) -> list[str]:
        """Remove unsupported aliases and blank cached text of existing entities."""
        return await self._prune.prune_entity_text(entity_ids)
