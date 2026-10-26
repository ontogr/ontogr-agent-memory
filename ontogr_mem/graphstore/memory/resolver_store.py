"""In-memory twin of the resolver store.

The twin keeps the same graph shape as the adapters: entities, aliases,
mentions, entity verdicts, merge records and live or withdrawn ``FACT`` edges.
Tenant isolation comes from using one instance per tenant.
"""

import asyncio
import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from ontogr_mem.data_models.entities import (
    Alias,
    Entity,
    EntityStatus,
    EntityVerdict,
    Mention,
    MergeRecord,
    MergeState,
)
from ontogr_mem.data_models.timestamps import OPEN_MICROS
from ontogr_mem.graphstore.resolver.selection import deduplication_entities
from ontogr_mem.graphstore.resolver.text import (
    pruned_entity_text,
    purged_merge_snapshot,
)
from ontogr_mem.resolution.blocking import BlockingRow
from ontogr_mem.resolution.errors import (
    EntityHasFactsError,
    EpisodeAlreadyResolvedError,
    MergeRefusedError,
    NotFoundError,
    ResolutionError,
)
from ontogr_mem.resolution.keys import stripped_key
from ontogr_mem.resolution.resolver_store import (
    MergePlan,
    MergeRows,
    UnmergePlan,
    merge_intent,
    merge_rows_moved,
    merge_source_entity,
    snapshot_with_intent,
)


log = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class FactEdge:
    """A ``FACT`` edge in the twin.

    The twin does not write facts, but the resolver store needs to see them when
    it checks whether an entity is safe to delete. Tests seed these edges.

    Attributes:
        src_id: Id of the subject entity.
        dst_id: Id of the object entity.
        rel_type: Relation label.
        props: Edge properties, including ``status``.
    """

    src_id: str
    dst_id: str
    rel_type: str
    props: Mapping[str, Any] = field(default_factory=dict)

    @property
    def live(self) -> bool:
        """Return whether the edge is the current version and not withdrawn."""
        if self.props.get("status", "active") != "active":
            return False
        return self.props.get("expired_at", OPEN_MICROS) >= OPEN_MICROS


class MemoryResolverStore:
    """`ResolverStore` that keeps one tenant's resolution rows in memory.

    The instance is bound to one tenant. Every read and write returns copies, so
    a caller cannot mutate stored rows by accident.

    Attributes:
        resolve_lock: Serializes resolution runs that share this store in one
            event loop. Backends with transactions do not need it.
    """

    def __init__(self) -> None:
        """Create an empty store."""
        self._entities: dict[str, Entity] = {}
        self._aliases: dict[str, Alias] = {}
        self._mentions: dict[str, Mention] = {}
        self._verdicts: dict[str, EntityVerdict] = {}
        self._merges: dict[str, MergeRecord] = {}
        self._fact_edges: list[FactEdge] = []
        self._revision = 0
        self.resolve_lock = asyncio.Lock()

    async def episode_resolution_state(self, episode_id: str) -> bool:
        """Return True when resolution already wrote mentions for the episode."""
        return any(m.episode_id == episode_id for m in self._mentions.values())

    async def entity_id_for_ref(self, episode_id: str, local_id: str) -> str | None:
        """Return the entity named by an episode-local reference, or None."""
        return next(
            (
                mention.entity_id
                for mention in self._mentions.values()
                if mention.episode_id == episode_id and mention.local_id == local_id
            ),
            None,
        )

    async def has_entity(self, entity_id: str) -> bool:
        """Return whether this tenant contains the entity id."""
        return entity_id in self._entities

    async def entities_by_uri(self, uri: str) -> list[Entity]:
        """Return the entities carrying `uri`, in any status, ordered by id."""
        return [
            self._copy(entity)
            for entity in sorted(self._entities.values(), key=lambda e: e.entity_id)
            if entity.uri == uri
        ]

    async def entities_by_strict_key(self, strict_key: str) -> list[Entity]:
        """Return the entities an alias with `strict_key` resolves to."""
        found: dict[str, Entity] = {}
        for alias in self._aliases.values():
            if alias.strict_key != strict_key:
                continue
            entity = self._follow(alias.entity_id)
            if entity is not None:
                found[entity.entity_id] = entity
        return [self._copy(found[i]) for i in sorted(found)]

    async def entities_by_loose_key(self, loose_key: str) -> list[Entity]:
        """Return candidate entities an alias's loose key resolves to."""
        return await self.get_entities(
            [
                alias.entity_id
                for alias in self._aliases.values()
                if alias.loose_key == loose_key
            ],
            follow_merged=True,
        )

    async def get_entities(
        self, ids: Sequence[str], *, follow_merged: bool = False
    ) -> list[Entity]:
        """Return the entities with these ids, ordered by id."""
        found: dict[str, Entity] = {}
        for entity_id in ids:
            entity = self._entities.get(entity_id)
            if entity is None:
                continue
            if follow_merged:
                entity = self._follow(entity_id)
                if entity is None:
                    continue
            found[entity.entity_id] = entity
        return [self._copy(found[i]) for i in sorted(found)]

    async def blocking_rows(
        self, *, include_archived: bool = False
    ) -> list[BlockingRow]:
        """Return the blocking rows of the active entities, ordered by id."""
        wanted = {EntityStatus.ACTIVE}
        if include_archived:
            wanted.add(EntityStatus.ARCHIVED)
        rows: list[BlockingRow] = []
        for entity in sorted(self._entities.values(), key=lambda e: e.entity_id):
            if entity.status not in wanted:
                continue
            aliases = sorted(
                (
                    alias
                    for alias in self._aliases.values()
                    if alias.entity_id == entity.entity_id
                ),
                key=lambda a: a.alias_id,
            )
            rows.append(
                BlockingRow(
                    entity_id=entity.entity_id,
                    loose_key=entity.loose_key,
                    stripped_key=stripped_key(entity.canonical_name),
                    alias_keys=tuple(alias.loose_key for alias in aliases),
                )
            )
        return rows

    async def write_entity_batch(
        self, entities: Sequence[Entity], aliases: Sequence[Alias]
    ) -> list[Entity]:
        """Write entities and aliases and stamp each entity with a new revision."""
        stored: list[Entity] = []
        for entity in entities:
            row = entity.model_copy(
                deep=True, update={"revision": self._next_revision()}
            )
            self._entities[row.entity_id] = row
            stored.append(self._copy(row))
        for alias in aliases:
            self._aliases[alias.alias_id] = alias.model_copy(deep=True)
        return stored

    async def write_mentions(
        self, episode_id: str, mentions: Sequence[Mention]
    ) -> None:
        """Write an episode's mentions and recount the touched entities."""
        incoming = {mention.mention_id: mention for mention in mentions}
        existing = {
            mention.mention_id
            for mention in self._mentions.values()
            if mention.episode_id == episode_id
        }
        if existing - incoming.keys():
            raise EpisodeAlreadyResolvedError(episode_id)

        new_ids = incoming.keys() - existing
        if new_ids:
            seq = self._next_revision()
            for mention_id in sorted(new_ids):
                mention = incoming[mention_id]
                self._mentions[mention_id] = mention.model_copy(
                    deep=True, update={"episode_id": episode_id, "seq": seq}
                )
        self._recount(
            {
                self._mentions[mention_id].entity_id
                for mention_id in incoming
                if mention_id in self._mentions
            }
        )

    async def mentions_of_episode(self, episode_id: str) -> list[Mention]:
        """Return one episode's mentions, ordered by mention id."""
        return [
            mention.model_copy(deep=True)
            for _, mention in sorted(self._mentions.items())
            if mention.episode_id == episode_id
        ]

    async def has_merge_record(self, entity_id: str) -> bool:
        """Return whether any merge record names this entity."""
        return self._named_by_merge(entity_id)

    async def live_verdict(self, a_id: str, b_id: str) -> EntityVerdict | None:
        """Return the verdict for the pair unless a merge superseded it."""
        verdict = self._verdicts.get(f"{min(a_id, b_id)}|{max(a_id, b_id)}")
        if verdict is None or verdict.superseded_by is not None:
            return None
        return verdict.model_copy(deep=True)

    async def write_verdicts(self, verdicts: Sequence[EntityVerdict]) -> None:
        """Write verdicts, replacing rows with the same id."""
        for verdict in verdicts:
            self._verdicts[verdict.verdict_id] = verdict.model_copy(deep=True)

    async def list_for_deduplication(
        self, after_revision: int, provisional: bool, sweep_limit: int
    ) -> list[Entity]:
        """Return the active entities a deduplication run should consider."""
        return deduplication_entities(
            self._entities.values(), after_revision, provisional, sweep_limit
        )

    async def apply_merge(self, plan: MergePlan) -> MergeRecord:
        """Apply one merge and return the stored record.

        The rows are written in order, like the graph adapter: the record is
        written ``APPLYING`` first and keeps the rows of the merge, then the
        rows move, then the record is written ``APPLIED``. An interrupted
        merge is finished by `recover_merges`.

        Raises:
            MergeRefusedError: If either entity is no longer active, or the
                merge was already applied.
        """
        self._require_active(plan.survivor.entity_id)
        self._require_active(plan.absorbed.entity_id)
        if plan.record.merge_id in self._merges:
            raise MergeRefusedError("already_applied", plan.record.merge_id)
        record = plan.record.model_copy(update={"seq": self._next_revision()})
        rows = MergeRows.from_merge_plan(plan)
        applying = record.model_copy(
            update={
                "state": MergeState.APPLYING,
                "snapshot": snapshot_with_intent(record.snapshot, rows),
            }
        )
        self._merges[applying.merge_id] = applying.model_copy(deep=True)
        self._apply_rows(rows)
        applied = applying.model_copy(update={"state": MergeState.APPLIED})
        self._merges[applied.merge_id] = applied.model_copy(deep=True)
        return applied.model_copy(deep=True)

    async def apply_unmerge(self, plan: UnmergePlan) -> MergeRecord:
        """Apply one unmerge and return the stored record.

        The record is written ``APPLYING`` with the unmerge rows before the
        rows move back. The record is then written ``REVERTED``. The saved rows
        let recovery finish an interrupted unmerge.

        Raises:
            MergeRefusedError: If the merge record does not exist.
        """
        if plan.record.merge_id not in self._merges:
            raise MergeRefusedError("missing", plan.record.merge_id)
        stored = plan.record.model_copy(update={"seq": self._next_revision()})
        rows = MergeRows.from_unmerge_plan(plan)
        applying = stored.model_copy(
            update={
                "state": MergeState.APPLYING,
                "snapshot": snapshot_with_intent(stored.snapshot, rows),
            }
        )
        self._merges[applying.merge_id] = applying.model_copy(deep=True)
        self._apply_rows(rows)
        reverted = applying.model_copy(update={"state": MergeState.REVERTED})
        self._merges[reverted.merge_id] = reverted.model_copy(deep=True)
        return reverted.model_copy(deep=True)

    async def merge_record(self, merge_id: str) -> MergeRecord | None:
        """Return the merge record, or None."""
        record = self._merges.get(merge_id)
        return None if record is None else record.model_copy(deep=True)

    async def recover_merges(self) -> list[MergeRecord]:
        """Finish every interrupted merge and return the finished records.

        A record left ``APPLYING`` is played out again from the rows it
        keeps, and the rows are checked before the record is settled. A
        record whose rows did not move stays ``APPLYING`` and is not
        returned.
        """
        finished: list[MergeRecord] = []
        for _, record in sorted(self._merges.items()):
            if record.state is not MergeState.APPLYING:
                continue
            settled = self._settle_merge(record)
            if settled is not None:
                finished.append(settled)
        return finished

    def _settle_merge(self, record: MergeRecord) -> MergeRecord | None:
        """Play out one interrupted merge and settle its record."""
        rows = merge_intent(record.snapshot)
        if rows is None:
            log.warning(
                "merge %s keeps no rows and cannot be finished", record.merge_id
            )
            return None
        self._apply_rows(rows)
        if not self._rows_moved(rows):
            log.warning(
                "merge %s did not move its rows and stays applying", record.merge_id
            )
            return None
        settled = record.model_copy(update={"state": rows.state})
        self._merges[settled.merge_id] = settled.model_copy(deep=True)
        return settled

    def _rows_moved(self, rows: MergeRows) -> bool:
        """Return whether the rows of a merge left the entity they must leave."""
        source = merge_source_entity(rows)
        mentions = {
            mention.mention_id: mention.entity_id
            for mention in self._mentions.values()
            if mention.entity_id == source
        }
        aliases = {
            alias.alias_id: alias.entity_id
            for alias in self._aliases.values()
            if alias.entity_id == source
        }
        return merge_rows_moved(
            rows,
            absorbed=self._entities.get(rows.absorbed.entity_id),
            mention_entities=mentions,
            alias_entities=aliases,
        )

    def _apply_rows(self, rows: MergeRows) -> None:
        """Write the rows of one merge or unmerge."""
        self._entities[rows.survivor.entity_id] = rows.survivor.model_copy(deep=True)
        self._entities[rows.absorbed.entity_id] = rows.absorbed.model_copy(deep=True)
        for entity in rows.redirected:
            self._entities[entity.entity_id] = entity.model_copy(deep=True)
        for mention in rows.mentions:
            self._mentions[mention.mention_id] = mention.model_copy(deep=True)
        for alias in rows.aliases:
            self._aliases[alias.alias_id] = alias.model_copy(deep=True)
        for verdict in rows.verdicts:
            self._verdicts[verdict.verdict_id] = verdict.model_copy(deep=True)

    async def remove_episode_mentions(self, episode_id: str) -> list[str]:
        """Remove an episode's mentions and recount their entities."""
        doomed = [
            mention.mention_id
            for mention in self._mentions.values()
            if mention.episode_id == episode_id
        ]
        touched = sorted({self._mentions[i].entity_id for i in doomed})
        for mention_id in doomed:
            del self._mentions[mention_id]
        self._recount(set(touched))
        return touched

    async def delete_orphans(self, entity_ids: Sequence[str]) -> list[str]:
        """Delete the given entities that are safe to delete."""
        deleted: list[str] = []
        for entity_id in sorted(set(entity_ids)):
            entity = self._entities.get(entity_id)
            if entity is None:
                continue
            if any(m.entity_id == entity_id for m in self._mentions.values()):
                continue
            if self._has_live_fact(entity_id):
                continue
            if self._named_by_merge(entity_id):
                continue
            self._delete_rows(entity_id)
            deleted.append(entity_id)
        return deleted

    async def set_entity_status(self, entity_id: str, status: EntityStatus) -> None:
        """Set the status without bumping the revision or ``updated_at``."""
        entity = self._entities.get(entity_id)
        if entity is None:
            raise NotFoundError(f"entity {entity_id!r}")
        self._entities[entity_id] = entity.model_copy(update={"status": status})

    async def delete_entity(self, entity_id: str) -> None:
        """Delete an entity with its aliases, mentions and tombstone edges."""
        if entity_id not in self._entities:
            raise NotFoundError(f"entity {entity_id!r}")
        if self._has_live_fact(entity_id):
            raise EntityHasFactsError(entity_id)
        self._delete_rows(entity_id)

    async def purge_snapshots(
        self, *, entity_id: str | None = None, before_seq: int | None = None
    ) -> int:
        """Blank merge snapshots and return how many records changed."""
        if entity_id is None and before_seq is None:
            raise ResolutionError("purge_snapshots needs entity_id or before_seq")
        changed = 0
        for merge_id, record in sorted(self._merges.items()):
            updated = purged_merge_snapshot(
                record, entity_id=entity_id, before_seq=before_seq
            )
            if updated is None:
                continue
            self._merges[merge_id] = updated
            changed += 1
        return changed

    async def next_revision(self) -> int:
        """Return the next value of the tenant's revision counter."""
        return self._next_revision()

    async def active_entities(self) -> list[Entity]:
        """Return the active entities, ordered by id."""
        return [
            self._copy(entity)
            for entity in sorted(self._entities.values(), key=lambda e: e.entity_id)
            if entity.status is EntityStatus.ACTIVE
        ]

    async def all_entities(self) -> list[Entity]:
        """Return every entity in any status, ordered by id."""
        return [
            self._copy(entity)
            for entity in sorted(self._entities.values(), key=lambda e: e.entity_id)
        ]

    async def save_entity(
        self, entity: Entity, *, bump_revision: bool = True
    ) -> Entity:
        """Write one entity and return the stored copy."""
        row = entity.model_copy(deep=True)
        if bump_revision:
            row = row.model_copy(update={"revision": self._next_revision()})
        self._entities[row.entity_id] = row
        return self._copy(row)

    async def aliases_of(self, entity_id: str) -> list[Alias]:
        """Return the aliases of `entity_id`, ordered by id."""
        return [
            alias.model_copy(deep=True)
            for _, alias in sorted(self._aliases.items())
            if alias.entity_id == entity_id
        ]

    async def all_aliases(self) -> list[Alias]:
        """Return every alias in this tenant, ordered by alias id."""
        return [
            alias.model_copy(deep=True) for _, alias in sorted(self._aliases.items())
        ]

    async def mentions_of(self, entity_id: str) -> list[Mention]:
        """Return the mentions of `entity_id`, ordered by id."""
        return [
            mention.model_copy(deep=True)
            for _, mention in sorted(self._mentions.items())
            if mention.entity_id == entity_id
        ]

    async def prune_entity_text(self, entity_ids: Sequence[str]) -> list[str]:
        """Remove unsupported aliases and blank cached text of existing entities."""
        pruned: list[str] = []
        for entity_id in sorted(set(entity_ids)):
            entity = self._entities.get(entity_id)
            if entity is None:
                continue

            mentions = await self.mentions_of(entity_id)
            aliases = await self.aliases_of(entity_id)
            updated, retained, new_aliases = pruned_entity_text(
                entity, mentions, aliases
            )
            retained.extend(new_aliases)

            retained_ids = {row.alias_id for row in retained}
            for row in aliases:
                if row.alias_id not in retained_ids:
                    del self._aliases[row.alias_id]
            for row in retained:
                self._aliases[row.alias_id] = row.model_copy(deep=True)

            if updated != entity:
                await self.save_entity(updated)
            pruned.append(entity_id)
        return pruned

    def seed_fact_edge(
        self,
        src_id: str,
        dst_id: str,
        rel_type: str,
        props: Mapping[str, Any],
    ) -> None:
        """Add a ``FACT`` edge for delete-guard tests.

        Args:
            src_id: Subject entity id.
            dst_id: Object entity id.
            rel_type: Relation label.
            props: Edge properties. A ``status`` other than ``active`` marks the
                edge withdrawn.
        """
        self._fact_edges.append(FactEdge(src_id, dst_id, rel_type, dict(props)))

    def fact_edge_count(self, entity_id: str, *, live: bool) -> int:
        """Count the live or withdrawn ``FACT`` edges that touch `entity_id`.

        Args:
            entity_id: Entity to count edges for.
            live: Count live edges when true, withdrawn edges when false.

        Returns:
            Number of matching edges.
        """
        return sum(
            1
            for edge in self._fact_edges
            if entity_id in (edge.src_id, edge.dst_id) and edge.live is live
        )

    def seed_merge_record(self, record: MergeRecord) -> None:
        """Store a merge record as it is, for recovery tests.

        Args:
            record: The record to store. Its state decides whether
                `recover_merges` settles it.
        """
        self._merges[record.merge_id] = record.model_copy(deep=True)

    def _next_revision(self) -> int:
        self._revision += 1
        return self._revision

    def _copy(self, entity: Entity) -> Entity:
        return entity.model_copy(deep=True)

    def _follow(self, entity_id: str) -> Entity | None:
        seen: set[str] = set()
        current = self._entities.get(entity_id)
        while current is not None and current.merged_into is not None:
            if current.entity_id in seen:
                return current
            seen.add(current.entity_id)
            current = self._entities.get(current.merged_into)
        return current

    def _require_active(self, entity_id: str) -> None:
        entity = self._entities.get(entity_id)
        if entity is None or entity.status is not EntityStatus.ACTIVE:
            raise MergeRefusedError("not_active", entity_id)

    def _has_live_fact(self, entity_id: str) -> bool:
        return any(
            entity_id in (edge.src_id, edge.dst_id) and edge.live
            for edge in self._fact_edges
        )

    def _named_by_merge(self, entity_id: str) -> bool:
        return any(
            entity_id in (record.survivor_id, record.absorbed_id)
            for record in self._merges.values()
        )

    def _recount(self, entity_ids: set[str]) -> None:
        counts: dict[str, int] = {}
        for mention in self._mentions.values():
            counts[mention.entity_id] = counts.get(mention.entity_id, 0) + 1
        for entity_id in entity_ids:
            entity = self._entities.get(entity_id)
            if entity is not None:
                self._entities[entity_id] = entity.model_copy(
                    update={"mention_count": counts.get(entity_id, 0)}
                )

    def _delete_rows(self, entity_id: str) -> None:
        self._entities.pop(entity_id, None)
        for alias_id in [
            alias_id
            for alias_id, alias in self._aliases.items()
            if alias.entity_id == entity_id
        ]:
            del self._aliases[alias_id]
        for mention_id in [
            mention_id
            for mention_id, mention in self._mentions.items()
            if mention.entity_id == entity_id
        ]:
            del self._mentions[mention_id]
        self._fact_edges = [
            edge
            for edge in self._fact_edges
            if entity_id not in (edge.src_id, edge.dst_id) or edge.live
        ]
