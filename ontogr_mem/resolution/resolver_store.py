"""The storage port the resolver writes through.

Methods take and return domain objects. The tenant is fixed when the store is
built, so no method takes a tenant or a query.
"""

import asyncio
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from ontogr_mem.data_models.entities import (
    Alias,
    Entity,
    EntityStatus,
    EntityVerdict,
    Mention,
    MergeRecord,
    MergeState,
)
from ontogr_mem.resolution.blocking import BlockingRow


INTENT_KEY = "intent"
"""Key under which a merge record keeps the rows of its merge."""


@dataclass(frozen=True, slots=True)
class MergePlan:
    """Everything one merge writes, built by the pure merge planner.

    The store stamps ``record.seq`` from `next_revision` and writes the rows
    in order. The store keeps a copy of the rows in the merge record, so a
    merge that was interrupted can be played out again.

    Attributes:
        record: The merge record. Its ``seq`` is filled by the store.
        survivor: Survivor with the absorbed contributions applied.
        absorbed: Absorbed entity, marked ``MERGED`` and pointing at the survivor.
        mentions: Mentions moved to the survivor.
        aliases: Aliases moved to the survivor.
        redirected: Merged entities whose pointer now targets the survivor.
        superseded_verdicts: Verdicts this merge makes no longer live.
        created_verdicts: Verdicts rewritten for the survivor.
        replaced_verdicts: Earlier versions of `created_verdicts`.
    """

    record: MergeRecord
    survivor: Entity
    absorbed: Entity
    mentions: tuple[Mention, ...] = ()
    aliases: tuple[Alias, ...] = ()
    redirected: tuple[Entity, ...] = ()
    superseded_verdicts: tuple[EntityVerdict, ...] = ()
    created_verdicts: tuple[EntityVerdict, ...] = ()
    replaced_verdicts: tuple[EntityVerdict, ...] = ()


@dataclass(frozen=True, slots=True)
class UnmergePlan:
    """Everything one unmerge writes, built by the pure merge planner.

    Attributes:
        record: The merge record, in the ``REVERTED`` state.
        survivor: Survivor restored to its state before the merge.
        absorbed: Absorbed entity, active again and no longer pointing anywhere.
        mentions: Mentions moved back to the absorbed entity.
        aliases: Alias rows restored from the snapshot.
        redirected: Merged entities whose pointer targets the absorbed entity
            again.
        superseded_verdicts: Verdicts the merge created and the unmerge drops.
        restored_verdicts: Verdicts the merge superseded and the unmerge revives.
        replaced_verdicts: Verdict rows the merge overwrote.
        verdict: User ``DISTINCT`` verdict for the pair, or None.
    """

    record: MergeRecord
    survivor: Entity
    absorbed: Entity
    mentions: tuple[Mention, ...] = ()
    aliases: tuple[Alias, ...] = ()
    redirected: tuple[Entity, ...] = ()
    superseded_verdicts: tuple[EntityVerdict, ...] = ()
    restored_verdicts: tuple[EntityVerdict, ...] = ()
    replaced_verdicts: tuple[EntityVerdict, ...] = ()
    verdict: EntityVerdict | None = None


@dataclass(frozen=True, slots=True)
class MergeRows:
    """The rows one merge or unmerge writes.

    A store keeps these rows with the merge record. A merge that was
    interrupted can then be played out again, and the rows can be checked
    before the record is settled.

    Attributes:
        state: The state the merge ends in.
        survivor: The survivor after the merge.
        absorbed: The absorbed entity after the merge.
        redirected: Entities whose pointer the merge retargeted.
        mentions: Mentions the merge moved.
        aliases: Aliases the merge moved.
        verdicts: Verdicts the merge wrote.
    """

    state: MergeState
    survivor: Entity
    absorbed: Entity
    redirected: tuple[Entity, ...] = ()
    mentions: tuple[Mention, ...] = ()
    aliases: tuple[Alias, ...] = ()
    verdicts: tuple[EntityVerdict, ...] = ()

    @classmethod
    def from_merge_plan(cls, plan: MergePlan) -> "MergeRows":
        """Return the rows one merge plan writes."""
        return cls(
            state=MergeState.APPLIED,
            survivor=plan.survivor,
            absorbed=plan.absorbed,
            redirected=tuple(plan.redirected),
            mentions=tuple(plan.mentions),
            aliases=tuple(plan.aliases),
            verdicts=(
                *plan.superseded_verdicts,
                *plan.created_verdicts,
                *plan.replaced_verdicts,
            ),
        )

    @classmethod
    def from_unmerge_plan(cls, plan: UnmergePlan) -> "MergeRows":
        """Return the rows one unmerge plan writes."""
        return cls(
            state=MergeState.REVERTED,
            survivor=plan.survivor,
            absorbed=plan.absorbed,
            redirected=tuple(plan.redirected),
            mentions=tuple(plan.mentions),
            aliases=tuple(plan.aliases),
            verdicts=(
                *plan.superseded_verdicts,
                *plan.restored_verdicts,
                *plan.replaced_verdicts,
                *((plan.verdict,) if plan.verdict else ()),
            ),
        )

    def to_json(self) -> dict[str, Any]:
        """Return the rows as plain JSON data."""
        return {
            "state": self.state.value,
            "survivor": self.survivor.model_dump(mode="json"),
            "absorbed": self.absorbed.model_dump(mode="json"),
            "redirected": [
                entity.model_dump(mode="json") for entity in self.redirected
            ],
            "mentions": [mention.model_dump(mode="json") for mention in self.mentions],
            "aliases": [alias.model_dump(mode="json") for alias in self.aliases],
            "verdicts": [verdict.model_dump(mode="json") for verdict in self.verdicts],
        }

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> "MergeRows":
        """Return the rows stored as JSON data.

        Raises:
            ValueError: If the data holds no readable rows.
        """
        return cls(
            state=MergeState(data["state"]),
            survivor=Entity.model_validate(data["survivor"]),
            absorbed=Entity.model_validate(data["absorbed"]),
            redirected=tuple(
                Entity.model_validate(entity) for entity in data.get("redirected", ())
            ),
            mentions=tuple(
                Mention.model_validate(mention) for mention in data.get("mentions", ())
            ),
            aliases=tuple(
                Alias.model_validate(alias) for alias in data.get("aliases", ())
            ),
            verdicts=tuple(
                EntityVerdict.model_validate(verdict)
                for verdict in data.get("verdicts", ())
            ),
        )


def merge_source_entity(rows: MergeRows) -> str:
    """Return the entity the rows of a merge must leave.

    Args:
        rows: The rows of the merge.

    Returns:
        The id of the entity the moved rows must no longer point at.
    """
    if rows.state is MergeState.APPLIED:
        return rows.absorbed.entity_id
    return rows.survivor.entity_id


def merge_rows_moved(
    rows: MergeRows,
    *,
    absorbed: Entity | None,
    mention_entities: Mapping[str, str],
    alias_entities: Mapping[str, str],
) -> bool:
    """Return whether a merge left the state it should leave.

    Args:
        rows: The rows of the merge.
        absorbed: The absorbed entity as it is stored now.
        mention_entities: The entity id of every mention, by mention id.
        alias_entities: The entity id of every alias, by alias id.

    Returns:
        True when the absorbed entity carries the state of the merge and
        every moved row left the entity it had to leave.
    """
    if absorbed is None:
        return False
    if rows.state is MergeState.APPLIED:
        settled = (
            absorbed.status is EntityStatus.MERGED
            and absorbed.merged_into == rows.survivor.entity_id
        )
    else:
        settled = (
            absorbed.status is EntityStatus.ACTIVE and absorbed.merged_into is None
        )
    if not settled:
        return False
    source = merge_source_entity(rows)
    mentions_left = any(
        mention_entities.get(mention.mention_id) == source for mention in rows.mentions
    )
    aliases_left = any(
        alias_entities.get(alias.alias_id) == source for alias in rows.aliases
    )
    return not mentions_left and not aliases_left


def merge_intent(snapshot: Mapping[str, Any] | None) -> MergeRows | None:
    """Return the rows a merge record keeps, or None.

    Args:
        snapshot: The snapshot of a merge record.

    Returns:
        The rows of the merge, or None when the snapshot holds no readable
        rows.
    """
    if not snapshot:
        return None
    raw = snapshot.get(INTENT_KEY)
    if not isinstance(raw, Mapping):
        return None
    try:
        return MergeRows.from_json(raw)
    except (KeyError, ValueError):
        return None


def snapshot_with_intent(
    snapshot: Mapping[str, Any] | None, rows: MergeRows
) -> dict[str, Any]:
    """Return the snapshot with the rows of the merge kept in it.

    Args:
        snapshot: The snapshot the planner built.
        rows: The rows the merge writes.

    Returns:
        The snapshot with the rows under the ``intent`` key.
    """
    kept = dict(snapshot or {})
    kept[INTENT_KEY] = rows.to_json()
    return kept


class ResolverStore(Protocol):
    """Storage of one tenant's entities, aliases, mentions and merges."""

    @property
    def resolve_lock(self) -> asyncio.Lock:
        """Lock shared by resolvers using this store instance."""
        ...

    async def episode_resolution_state(self, episode_id: str) -> bool:
        """Return True when resolution already wrote mentions for the episode."""
        ...

    async def entity_id_for_ref(self, episode_id: str, local_id: str) -> str | None:
        """Return the entity named by an episode-local reference, or None."""
        ...

    async def has_entity(self, entity_id: str) -> bool:
        """Return whether this tenant contains the entity id."""
        ...

    async def entities_by_uri(self, uri: str) -> list[Entity]:
        """Return the entities carrying ``uri``, in any status, ordered by id."""
        ...

    async def entities_by_strict_key(self, strict_key: str) -> list[Entity]:
        """Return the entities an alias with ``strict_key`` resolves to.

        ``merged_into`` is followed, so a merged entity's alias returns its
        survivor. The result is ordered by id and has no duplicates.
        """
        ...

    async def entities_by_loose_key(self, loose_key: str) -> list[Entity]:
        """Return candidate entities for an accent-insensitive alias key.

        Follow merges and return distinct entities ordered by id. A loose-key
        match is a retrieval candidate, never proof of shared identity.
        """
        ...

    async def get_entities(
        self, ids: Sequence[str], *, follow_merged: bool = False
    ) -> list[Entity]:
        """Return the entities with these ids, ordered by id.

        Missing ids are omitted. With ``follow_merged`` a merged entity is
        replaced by the entity its pointer chain ends at.
        """
        ...

    async def blocking_rows(
        self, *, include_archived: bool = False
    ) -> list[BlockingRow]:
        """Return the blocking rows of the active entities, ordered by id.

        With ``include_archived`` the archived entities are returned too, so a
        new mention can find and restore them.
        """
        ...

    async def write_entity_batch(
        self, entities: Sequence[Entity], aliases: Sequence[Alias]
    ) -> list[Entity]:
        """Write entities and aliases and stamp each entity with a new revision.

        Writing the same entities again stores the same fields under a new
        revision. Returns the stored entities in input order.
        """
        ...

    async def write_mentions(
        self, episode_id: str, mentions: Sequence[Mention]
    ) -> None:
        """Write an episode's mentions and recount the touched entities.

        Mentions are the commit marker of an episode. The write is refused when
        the episode already has a mention whose id is not in ``mentions``, which
        means another or a partial run wrote the episode. Writing the same ids
        again is idempotent. Every mention of the write carries the store
        revision of that write in its ``seq``.

        Raises:
            EpisodeAlreadyResolvedError: If the episode has other mentions.
        """
        ...

    async def mentions_of_episode(self, episode_id: str) -> list[Mention]:
        """Return one episode's mentions, ordered by mention id."""
        ...

    async def prune_entity_text(self, entity_ids: Sequence[str]) -> list[str]:
        """Remove unsupported names and clear cached text on existing entities.

        Aliases without a remaining mention are removed. If the canonical name
        has no remaining mention, it is replaced by the most frequent mention
        surface form, with ties broken by the lowest mention id. The canonical
        name's strict and loose keys and alias are updated with it. Summaries,
        descriptions and attributes are cleared, and the summary is marked
        stale. Missing entity ids are ignored. Returns the existing entity ids
        in sorted order.
        """
        ...

    async def has_merge_record(self, entity_id: str) -> bool:
        """Return whether any merge record names this entity."""
        ...

    async def live_verdict(self, a_id: str, b_id: str) -> EntityVerdict | None:
        """Return the verdict for the pair unless a merge superseded it."""
        ...

    async def write_verdicts(self, verdicts: Sequence[EntityVerdict]) -> None:
        """Write verdicts, replacing rows with the same id."""
        ...

    async def list_for_deduplication(
        self, after_revision: int, provisional: bool, sweep_limit: int
    ) -> list[Entity]:
        """Return the active entities a deduplication run should consider.

        The result contains the entities with ``revision`` greater than
        ``after_revision``; when ``provisional`` is true, the provisional ones;
        and up to ``sweep_limit`` further entities in ascending
        ``(last_checked_version, entity_id)`` order. It is ordered by
        ``(revision, entity_id)``.
        """
        ...

    async def apply_merge(self, plan: MergePlan) -> MergeRecord:
        """Apply one merge and return the stored record.

        The rows are written in order, not in one transaction. The record is
        written ``APPLYING`` first and keeps the rows of the merge, then the
        entities, mentions, aliases and verdicts move, then the record is
        written ``APPLIED``. A merge that was interrupted in between is
        finished by `recover_merges`.

        The record's ``seq`` is stamped from `next_revision`. The write is
        refused when either entity is no longer active, and when the merge
        was already applied. The record is created with a compare-and-swap,
        so two processes cannot apply the same merge. Merges of one tenant
        are serialized within one process only; two processes can still merge
        the same pair under two different merge records.

        Raises:
            MergeRefusedError: If the survivor or the absorbed entity is not
                active, or the merge was already applied.
        """
        ...

    async def apply_unmerge(self, plan: UnmergePlan) -> MergeRecord:
        """Apply one unmerge and return the stored record.

        The record is written ``APPLYING`` with the unmerge rows before the
        rows move back. The record is then written ``REVERTED``. The saved rows
        let `recover_merges` finish an interrupted unmerge.

        Raises:
            MergeRefusedError: If the merge record does not exist.
        """
        ...

    async def merge_record(self, merge_id: str) -> MergeRecord | None:
        """Return the merge record, or None."""
        ...

    async def recover_merges(self) -> list[MergeRecord]:
        """Finish every interrupted merge and return the finished records.

        A record left ``APPLYING`` by an interrupted merge or unmerge is
        played out again from the rows it keeps. The rows are checked before
        the record is settled, so a merge whose rows did not move stays
        ``APPLYING`` and is not returned.
        """
        ...

    async def remove_episode_mentions(self, episode_id: str) -> list[str]:
        """Remove an episode's mentions and recount their entities.

        Returns the distinct entity ids that lost a mention, ordered by id.
        """
        ...

    async def delete_orphans(self, entity_ids: Sequence[str]) -> list[str]:
        """Delete the given entities that are safe to delete.

        An entity is an orphan when it has no mentions, no live fact edge and no
        merge record. Its ``WITHDRAWN`` fact edges are deleted with it. Returns
        the deleted ids, ordered by id.
        """
        ...

    async def set_entity_status(self, entity_id: str, status: EntityStatus) -> None:
        """Set the status without bumping the revision or changing ``updated_at``.

        Raises:
            NotFoundError: If the entity does not exist.
        """
        ...

    async def delete_entity(self, entity_id: str) -> None:
        """Delete an entity with its aliases, mentions and tombstone edges.

        Refuses an entity that still has a live fact edge, so a delete never
        removes a fact.

        Raises:
            NotFoundError: If the entity does not exist.
            EntityHasFactsError: If the entity still has a live fact edge.
        """
        ...

    async def purge_snapshots(
        self, *, entity_id: str | None = None, before_seq: int | None = None
    ) -> int:
        """Blank merge snapshots and return how many records changed.

        ``entity_id`` limits the purge to records naming that entity;
        ``before_seq`` limits it to records older than that revision. At least
        one must be given.

        Raises:
            ResolutionError: If both arguments are None.
        """
        ...

    async def next_revision(self) -> int:
        """Return the next value of the tenant's revision counter.

        Every call hands out a new value. A backend that cannot tell whether
        the counter was incremented fails the call instead of handing out the
        same value twice.
        """
        ...

    async def active_entities(self) -> list[Entity]:
        """Return the active entities, ordered by id."""
        ...

    async def all_entities(self) -> list[Entity]:
        """Return every entity status, ordered by id."""
        ...

    async def save_entity(
        self, entity: Entity, *, bump_revision: bool = True
    ) -> Entity:
        """Write one entity and return the stored copy.

        With ``bump_revision`` the entity is stamped with a new revision.
        """
        ...

    async def aliases_of(self, entity_id: str) -> list[Alias]:
        """Return the aliases of ``entity_id``, ordered by id."""
        ...

    async def all_aliases(self) -> list[Alias]:
        """Return every alias in this tenant, ordered by alias id."""
        ...

    async def mentions_of(self, entity_id: str) -> list[Mention]:
        """Return the mentions of ``entity_id``, ordered by id."""
        ...
