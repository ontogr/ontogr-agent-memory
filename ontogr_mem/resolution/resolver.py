"""`EntityResolver`: link extracted entities to canonical entities."""

import asyncio
from collections.abc import Sequence
from dataclasses import dataclass, field

from ontogr_mem.data_models import Clock, ExtractedEntity
from ontogr_mem.data_models.entities import (
    Alias,
    DecisionMethod,
    Entity,
    EntityStatus,
    Mention,
    MergeRecord,
    Outcome,
    VerdictKind,
)
from ontogr_mem.llm import LLMClient
from ontogr_mem.resolution.candidates import candidate_pass
from ontogr_mem.resolution.config import Budget, ResolverSettings
from ontogr_mem.resolution.decision import collect_matches
from ontogr_mem.resolution.deduplication import ConsolidationReport, run_deduplication
from ontogr_mem.resolution.errors import (
    EpisodeAlreadyResolvedError,
    NotFoundError,
    SnapshotPurgedError,
    UnmergeBlockedError,
)
from ontogr_mem.resolution.grouping import Group, build_groups
from ontogr_mem.resolution.inline_tier import inline_verify
from ontogr_mem.resolution.keys import entity_text
from ontogr_mem.resolution.merge import (
    build_merge_plan,
    build_unmerge_plan,
    is_user,
)
from ontogr_mem.resolution.plan import Plan
from ontogr_mem.resolution.policy import ResolutionPolicy, TypeRelations
from ontogr_mem.resolution.resolver_store import ResolverStore
from ontogr_mem.resolution.scoring import Candidate
from ontogr_mem.resolution.verdicts import apply_verdicts
from ontogr_mem.resolution.verifier import Verifier
from ontogr_mem.resolution.views import view_for
from ontogr_mem.resolution.write import write_group
from ontogr_mem.vectorstore import (
    EmbedItem,
    NodeExtras,
    VectorKind,
    VectorStore,
)


@dataclass(frozen=True, slots=True)
class Decision:
    """The resolution decision of one mention.

    Attributes:
        mention_id: Mention the decision belongs to.
        entity_id: Entity the mention is linked to.
        method: How the mention was linked.
        outcome: Whether the entity was matched or created.
        score: Score that decided the link, if any.
        reason: Stable reason code.
        candidates: Candidate ids the decision considered.
    """

    mention_id: str
    entity_id: str
    method: DecisionMethod
    outcome: Outcome
    score: float | None = None
    reason: str = ""
    candidates: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class ResolutionResult:
    """What one `resolve` call wrote.

    Attributes:
        entities: Entities that were matched or created, in batch order.
        aliases: Aliases written with those entities.
        mentions: Mentions written for the batch.
        decisions: The decision stored on each mention.
    """

    entities: list[Entity] = field(default_factory=list)
    aliases: list[Alias] = field(default_factory=list)
    mentions: list[Mention] = field(default_factory=list)
    decisions: list[Decision] = field(default_factory=list)


class EntityResolver:
    """Resolve, embed and look up the entities of one store.

    Args:
        store: Storage of the tenant's entities, aliases, mentions and merges.
        vectors: Vector port that holds one vector per entity.
        policy: Per-type resolution policy.
        settings: Resolver thresholds and version.
        clock: Time source for every timestamp the resolver writes.
        llm: Optional client for the inline verification tier.
        relations: Type ancestry used when comparing labels.
    """

    def __init__(
        self,
        store: ResolverStore,
        vectors: VectorStore,
        policy: ResolutionPolicy,
        settings: ResolverSettings,
        clock: Clock,
        *,
        llm: LLMClient | None = None,
        relations: TypeRelations | None = None,
    ) -> None:
        """Wire the resolver; `llm` enables inline verification."""
        self.store = store
        self.vectors = vectors
        self.policy = policy
        self.settings = settings
        self.clock = clock
        self.relations = relations or TypeRelations()
        self._verifier = Verifier(llm, settings) if llm is not None else None
        self._lock = store.resolve_lock

    @property
    def verifier(self) -> Verifier | None:
        """Return the pair verifier, or None when inline verification is off."""
        return self._verifier

    async def resolve(self, entities: Sequence[ExtractedEntity]) -> ResolutionResult:
        """Link a batch of extracted entities and write their mentions."""
        batch = list(entities)
        if not batch:
            return ResolutionResult()
        async with self._lock:
            groups = build_groups(batch, self.policy, self.relations)
            if not groups:
                return ResolutionResult()
            episode_ids = sorted(
                {member.episode_id for group in groups for member in group.members}
            )
            states = await asyncio.gather(
                *(
                    self.store.episode_resolution_state(episode_id)
                    for episode_id in episode_ids
                )
            )
            for episode_id, resolved in zip(episode_ids, states, strict=True):
                if resolved:
                    raise EpisodeAlreadyResolvedError(episode_id)
            candidates, incomplete = await candidate_pass(
                groups,
                store=self.store,
                vectors=self.vectors,
                policy=self.policy,
                settings=self.settings,
            )
            verified: dict[int, dict[str, VerdictKind]] = {}
            if self.settings.inline_verification and self._verifier is not None:
                verified = await inline_verify(
                    groups,
                    candidates,
                    store=self.store,
                    policy=self.policy,
                    settings=self.settings,
                    relations=self.relations,
                    verifier=self._verifier,
                    incomplete=incomplete,
                    clock=self.clock,
                )
            return await self._write(groups, candidates, verified, incomplete)

    async def _write(
        self,
        groups: Sequence[Group],
        candidates: Sequence[Sequence[Candidate]],
        verified: dict[int, dict[str, VerdictKind]],
        incomplete: bool,
    ) -> ResolutionResult:
        entities: list[Entity] = []
        aliases: list[Alias] = []
        mentions: list[Mention] = []
        decisions: list[Decision] = []
        by_episode: dict[str, list[Mention]] = {}
        for index, group in enumerate(groups):
            group_candidates = candidates[index]
            view = await view_for(self.store, group, group_candidates)
            collected = collect_matches(
                group,
                group_candidates,
                view,
                policy=self.policy,
                settings=self.settings,
                relations=self.relations,
            )
            plan: Plan = apply_verdicts(collected, verified.get(index, {}), incomplete)
            entity, written_aliases, written_mentions = await write_group(
                self.store,
                group,
                plan,
                settings=self.settings,
                relations=self.relations,
                clock=self.clock,
                incomplete=incomplete,
            )
            entities.append(entity)
            aliases.extend(written_aliases)
            await self.embed(entity)
            for mention in written_mentions:
                mentions.append(mention)
                by_episode.setdefault(mention.episode_id, []).append(mention)
                decisions.append(_decision(mention))
        for episode_id in sorted(by_episode):
            await self.store.write_mentions(episode_id, by_episode[episode_id])
        return ResolutionResult(
            entities=entities, aliases=aliases, mentions=mentions, decisions=decisions
        )

    async def merge(
        self,
        survivor_id: str,
        absorbed_id: str,
        *,
        actor: str = "user",
        reason: str = "",
    ) -> MergeRecord:
        """Absorb one entity into another and return the merge record."""
        if not is_user(actor):
            verdict = await self.store.live_verdict(survivor_id, absorbed_id)
            if verdict is not None and verdict.verdict is VerdictKind.DISTINCT:
                raise UnmergeBlockedError(
                    f"user verdict keeps {survivor_id} and {absorbed_id} apart"
                )
        plan = await build_merge_plan(
            self.store,
            self.policy,
            self.relations,
            survivor_id,
            absorbed_id,
            clock=self.clock,
            actor=actor,
            reason=reason,
        )
        return await self.store.apply_merge(plan)

    async def unmerge(
        self, merge_id: str, *, actor: str = "user", reason: str = ""
    ) -> MergeRecord:
        """Reverse an applied merge from its snapshot."""
        record = await self.store.merge_record(merge_id)
        if record is None:
            raise NotFoundError(f"merge {merge_id!r}")
        if record.snapshot_purged or record.snapshot is None:
            raise SnapshotPurgedError(merge_id)
        stored = await self.store.get_entities([record.survivor_id, record.absorbed_id])
        survivor = next((e for e in stored if e.entity_id == record.survivor_id), None)
        absorbed = next((e for e in stored if e.entity_id == record.absorbed_id), None)
        if survivor is None or absorbed is None:
            raise NotFoundError(merge_id)
        mentions = await self.store.mentions_of(record.survivor_id)
        moved = [
            mention
            for mention in mentions
            if mention.mention_id in record.moved_mention_ids
        ]
        aliases = await self.store.aliases_of(record.survivor_id)
        moved_aliases = [
            alias for alias in aliases if alias.alias_id in record.moved_alias_ids
        ]
        updated = record.model_copy(update={"actor": actor, "reason": reason})
        plan = build_unmerge_plan(
            updated,
            survivor,
            absorbed,
            moved,
            moved_aliases,
            clock=self.clock,
        )
        return await self.store.apply_unmerge(plan)

    async def purge_snapshots(
        self, *, entity_id: str | None = None, before_seq: int | None = None
    ) -> int:
        """Blank merge snapshots that are no longer needed for an unmerge."""
        return await self.store.purge_snapshots(
            entity_id=entity_id, before_seq=before_seq
        )

    async def remove_episode_mentions(self, episode_id: str) -> list[str]:
        """Remove an episode's mentions, delete orphan entities and their vectors."""
        touched = await self.store.remove_episode_mentions(episode_id)
        deleted = await self.store.delete_orphans(touched)
        if deleted:
            await self.vectors.delete(VectorKind.NODE, deleted)
        return deleted

    async def get_entities(
        self, ids: Sequence[str], *, follow_merged: bool = False
    ) -> list[Entity]:
        """Return the entities with these ids; see ``ResolverStore.get_entities``."""
        return await self.store.get_entities(ids, follow_merged=follow_merged)

    async def save_entity(
        self, entity: Entity, *, bump_revision: bool = True
    ) -> Entity:
        """Write one entity and return the stored copy."""
        return await self.store.save_entity(entity, bump_revision=bump_revision)

    async def archive(self, entity_id: str) -> None:
        """Archive an entity so it is readable but no longer matched."""
        await self.store.set_entity_status(entity_id, EntityStatus.ARCHIVED)

    async def delete(self, entity_id: str) -> None:
        """Delete an entity and its vector."""
        await self.store.delete_entity(entity_id)
        await self.vectors.delete(VectorKind.NODE, [entity_id])

    async def active_entities(self) -> list[Entity]:
        """Return the active entities, ordered by id."""
        return await self.store.active_entities()

    async def deduplicate(
        self, *, plan_only: bool = False, budget: Budget | None = None
    ) -> ConsolidationReport:
        """Run one deduplication pass over the tenant's active entities."""
        return await run_deduplication(
            self, plan_only=plan_only, budget=budget or Budget()
        )

    async def embed(self, entity: Entity) -> None:
        """Write the entity vector if it is missing or stale."""
        item = EmbedItem(
            entity.entity_id,
            entity_text(entity),
            NodeExtras(status=entity.status.value),
        )
        missing = await self.vectors.needs_embedding(VectorKind.NODE, [item])
        if missing:
            await self.vectors.upsert(VectorKind.NODE, missing)


def _decision(mention: Mention) -> Decision:
    return Decision(
        mention_id=mention.mention_id,
        entity_id=mention.entity_id,
        method=mention.decision_method,
        outcome=mention.decision_outcome,
        score=mention.decision_score,
        reason=mention.decision_reason,
        candidates=tuple(mention.decision_candidates),
    )
