"""Run resumable removal steps over a tenant runtime.

Entity ids remain tied to their original Episode ids; Erase does not clean
append-only ledgers, audit history, consolidation output, or external copies.
"""

from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, replace
from typing import Literal, TypeVar
from uuid import uuid4

from ontogr_mem.data_models.clock import Clock
from ontogr_mem.data_models.fact_versions import Fact, FactStatus
from ontogr_mem.llm.cache.port import ResponseCache
from ontogr_mem.relational_store.jobs.erase_journal import (
    EraseJournal,
    ErasureRecord,
    ErasureStatus,
)
from ontogr_mem.resolution.keys import entity_text
from ontogr_mem.service.contracts import ForgetMode
from ontogr_mem.service.forget.plan import ForgetPlan
from ontogr_mem.service.forget.results import ForgetResult, RemovedCounts
from ontogr_mem.service.forget.scan import scan_residue
from ontogr_mem.service.runtime import TenantRuntime
from ontogr_mem.vectorstore.fact_vectors import FactVectorObserver
from ontogr_mem.vectorstore.models import EmbedItem, NodeExtras, VectorKind


RuntimeFactory = Callable[[str], Awaitable[TenantRuntime]]
TenantOperation = Callable[[str], Awaitable[None]]
WorkflowMode = ForgetMode | Literal["clear"]
T = TypeVar("T")


@dataclass(frozen=True, slots=True)
class ForgetState:
    """Inputs and accumulated results of one resumable cleanup run."""

    record: ErasureRecord
    runtime: TenantRuntime
    plan: ForgetPlan
    needles: tuple[str, ...]
    recovered: bool
    mode: ForgetMode | None
    observation_count: int
    facts_before: tuple[Fact, ...]
    target_entity_ids: frozenset[str]
    counts: RemovedCounts = RemovedCounts()
    deleted_entities: tuple[str, ...] = ()
    blanked_entity_ids: tuple[str, ...] = ()
    residue: tuple[str, ...] = ()

    @property
    def is_clear(self) -> bool:
        """Whether cleanup removes the entire tenant graph."""
        return self.record.mode == "clear"

    @property
    def scans_text(self) -> bool:
        """Whether cleanup checks the remaining text for residue."""
        return self.mode is ForgetMode.ERASE or self.is_clear


class EraseWorkflow:
    """Apply idempotent forget steps and recover runs left in progress.

    The temporal engine owns fact retraction, while the resolver store owns
    mention cleanup. Both are needed because the two record sets are separate.
    """

    def __init__(
        self,
        journal: EraseJournal,
        runtime_factory: RuntimeFactory,
        clock: Clock,
        *,
        cache: ResponseCache | None = None,
        drop_graph: TenantOperation | None = None,
        delete_tenant_jobs: Callable[[str], Awaitable[int]] | None = None,
        after_step: Callable[[str], None] | None = None,
    ) -> None:
        """Bind persistence, tenant runtimes and optional cleanup boundaries."""
        self._journal = journal
        self._runtime_factory = runtime_factory
        self._clock = clock
        self._cache = cache
        self._drop_graph = drop_graph
        self._delete_tenant_jobs = delete_tenant_jobs
        self._after_step = after_step or (lambda _: None)

    async def run(
        self,
        principal_id: str,
        tenant: str,
        plan: ForgetPlan,
        mode: WorkflowMode,
        *,
        text_scanned_needles: Sequence[str] = (),
    ) -> ForgetResult:
        """Journal and run deletion, memory-only removal, or Erase.

        ``INVALIDATE`` is a temporal write and is handled by the memory service.

        Erase also searches for stored observation statements and quotes that
        contain at least 12 non-whitespace characters.

        Args:
            principal_id: Caller recorded on the erasure journal row.
            tenant: Tenant whose stores are changed.
            plan: Resolved target and related Fact and Entity ids.
            mode: Deletion, memory-only removal, or Erase.
            text_scanned_needles: Additional source text to check during Erase.

        Raises:
            ValueError: If ``mode`` is not supported by this workflow.
        """
        if mode not in {
            ForgetMode.DELETE,
            ForgetMode.MEMORY_ONLY,
            ForgetMode.ERASE,
            "clear",
        }:
            raise ValueError(f"unsupported workflow mode: {mode}")
        mode_value = mode.value if isinstance(mode, ForgetMode) else mode
        self._after_step("before_tombstone")
        record = ErasureRecord(
            erasure_id=uuid4().hex,
            tenant=tenant,
            mode=mode_value,
            principal_id=principal_id,
            selector_kind=plan.selector_kind,
            target=plan.target,
            episode_ids=plan.episode_ids,
            fact_ids=plan.fact_ids,
            observation_ids=plan.observation_ids,
            entity_ids=plan.entity_ids,
            created_at=self._clock.now(),
        )
        await self._journal.create(record)
        return await self._execute(
            record,
            plan,
            tuple(text_scanned_needles),
            recovered=False,
        )

    async def recover(self, tenant: str | None = None) -> list[ForgetResult]:
        """Resume all running journal rows, oldest first."""
        results = []
        for record in await self._journal.running(tenant):
            plan = ForgetPlan(
                selector_kind=record.selector_kind,
                episode_ids=record.episode_ids,
                observation_ids=record.observation_ids,
                fact_ids=record.fact_ids,
                entity_ids=record.entity_ids,
                collateral_fact_ids=(),
                reopened_fact_ids=(),
                target=record.target,
            )
            results.append(await self._execute(record, plan, (), recovered=True))
        return results

    async def _execute(
        self,
        record: ErasureRecord,
        plan: ForgetPlan,
        needles: Sequence[str],
        *,
        recovered: bool,
    ) -> ForgetResult:
        """Replay the cleanup steps and close the journal row."""
        state = await self._prepare(record, plan, needles, recovered=recovered)
        for action in (self._temporal_step, self._derived_step, self._residue_step):
            state = await action(state)
        return await self._finish(state)

    async def _prepare(
        self,
        record: ErasureRecord,
        plan: ForgetPlan,
        needles: Sequence[str],
        *,
        recovered: bool,
    ) -> ForgetState:
        """Capture source records before any cleanup step changes them."""
        runtime = await self._runtime_factory(record.tenant)
        is_clear = record.mode == "clear"
        mode = None if is_clear else ForgetMode(record.mode)
        scans_text = mode is ForgetMode.ERASE or is_clear
        temporal = runtime.stores.temporal_store
        if recovered:
            plan = await _restore_affected_fact_ids(runtime, plan)
        observations_before = [
            observation
            for episode_id in plan.episode_ids
            for observation in await temporal.observations_of_episode(episode_id)
        ]
        if scans_text and not recovered:
            candidate_needles = list(needles)
            for observation in observations_before:
                candidate_needles.extend((observation.statement, observation.quote))
            needles = tuple(
                sorted(
                    {
                        text.strip()
                        for text in candidate_needles
                        if len(text.strip()) >= 12
                    }
                )
            )
        facts_before = await temporal.facts_by_ids(plan.fact_ids)
        target_entity_ids = {
            mention.entity_id
            for episode_id in plan.episode_ids
            for mention in await runtime.stores.resolver_store.mentions_of_episode(
                episode_id
            )
        }
        target_episode_ids = set(plan.episode_ids)
        target_entity_ids.update(
            alias.entity_id
            for alias in await runtime.stores.resolver_store.all_aliases()
            if alias.first_episode_id in target_episode_ids
        )
        return ForgetState(
            record=record,
            runtime=runtime,
            plan=plan,
            needles=tuple(needles),
            recovered=recovered,
            mode=mode,
            observation_count=len(observations_before),
            facts_before=tuple(facts_before),
            target_entity_ids=frozenset(target_entity_ids),
        )

    async def _step(
        self, state: ForgetState, name: str, action: Callable[[], Awaitable[T]]
    ) -> T:
        """Record each successful boundary before invoking the recovery hook."""
        result = await action()
        await self._journal.step(state.record.erasure_id, name)
        self._after_step(name)
        return result

    async def _temporal_step(self, state: ForgetState) -> ForgetState:
        """Remove temporal evidence and count changed fact identities."""
        runtime, plan = state.runtime, state.plan
        if state.mode is ForgetMode.MEMORY_ONLY:
            await self._step(
                state, "reset", lambda: self._reset(runtime, plan.episode_ids)
            )
        else:
            await self._step(
                state, "retract", lambda: self._retract(runtime, plan.episode_ids)
            )
        facts_after = await runtime.stores.temporal_store.facts_by_ids(plan.fact_ids)
        return replace(
            state,
            counts=replace(
                state.counts,
                facts=_changed_facts(state.facts_before, facts_after),
                observations=state.observation_count,
            ),
        )

    async def _derived_step(self, state: ForgetState) -> ForgetState:
        """Clean vectors, entity text, cache entries and source episodes in order."""
        runtime, plan = state.runtime, state.plan
        vector_count = await self._step(
            state,
            "index",
            lambda: self._index(
                runtime,
                plan,
                remove_episode_vectors=state.mode is not ForgetMode.MEMORY_ONLY,
                clear_all=state.is_clear,
            ),
        )
        deleted_entities, blanked_entity_ids, node_vector_count = await self._step(
            state,
            "entities",
            lambda: self._entities(
                runtime, plan, set(state.target_entity_ids), reembed=not state.is_clear
            ),
        )
        vector_count += node_vector_count
        cache_count = await self._step(
            state,
            "cache",
            lambda: self._evict(
                plan.episode_ids,
                state.needles,
                ForgetMode.ERASE
                if state.scans_text
                else (state.mode or ForgetMode.DELETE),
            ),
        )
        episode_count = 0
        if state.mode is not ForgetMode.MEMORY_ONLY:
            episode_count = await self._step(
                state,
                "episodes",
                lambda: self._remove_episodes(runtime, plan.episode_ids),
            )
        return replace(
            state,
            deleted_entities=deleted_entities,
            blanked_entity_ids=blanked_entity_ids,
            counts=replace(
                state.counts,
                episodes=episode_count,
                entities=len(deleted_entities),
                vector_documents=vector_count,
                cache_entries=cache_count,
                entity_text_updates=len(blanked_entity_ids),
            ),
        )

    async def _residue_step(self, state: ForgetState) -> ForgetState:
        """Check residue before dropping tenant storage and side rows."""
        residue: tuple[str, ...] = ()
        if state.scans_text:
            residue = await self._step(
                state,
                "scan",
                lambda: scan_residue(
                    state.runtime,
                    state.plan,
                    needles=() if state.recovered else state.needles,
                    cache=self._cache,
                    deleted_entity_ids=state.deleted_entities,
                    candidate_entity_ids=tuple(sorted(state.target_entity_ids)),
                ),
            )
        if state.is_clear:
            drop_graph = self._drop_graph
            delete_tenant_jobs = self._delete_tenant_jobs
            if drop_graph is None or delete_tenant_jobs is None:
                raise RuntimeError("clear requires graph and job-store lifecycle ports")
            await self._step(
                state, "drop_graph", lambda: drop_graph(state.record.tenant)
            )
            await self._step(
                state, "side_rows", lambda: delete_tenant_jobs(state.record.tenant)
            )
        return replace(state, residue=residue)

    async def _finish(self, state: ForgetState) -> ForgetResult:
        """Close the journal once and report the cleanup boundary."""
        status = ErasureStatus.FAILED if state.residue else ErasureStatus.COMPLETED
        counts = state.counts
        encoded_counts = tuple(
            sorted(
                (
                    ("episodes", counts.episodes),
                    ("facts", counts.facts),
                    ("observations", counts.observations),
                    ("entities", counts.entities),
                    ("vector_documents", counts.vector_documents),
                    ("cache_entries", counts.cache_entries),
                    ("entity_text_updates", counts.entity_text_updates),
                )
            )
        )
        await self._journal.finish(
            state.record.erasure_id,
            status,
            state.residue,
            encoded_counts,
            self._clock.now(),
        )
        not_cleaned = [
            "consolidation output",
            "ledger hashes",
            "audit and operation rows",
            "copies outside this system",
        ]
        if self._cache is None:
            not_cleaned.append("model cache")
        return ForgetResult(
            erasure_id=state.record.erasure_id,
            status=status.value,
            removed=counts,
            residue=state.residue,
            not_cleaned=tuple(not_cleaned),
            text_scanned=state.mode is ForgetMode.ERASE and not state.recovered,
            blanked_entity_ids=state.blanked_entity_ids,
        )

    async def _retract(
        self, runtime: TenantRuntime, episode_ids: Sequence[str]
    ) -> None:
        """Retract selected observations and settle their affected slots."""
        ontology = await runtime.ontology.catalog()
        await runtime.engine(ontology, runtime.resolver.store).retract_episodes(
            episode_ids
        )

    async def _reset(self, runtime: TenantRuntime, episode_ids: Sequence[str]) -> None:
        """Remove temporal facts and mentions while keeping Episode rows."""
        ontology = await runtime.ontology.catalog()
        engine = runtime.engine(ontology, runtime.resolver.store)
        for episode_id in sorted(set(episode_ids)):
            await engine.reset_episode(episode_id, runtime.stores.resolver_store)

    async def _index(
        self,
        runtime: TenantRuntime,
        plan: ForgetPlan,
        *,
        remove_episode_vectors: bool,
        clear_all: bool,
    ) -> int:
        """Remove stale vectors and rewrite surviving current fact vectors."""
        temporal = runtime.stores.temporal_store
        affected_fact_ids = tuple(
            sorted(
                set(plan.fact_ids)
                | set(plan.collateral_fact_ids)
                | set(plan.reopened_fact_ids)
            )
        )
        versions = await temporal.facts_by_ids(affected_fact_ids)
        current = [fact for fact in versions if fact.current]
        withdrawn_ids = [
            fact.fact_id for fact in current if fact.status is FactStatus.WITHDRAWN
        ]
        if clear_all:
            return await _delete_all_vectors(runtime)
        vector_count = await runtime.stores.vectors.delete(
            VectorKind.EDGE, withdrawn_ids
        )
        survivors = [
            fact for fact in current if fact.status is not FactStatus.WITHDRAWN
        ]
        observation_ids = sorted(
            {observation_id for fact in survivors for observation_id in fact.evidence}
        )
        observations = await temporal.observations_by_ids(observation_ids)
        by_id = {observation.obs_id: observation for observation in observations}
        vector_count += await FactVectorObserver(runtime.writer).rewrite(
            survivors, by_id
        )
        if remove_episode_vectors:
            vector_count += await runtime.stores.vectors.delete(
                VectorKind.EPISODE, plan.episode_ids
            )
        return vector_count

    async def _entities(
        self,
        runtime: TenantRuntime,
        plan: ForgetPlan,
        target_entity_ids: set[str],
        *,
        reembed: bool,
    ) -> tuple[tuple[str, ...], tuple[str, ...], int]:
        """Remove mentions, delete orphans, and prune surviving entity text."""
        resolver = runtime.stores.resolver_store
        touched: set[str] = set()
        for episode_id in plan.episode_ids:
            touched.update(await resolver.remove_episode_mentions(episode_id))
        candidates = touched | target_entity_ids | set(plan.entity_ids)
        deleted = tuple(await resolver.delete_orphans(sorted(candidates)))
        vector_count = await runtime.stores.vectors.delete(VectorKind.NODE, deleted)
        blanked = tuple(await resolver.prune_entity_text(sorted(candidates)))
        entities = await resolver.get_entities(blanked)
        items = [
            EmbedItem(
                entity.entity_id,
                entity_text(entity),
                NodeExtras(status=entity.status.value),
            )
            for entity in entities
        ]
        if items and reembed:
            result = await runtime.stores.vectors.upsert(VectorKind.NODE, items)
            vector_count += result.written
        return deleted, blanked, vector_count

    async def _evict(
        self,
        episode_ids: Sequence[str],
        needles: Sequence[str],
        mode: ForgetMode,
    ) -> int:
        """Evict episode-tagged cache entries and optional text matches."""
        if self._cache is None:
            return 0
        removed = 0
        for episode_id in episode_ids:
            removed += await self._cache.evict_tag(f"episode:{episode_id}")
        if mode is ForgetMode.ERASE:
            for needle in needles:
                removed += await self._cache.evict_text(needle)
        return removed

    async def _remove_episodes(
        self, runtime: TenantRuntime, episode_ids: Sequence[str]
    ) -> int:
        """Delete the selected Episode rows after their derived rows are gone."""
        existing = await runtime.stores.episodes.get_episodes(episode_ids)
        for episode in existing:
            await runtime.stores.episodes.remove_episode(episode.id)
        return len(existing)


def _changed_facts(before: Sequence[Fact], after: Sequence[Fact]) -> int:
    """Count planned Fact ids whose stored versions changed during this run."""
    before_by_id = _versions_by_fact_id(before)
    after_by_id = _versions_by_fact_id(after)
    return sum(
        before_by_id.get(fact_id, ()) != after_by_id.get(fact_id, ())
        for fact_id in before_by_id.keys() | after_by_id.keys()
    )


def _versions_by_fact_id(facts: Sequence[Fact]) -> dict[str, tuple[Fact, ...]]:
    """Group Fact versions by id in stable version order."""
    grouped: dict[str, list[Fact]] = {}
    for fact in facts:
        grouped.setdefault(fact.fact_id, []).append(fact)
    return {
        fact_id: tuple(sorted(versions, key=lambda fact: fact.version))
        for fact_id, versions in grouped.items()
    }


async def _restore_affected_fact_ids(
    runtime: TenantRuntime, plan: ForgetPlan
) -> ForgetPlan:
    """Find collateral and reopened Facts from the journaled Fact slots."""
    facts = await runtime.stores.temporal_store.facts_by_ids(plan.fact_ids)
    slot_keys = sorted({fact.slot_key for fact in facts})
    target_fact_ids = set(plan.fact_ids)
    collateral: set[str] = set(plan.collateral_fact_ids)
    reopened: set[str] = set(plan.reopened_fact_ids)
    for slot_key in slot_keys:
        slot = await runtime.stores.temporal_store.load_slot(slot_key)
        collateral.update(
            fact.fact_id
            for fact in slot.facts
            if fact.current and fact.fact_id not in target_fact_ids
        )
        supersessions = await runtime.stores.temporal_store.supersessions_of_slot(
            slot_key
        )
        reopened.update(
            supersession.old_fact_id
            for supersession in supersessions
            if supersession.source_episode_id in plan.episode_ids
            and supersession.old_fact_id not in target_fact_ids
        )
    return replace(
        plan,
        collateral_fact_ids=tuple(sorted(collateral)),
        reopened_fact_ids=tuple(sorted(reopened)),
    )


async def _delete_all_vectors(runtime: TenantRuntime) -> int:
    """Remove every vector, paging through each configured vector kind."""
    removed = 0
    for kind in VectorKind:
        after = None
        while True:
            ids = await runtime.stores.vectors.iter_ids(kind, after, 500)
            if not ids:
                break
            removed += await runtime.stores.vectors.delete(kind, ids)
            after = ids[-1]
            if len(ids) < 500:
                break
    return removed
