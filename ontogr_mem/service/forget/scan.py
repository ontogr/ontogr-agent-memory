"""Check structural and indexed residue after a forget operation."""

from collections.abc import Sequence

from ontogr_mem.data_models.fact_versions import FactStatus
from ontogr_mem.llm.cache.port import ResponseCache
from ontogr_mem.resolution.keys import strict_key
from ontogr_mem.retrieval.keyword import KeywordKind
from ontogr_mem.service.forget.plan import ForgetPlan
from ontogr_mem.service.runtime import TenantRuntime
from ontogr_mem.temporal.query import TemporalQuery
from ontogr_mem.vectorstore.models import VectorKind


async def scan_residue(
    runtime: TenantRuntime,
    plan: ForgetPlan,
    *,
    needles: Sequence[str] = (),
    cache: ResponseCache | None = None,
    deleted_entity_ids: Sequence[str] = (),
    candidate_entity_ids: Sequence[str] = (),
) -> tuple[str, ...]:
    """Return sorted names of rows that still expose erased content.

    Args:
        runtime: Tenant stores and readers used by the checks.
        plan: Ids selected for the erase operation.
        needles: Source text snippets to search for. Empty skips text checks.
        cache: Optional response cache for text and tag checks.
        deleted_entity_ids: Entity ids deleted during the current run.
        candidate_entity_ids: Entities whose text may contain selected content.

    Returns:
        Sorted residue names. Structural checks run even when ``needles`` is
        empty, which is the recovery behavior.
    """
    residue: set[str] = set()
    temporal = runtime.stores.temporal_store
    episodes = runtime.stores.episodes
    resolver = runtime.stores.resolver_store
    vectors = runtime.stores.vectors
    episode_ids = set(plan.episode_ids)
    observation_ids = set(plan.observation_ids)
    fact_ids = set(plan.fact_ids)
    candidates = set(plan.entity_ids) | set(candidate_entity_ids)
    existing_candidates = {
        entity.entity_id for entity in await resolver.get_entities(sorted(candidates))
    }
    deleted_ids = set(deleted_entity_ids) | (set(plan.entity_ids) - existing_candidates)

    for episode_id in plan.episode_ids:
        if await temporal.observations_of_episode(episode_id):
            residue.add(f"observations:{episode_id}")
    if observation_ids and await temporal.observations_by_ids(sorted(observation_ids)):
        residue.add("observation_ids")

    facts = await temporal.facts_by_ids(sorted(fact_ids)) if fact_ids else []
    for fact in facts:
        if fact.current and observation_ids.intersection(fact.evidence):
            residue.add(f"evidence:{fact.fact_id}")
    withdrawn = [
        fact.fact_id
        for fact in facts
        if fact.current and fact.status is FactStatus.WITHDRAWN
    ]
    for fact_id in await vectors.exists(VectorKind.EDGE, withdrawn):
        residue.add(f"vector:edge:{fact_id}")
    if plan.episode_ids and await vectors.exists(VectorKind.EPISODE, plan.episode_ids):
        residue.add("vector:episode")
    if deleted_ids and await vectors.exists(VectorKind.NODE, sorted(deleted_ids)):
        residue.add("vector:node")
    for episode in await episodes.get_episodes(plan.episode_ids):
        residue.add(f"episode_row:{episode.id}")

    for alias in await resolver.all_aliases():
        if alias.first_episode_id not in episode_ids:
            continue
        remaining_mentions = await resolver.mentions_of(alias.entity_id)
        if not any(
            strict_key(mention.surface_form) == alias.strict_key
            for mention in remaining_mentions
        ):
            residue.add(f"alias:{alias.entity_id}")

    if needles:
        for fact in await runtime.reader.by_ids(
            sorted(fact_ids),
            TemporalQuery(now=runtime.clock.now(), include_history=True),
        ):
            if any(needle in fact.statement for needle in needles):
                residue.add("fact_statement")
                break
        for kind in (KeywordKind.ENTITY, KeywordKind.OBSERVATION, KeywordKind.EPISODE):
            for needle in needles:
                hits = await runtime.stores.keywords.search(kind, needle, limit=100)
                if any(
                    hit.id in episode_ids | fact_ids | observation_ids | candidates
                    for hit in hits
                ):
                    residue.add(f"keyword:{kind.value}")
                    break
        entities = await resolver.get_entities(sorted(candidates - deleted_ids))
        for entity in entities:
            text = " ".join(
                [
                    entity.canonical_name,
                    entity.summary,
                    *entity.descriptions,
                    str(entity.attributes),
                ]
            )
            if any(needle in text for needle in needles):
                residue.add(f"entity_text:{entity.entity_id}")
        if cache is not None:
            for needle in needles:
                if await cache.scan(needle):
                    residue.add("model_cache")
                    break

    if cache is not None:
        for episode_id in plan.episode_ids:
            if await cache.count_tag(f"episode:{episode_id}"):
                residue.add("model_cache:tag")
    return tuple(sorted(residue))
