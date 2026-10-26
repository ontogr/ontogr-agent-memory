"""Read-only forget-plan construction."""

from collections.abc import Sequence
from dataclasses import dataclass

from ontogr_mem.data_models.fact_versions import FactStatus
from ontogr_mem.ingestion.store import EpisodeStore
from ontogr_mem.resolution.resolver_store import ResolverStore
from ontogr_mem.service.errors import ObjectNotFoundError
from ontogr_mem.service.forget.confirm import target_digest
from ontogr_mem.service.forget.selector import ForgetSelector, validate_selector
from ontogr_mem.temporal.temporal_store import SourceSelector, TemporalStore


@dataclass(frozen=True, slots=True)
class ForgetPlan:
    """Resolved Episodes and the derived records a forget operation may touch.

    Attributes:
        selector_kind: Field used to resolve the target.
        episode_ids: Existing Episodes selected by the request.
        observation_ids: Observations owned by the selected Episodes.
        fact_ids: Facts with evidence in the selected Observations.
        entity_ids: Entities with no remaining mention or current Fact evidence.
        collateral_fact_ids: Other current Facts in the selected Fact slots.
        reopened_fact_ids: Older Facts closed by a selected Episode.
        target: Digest of the selector kind and resolved Episode ids.
    """

    selector_kind: str
    episode_ids: tuple[str, ...]
    observation_ids: tuple[str, ...]
    fact_ids: tuple[str, ...]
    entity_ids: tuple[str, ...]
    collateral_fact_ids: tuple[str, ...]
    reopened_fact_ids: tuple[str, ...]
    target: str


async def build_plan(
    selector: ForgetSelector,
    *,
    episodes: EpisodeStore,
    temporal: TemporalStore,
    resolver: ResolverStore,
) -> ForgetPlan:
    """Resolve a selector and calculate its evidence, collateral, and target.

    The function reads its ports only. It does not mutate stores or issue a
    confirmation token.

    Args:
        selector: Exactly one forget target.
        episodes: Episode metadata and record reader.
        temporal: Observation and Fact reader.
        resolver: Entity and mention reader.

    Returns:
        A deterministic plan whose ids are sorted and unique.

    Raises:
        InvalidArgumentError: If `selector` does not name exactly one target.
        ObjectNotFoundError: If a requested Fact does not exist.
    """
    selector = validate_selector(selector)
    candidate_ids = await _episode_ids(selector, episodes, temporal)
    existing = await episodes.get_episodes(candidate_ids)
    episode_ids = tuple(sorted({episode.id for episode in existing}))

    observations = [
        observation
        for episode_id in episode_ids
        for observation in await temporal.observations_of_episode(episode_id)
    ]
    observation_ids = tuple(
        sorted({observation.obs_id for observation in observations})
    )
    target_observation_ids = set(observation_ids)
    slot_keys = sorted({observation.slot_key for observation in observations})
    slots = [await temporal.load_slot(slot_key) for slot_key in slot_keys]
    slot_facts = [fact for slot in slots for fact in slot.facts]
    fact_ids = tuple(
        sorted(
            {
                fact.fact_id
                for fact in slot_facts
                if target_observation_ids.intersection(fact.evidence)
            }
        )
    )
    target_fact_ids = set(fact_ids)
    collateral_fact_ids = tuple(
        sorted(
            {
                fact.fact_id
                for fact in slot_facts
                if fact.current
                and fact.status is not FactStatus.WITHDRAWN
                and fact.fact_id not in target_fact_ids
            }
        )
    )
    selected_episodes = set(episode_ids)
    reopened_fact_ids_set: set[str] = set()
    for slot_key in slot_keys:
        for supersession in await temporal.supersessions_of_slot(slot_key):
            if (
                supersession.source_episode_id in selected_episodes
                and supersession.old_fact_id not in target_fact_ids
            ):
                reopened_fact_ids_set.add(supersession.old_fact_id)
    entity_ids = await _orphan_candidates(
        episode_ids, target_fact_ids, resolver, temporal
    )
    return ForgetPlan(
        selector_kind=selector.kind,
        episode_ids=episode_ids,
        observation_ids=observation_ids,
        fact_ids=fact_ids,
        entity_ids=entity_ids,
        collateral_fact_ids=collateral_fact_ids,
        reopened_fact_ids=tuple(sorted(reopened_fact_ids_set)),
        target=target_digest(selector.kind, episode_ids),
    )


async def _episode_ids(
    selector: ForgetSelector,
    episodes: EpisodeStore,
    temporal: TemporalStore,
) -> Sequence[str]:
    """Resolve one validated selector to episode candidates."""
    match selector.kind:
        case "episode_ids":
            return selector.episode_ids or ()
        case "subject_ref":
            return await episodes.episode_ids_by_subject(selector.subject_ref or "")
        case "source_system":
            return await temporal.episodes_matching(
                SourceSelector(system=selector.source_system or "")
            )
        case "fact_id":
            facts = await temporal.facts_by_ids([selector.fact_id or ""])
            if not facts:
                raise ObjectNotFoundError("fact not found")
            observation_ids = sorted(
                {obs_id for fact in facts for obs_id in fact.evidence}
            )
            observations = await temporal.observations_by_ids(observation_ids)
            return sorted({observation.episode_id for observation in observations})
        case _:
            return ()


async def _orphan_candidates(
    episode_ids: Sequence[str],
    target_fact_ids: set[str],
    resolver: ResolverStore,
    temporal: TemporalStore,
) -> tuple[str, ...]:
    """Return mentioned entities with no mentions or current Facts left behind."""
    target_episodes = set(episode_ids)
    mentions = [
        mention
        for episode_id in episode_ids
        for mention in await resolver.mentions_of_episode(episode_id)
    ]
    candidates = sorted({mention.entity_id for mention in mentions})
    if not candidates:
        return ()

    surviving_mentions = {
        mention.entity_id
        for entity_id in candidates
        for mention in await resolver.mentions_of(entity_id)
        if mention.episode_id not in target_episodes
    }
    facts = await temporal.facts_touching(candidates)
    surviving_facts = {
        entity_id
        for fact in facts
        if fact.current
        and fact.status is not FactStatus.WITHDRAWN
        and fact.fact_id not in target_fact_ids
        for entity_id in (fact.subject_id, fact.object_id)
        if entity_id in candidates
    }
    protected_by_merge: set[str] = set()
    for entity_id in candidates:
        if await resolver.has_merge_record(entity_id):
            protected_by_merge.add(entity_id)
    return tuple(
        sorted(
            set(candidates) - surviving_mentions - surviving_facts - protected_by_merge
        )
    )
