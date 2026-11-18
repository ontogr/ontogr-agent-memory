"""Keyword lane over observations and episodes."""

import asyncio

from ontogr_mem.retrieval.gate import FactGate
from ontogr_mem.retrieval.keyword import KeywordKind, KeywordSearch
from ontogr_mem.retrieval.lanes import LaneOutput
from ontogr_mem.retrieval.lanes._shared import lane_record, run_search, widen
from ontogr_mem.retrieval.settings import RetrievalSettings
from ontogr_mem.temporal.query import TemporalQuery
from ontogr_mem.temporal.reader import FactReader


async def run_keyword_lane(
    keywords: KeywordSearch,
    facts: FactReader,
    *,
    query: str,
    temporal_query: TemporalQuery,
    limit: int,
    settings: RetrievalSettings,
    disputed: str = "flag",
    gate: FactGate | None = None,
) -> LaneOutput:
    """Find observations and episodes, mapping evidence hits to facts."""
    last_success = None

    async def observation_candidates(window: int):
        nonlocal last_success
        hits = await keywords.search(KeywordKind.OBSERVATION, query, limit=window)
        observation_ids = tuple(dict.fromkeys(hit.id for hit in hits))
        views = (
            await facts.by_observations(observation_ids, temporal_query)
            if observation_ids
            else []
        )
        positions = {hit.id: rank for rank, hit in enumerate(hits)}
        view_ranks = {
            (view.fact_id, view.version): min(
                (positions[obs_id] for obs_id in view.evidence if obs_id in positions),
                default=len(positions),
            )
            for view in views
        }
        views.sort(
            key=lambda view: (
                view_ranks[(view.fact_id, view.version)],
                view.fact_id,
                view.version,
            )
        )
        if gate is not None:
            views = await gate.admit(views)
        if disputed == "exclude":
            views = [view for view in views if not view.disputed]
        result = (hits, views, len(hits), window)
        last_success = result
        return result

    candidate_limit = settings.candidate_factor * limit
    observations, episodes = await asyncio.gather(
        run_search(
            lambda: widen(
                observation_candidates,
                lambda result: len(result[1]),
                lambda result: result[2] == result[3],
                start=candidate_limit,
                want=limit,
                rechecks=settings.filter_rechecks if gate is not None else 0,
            )
        ),
        run_search(
            lambda: keywords.search(KeywordKind.EPISODE, query, limit=candidate_limit)
        ),
    )
    failures = [
        (name, result.error)
        for name, result in (("observation", observations), ("episode", episodes))
        if result.error is not None
    ]
    observation_result = observations.value
    if observation_result is None and observations.error is not None:
        observation_result = last_success
    observation_hits = observation_result[0] if observation_result else []
    episode_hits = episodes.value or []
    observation_ids = tuple(dict.fromkeys(hit.id for hit in observation_hits))
    views = observation_result[1] if observation_result else []
    episode_ids = tuple(dict.fromkeys(hit.id for hit in episode_hits))
    record = lane_record(
        "keyword", bool(views or episode_ids), failures, search_count=2
    )
    return LaneOutput(
        record,
        facts=tuple(views),
        episodes=episode_ids,
        observation_ids=observation_ids,
        error=failures[0][1] if failures else None,
    )
