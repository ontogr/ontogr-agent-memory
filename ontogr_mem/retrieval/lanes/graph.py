"""Capped breadth-first expansion from retrieval seed entities."""

from collections.abc import Sequence

from ontogr_mem.retrieval.lanes import LaneOutput
from ontogr_mem.retrieval.lanes._shared import lane_record, run_search
from ontogr_mem.retrieval.models import LaneRecord, LaneStatus
from ontogr_mem.retrieval.settings import RetrievalSettings
from ontogr_mem.retrieval.text import terms
from ontogr_mem.temporal.query import FactView, TemporalQuery
from ontogr_mem.temporal.reader import FactReader


async def run_graph_lane(
    facts: FactReader,
    *,
    seeds: Sequence[str],
    query: TemporalQuery,
    graph_depth: int,
    settings: RetrievalSettings,
    query_text: str = "",
    disputed: str = "flag",
) -> LaneOutput:
    """Walk visible facts breadth-first, abandoning results if any hop fails."""
    if not seeds:
        return LaneOutput(LaneRecord("graph", LaneStatus.SKIPPED))

    frontier = tuple(dict.fromkeys(seeds))
    seen_entities = set(frontier)
    found: dict[tuple[str, int], tuple[int, FactView]] = {}
    query_terms = set(terms(query_text))
    for hop in range(1, graph_depth + 1):
        result = await run_search(
            lambda current_frontier=frontier: facts.touching(current_frontier, query)
        )
        if result.error is not None:
            record = lane_record(
                "graph", False, [("touching", result.error)], search_count=1
            )
            return LaneOutput(record, error=result.error)
        views = result.value or []
        next_entities: set[str] = set()
        for view in views:
            if disputed == "exclude" and view.disputed:
                continue
            key = (view.fact_id, view.version)
            found.setdefault(key, (hop, view))
            next_entities.update((view.subject_id, view.object_id))
        seen_entities.update(frontier)
        candidates = sorted(next_entities - seen_entities)
        frontier = tuple(candidates[: settings.frontier_cap])
        seen_entities.update(frontier)
        if not frontier:
            break

    ranked = sorted(
        found.values(),
        key=lambda item: (
            item[0],
            -len(query_terms & set(terms(item[1].statement))),
            item[1].fact_id,
            item[1].version,
        ),
    )
    views = tuple(view for _, view in ranked)
    return LaneOutput(
        lane_record("graph", bool(views), [], search_count=1), facts=views
    )
