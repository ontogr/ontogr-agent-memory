"""Vector lane over fact and episode vectors."""

import asyncio
from collections.abc import Sequence

from ontogr_mem.retrieval.lanes import LaneOutput
from ontogr_mem.retrieval.lanes._shared import lane_record, run_search
from ontogr_mem.retrieval.models import LaneRecord
from ontogr_mem.retrieval.settings import RetrievalSettings
from ontogr_mem.temporal.query import FactView, TemporalQuery
from ontogr_mem.temporal.reader import FactReader
from ontogr_mem.vectorstore.models import (
    SearchFilters,
    TemporalFilter,
    VectorKind,
)
from ontogr_mem.vectorstore.vector_store import VectorStore


async def run_vector_lane(
    vectors: VectorStore,
    embedding: Sequence[float] | None,
    *,
    limit: int,
    settings: RetrievalSettings,
    facts: FactReader | None = None,
    query: TemporalQuery | None = None,
    disputed: str = "flag",
    embedding_error: str = "query embedding is unavailable",
) -> LaneOutput:
    """Search fact history and episode vectors, then hydrate visible facts."""
    if embedding is None:
        error = RuntimeError(embedding_error)
        record = lane_record(
            "vector", False, [("edge", error), ("episode", error)], search_count=2
        )
        return LaneOutput(record, error=error)

    edge_search, episode_search = await asyncio.gather(
        run_search(
            lambda: vectors.search(
                VectorKind.EDGE,
                embedding,
                k=settings.candidate_factor * limit,
                filters=SearchFilters(temporal=TemporalFilter(history=True)),
            )
        ),
        run_search(
            lambda: vectors.search(
                VectorKind.EPISODE,
                embedding,
                k=settings.candidate_factor * limit,
            )
        ),
    )
    failures = [
        (name, result.error)
        for name, result in (("edge", edge_search), ("episode", episode_search))
        if result.error is not None
    ]
    edge_hits = edge_search.value or []
    episode_hits = episode_search.value or []
    fact_views: list[FactView] = []
    dropped = 0
    if edge_hits and facts is not None and query is not None:
        hit_ids = list(dict.fromkeys(hit.id for hit in edge_hits))
        fact_views = await facts.by_ids(hit_ids, query)
        visible_ids = {view.fact_id for view in fact_views}
        dropped = len(set(hit_ids) - visible_ids)
        positions: dict[str, int] = {}
        for rank, hit in enumerate(edge_hits):
            positions.setdefault(hit.id, rank)
        fact_views.sort(
            key=lambda view: (positions[view.fact_id], view.fact_id, view.version)
        )
        if disputed == "exclude":
            fact_views = [view for view in fact_views if not view.disputed]

    episode_ids = tuple(dict.fromkeys(hit.id for hit in episode_hits))
    found = bool(fact_views or episode_ids)
    record = lane_record("vector", found, failures, search_count=2)
    if dropped:
        detail = f"{record.detail}; " if record.detail else ""
        record = LaneRecord(
            record.name, record.status, f"{detail}dropped={dropped}", record.failed
        )
    error = failures[0][1] if failures else None
    return LaneOutput(record, tuple(fact_views), episode_ids, error=error)
