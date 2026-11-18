"""Vector lane over fact and episode vectors."""

import asyncio
from collections.abc import Sequence

from ontogr_mem.retrieval.gate import FactGate
from ontogr_mem.retrieval.lanes import LaneOutput
from ontogr_mem.retrieval.lanes._shared import lane_record, run_search, widen
from ontogr_mem.retrieval.models import LaneRecord
from ontogr_mem.retrieval.settings import RetrievalSettings
from ontogr_mem.temporal.query import TemporalQuery
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
    gate: FactGate | None = None,
) -> LaneOutput:
    """Search fact history and episode vectors, then hydrate visible facts."""
    if embedding is None:
        error = RuntimeError(embedding_error)
        record = lane_record(
            "vector", False, [("edge", error), ("episode", error)], search_count=2
        )
        return LaneOutput(record, error=error)

    last_success = None

    async def edge_candidates(window: int):
        nonlocal last_success
        hits = await vectors.search(
            VectorKind.EDGE,
            embedding,
            k=window,
            filters=SearchFilters(temporal=TemporalFilter(history=True)),
        )
        positions: dict[str, int] = {}
        for rank, hit in enumerate(hits):
            positions.setdefault(hit.id, rank)
        views = []
        if hits and facts is not None and query is not None:
            hit_ids = list(dict.fromkeys(hit.id for hit in hits))
            views = await facts.by_ids(hit_ids, query)
            views.sort(
                key=lambda view: (
                    positions.get(view.fact_id, len(positions)),
                    view.fact_id,
                    view.version,
                )
            )
            if gate is not None:
                views = await gate.admit(views)
            if disputed == "exclude":
                views = [view for view in views if not view.disputed]
        result = (hits, views, len(hits), window, positions if hits else {})
        last_success = result
        return result

    edge_search, episode_search = await asyncio.gather(
        run_search(
            lambda: widen(
                edge_candidates,
                lambda result: len(result[1]),
                lambda result: result[2] == result[3],
                start=settings.candidate_factor * limit,
                want=limit,
                rechecks=(
                    settings.filter_rechecks
                    if facts is not None and query is not None
                    else 0
                ),
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
    edge_result = edge_search.value
    if edge_result is None and edge_search.error is not None:
        edge_result = last_success
    edge_hits = edge_result[0] if edge_result else []
    episode_hits = episode_search.value or []
    fact_views = edge_result[1] if edge_result else []
    positions = edge_result[4] if edge_result else {}
    fact_views.sort(
        key=lambda view: (
            positions.get(view.fact_id, len(positions)),
            view.fact_id,
            view.version,
        )
    )
    retained_ids = {view.fact_id for view in fact_views}
    # Without hydration, every vector hit remains a candidate.
    dropped = (
        len({hit.id for hit in edge_hits} - retained_ids)
        if facts is not None and query is not None
        else 0
    )

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
