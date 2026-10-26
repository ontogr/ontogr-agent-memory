"""Entity lane over explicit mentions, alias text and node vectors."""

import asyncio
from collections.abc import Sequence

from ontogr_mem.data_models import Entity, EntityStatus
from ontogr_mem.resolution.keys import loose_key, strict_key
from ontogr_mem.resolution.resolver_store import ResolverStore
from ontogr_mem.retrieval.fusion import fuse
from ontogr_mem.retrieval.keyword import KeywordKind, KeywordSearch
from ontogr_mem.retrieval.lanes import LaneOutput
from ontogr_mem.retrieval.lanes._shared import SearchResult, lane_record, run_search
from ontogr_mem.retrieval.settings import RetrievalSettings
from ontogr_mem.vectorstore.errors import VectorStoreError
from ontogr_mem.vectorstore.models import SearchFilters, VectorKind
from ontogr_mem.vectorstore.vector_store import VectorStore


def _mention_candidates(
    result: SearchResult[tuple[list[list[Entity]], list[Exception]]],
) -> tuple[list[list[str]], list[tuple[str, Exception]]]:
    """Return active mention ids and failures from strict/loose lookups."""
    if result.error is not None:
        return [], [("mention", result.error)]
    if result.value is None:
        return [], []

    mention_entities, mention_errors = result.value
    mention_lists = [
        [entity.entity_id for entity in candidates] for candidates in mention_entities
    ]
    failures = [("mention", mention_errors[0])] if mention_errors else []
    return mention_lists, failures


def _entity_sources(
    mention_lists: list[list[str]], keyword_ids: list[str], node_ids: list[str]
) -> dict[str, list[str]]:
    """Map each candidate to the lanes that found it."""
    source_map: dict[str, list[str]] = {}
    for mention_ids in mention_lists:
        for entity_id in mention_ids:
            sources = source_map.setdefault(entity_id, [])
            if "mention" not in sources:
                sources.append("mention")
    for source, ids in (("keyword", keyword_ids), ("vector", node_ids)):
        for entity_id in ids:
            sources = source_map.setdefault(entity_id, [])
            if source not in sources:
                sources.append(source)
    return source_map


def _finish_entity_lane(
    mention_result: SearchResult[tuple[list[list[Entity]], list[Exception]]],
    keyword_result: SearchResult[list[str]],
    node_result: SearchResult[list[str]],
    *,
    settings: RetrievalSettings,
    has_mentions: bool,
) -> LaneOutput:
    """Fuse successful entity searches and retain partial-failure details."""
    mention_lists, failures = _mention_candidates(mention_result)
    keyword_ids = keyword_result.value or []
    node_ids = node_result.value or []
    if keyword_result.error is not None:
        failures.append(("keyword", keyword_result.error))
    if node_result.error is not None:
        failures.append(("node", node_result.error))

    fused = fuse(
        [*mention_lists, keyword_ids, node_ids],
        limit=settings.entity_candidates,
        settings=settings,
    )
    result_ids = tuple(entity_id for entity_id, _ in fused)
    source_map = _entity_sources(mention_lists, keyword_ids, node_ids)
    record = lane_record(
        "entity", bool(result_ids), failures, search_count=2 + bool(has_mentions)
    )
    return LaneOutput(
        record,
        entities=result_ids,
        entity_sources={
            entity_id: tuple(source_map[entity_id]) for entity_id in result_ids
        },
        error=next((error for _, error in failures), None),
    )


async def run_entity_lane(
    vectors: VectorStore,
    keywords: KeywordSearch,
    entities: ResolverStore,
    embedding: Sequence[float] | None,
    *,
    query: str,
    mentions: Sequence[str],
    settings: RetrievalSettings,
    embedding_error: str = "query embedding is unavailable",
) -> LaneOutput:
    """Fuse candidate entities from explicit mentions and two query searches."""

    async def mention_search() -> tuple[list[list[Entity]], list[Exception]]:
        lists: list[list[Entity]] = []
        errors: list[Exception] = []
        for mention in mentions:
            strict_result, loose_result = await asyncio.gather(
                run_search(
                    lambda name=mention: entities.entities_by_strict_key(
                        strict_key(name)
                    )
                ),
                run_search(
                    lambda name=mention: entities.entities_by_loose_key(loose_key(name))
                ),
            )
            for result in (strict_result, loose_result):
                if result.error is not None:
                    errors.append(result.error)
            candidates = {
                entity.entity_id: entity
                for result in (strict_result, loose_result)
                for entity in (result.value or [])
                if entity.status is EntityStatus.ACTIVE
            }
            lists.append([candidates[key] for key in sorted(candidates)])
        return lists, errors

    async def keyword_search() -> list[str]:
        hits = await keywords.search(
            KeywordKind.ENTITY, query, limit=settings.entity_candidates
        )
        if not hits:
            return []
        found = await entities.get_entities(
            [hit.id for hit in hits], follow_merged=True
        )
        active = {
            entity.entity_id for entity in found if entity.status is EntityStatus.ACTIVE
        }
        return list(dict.fromkeys(hit.id for hit in hits if hit.id in active))

    async def node_search() -> list[str]:
        if embedding is None:
            raise VectorStoreError(embedding_error)
        hits = await vectors.search(
            VectorKind.NODE,
            embedding,
            k=settings.entity_candidates,
            filters=SearchFilters(statuses=frozenset({EntityStatus.ACTIVE.value})),
        )
        if not hits:
            return []
        found = await entities.get_entities(
            [hit.id for hit in hits], follow_merged=True
        )
        active = {
            entity.entity_id for entity in found if entity.status is EntityStatus.ACTIVE
        }
        return list(dict.fromkeys(hit.id for hit in hits if hit.id in active))

    mention_result, keyword_result, node_result = await asyncio.gather(
        run_search(mention_search),
        run_search(keyword_search),
        run_search(node_search),
    )
    return _finish_entity_lane(
        mention_result,
        keyword_result,
        node_result,
        settings=settings,
        has_mentions=bool(mentions),
    )
