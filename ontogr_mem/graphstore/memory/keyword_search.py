"""In-memory keyword search through the public storage ports."""

import math
from collections import Counter
from datetime import UTC, datetime

from ontogr_mem.ingestion.store import EpisodeStore
from ontogr_mem.resolution.resolver_store import ResolverStore
from ontogr_mem.retrieval.keyword import KeywordHit, KeywordKind
from ontogr_mem.retrieval.text import terms, tokens
from ontogr_mem.temporal.temporal_store import TemporalStore


def _rank(documents: list[tuple[str, str]], query: str, limit: int) -> list[KeywordHit]:
    """Score token overlap with BM25 and keep each id's best document score."""
    documents_tokens = [Counter(tokens(text)) for _, text in documents]
    query_terms = terms(query)
    if not documents_tokens or not query_terms or limit < 1:
        return []
    lengths = [sum(doc.values()) for doc in documents_tokens]
    average = sum(lengths) / len(lengths) or 1.0
    scores: dict[str, float] = {}
    counts = {
        term: sum(term in doc for doc in documents_tokens) for term in query_terms
    }
    for (key, _), doc, length in zip(documents, documents_tokens, lengths, strict=True):
        score = 0.0
        for term in query_terms:
            frequency = doc[term]
            if not frequency:
                continue
            count = counts[term]
            inverse = math.log(
                1 + (len(documents_tokens) - count + 0.5) / (count + 0.5)
            )
            score += (
                inverse
                * frequency
                * 2.2
                / (frequency + 1.2 * (0.25 + 0.75 * length / average))
            )
        if score:
            scores[key] = max(score, scores.get(key, 0.0))
    ordered = sorted(scores, key=lambda key: (-scores[key], key))[:limit]
    return [KeywordHit(key, scores[key], rank) for rank, key in enumerate(ordered)]


class MemoryKeywordSearch:
    """Keyword twin over one tenant's resolver, temporal and episode stores."""

    def __init__(
        self, resolver: ResolverStore, temporal: TemporalStore, episodes: EpisodeStore
    ) -> None:
        """Bind the three stores whose documents are searched."""
        self._resolver = resolver
        self._temporal = temporal
        self._episodes = episodes

    async def search(
        self, kind: KeywordKind, query: str, *, limit: int
    ) -> list[KeywordHit]:
        """Return ranked token matches over the current documents of `kind`."""
        if not terms(query) or limit < 1:
            return []
        documents: list[tuple[str, str]] = []
        if kind is KeywordKind.ENTITY:
            for entity in await self._resolver.active_entities():
                documents.extend(
                    (entity.entity_id, alias.text)
                    for alias in await self._resolver.aliases_of(entity.entity_id)
                )
        elif kind is KeywordKind.OBSERVATION:
            documents = [
                (obs.obs_id, f"{obs.statement} {obs.quote}")
                for obs in (await self._temporal.snapshot()).observations
            ]
        elif kind is KeywordKind.EPISODE:
            documents = [
                (episode.id, episode.text)
                for episode in await self._episodes.latest_before(
                    datetime.max.replace(tzinfo=UTC), 2**63 - 1
                )
            ]
        return _rank(documents, query, limit)
