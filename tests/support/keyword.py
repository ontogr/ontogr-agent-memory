"""Keyword contract harnesses sharing the same tenant's three public stores."""

from dataclasses import dataclass

from ontogr_mem.graphstore.episode_store import GraphEpisodeStore
from ontogr_mem.graphstore.keyword_search import GraphKeywordSearch
from ontogr_mem.graphstore.memory import (
    MemoryEpisodeStore,
    MemoryResolverStore,
    MemoryTemporalStore,
)
from ontogr_mem.graphstore.memory.keyword_search import MemoryKeywordSearch
from ontogr_mem.graphstore.protocol import GraphStore
from ontogr_mem.graphstore.resolver_store import GraphResolverStore
from ontogr_mem.graphstore.temporal.store import GraphTemporalStore
from ontogr_mem.graphstore.types import GraphHandle
from ontogr_mem.ingestion.store import EpisodeStore
from ontogr_mem.resolution.resolver_store import ResolverStore
from ontogr_mem.retrieval.keyword import KeywordSearch
from ontogr_mem.temporal.temporal_store import TemporalStore


@dataclass
class KeywordHarness:
    """Search and the public stores used to seed its corpus."""

    keyword: KeywordSearch
    resolver: ResolverStore
    temporal: TemporalStore
    episodes: EpisodeStore


def memory_keyword_harness() -> KeywordHarness:
    """Return an isolated in-memory tenant."""
    episodes = MemoryEpisodeStore()
    resolver = MemoryResolverStore()
    temporal = MemoryTemporalStore(episodes, resolver)
    return KeywordHarness(
        MemoryKeywordSearch(resolver, temporal, episodes), resolver, temporal, episodes
    )


def graph_keyword_harness(store: GraphStore, handle: GraphHandle) -> KeywordHarness:
    """Bind search and storage adapters to the same tenant runner."""
    runner = store.runner(handle)
    return KeywordHarness(
        GraphKeywordSearch(runner),
        GraphResolverStore(runner),
        GraphTemporalStore(store, handle),
        GraphEpisodeStore(runner),
    )
