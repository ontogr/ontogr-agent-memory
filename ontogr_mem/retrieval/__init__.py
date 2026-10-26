"""Retrieval: the request, the result and the ranking that builds them."""

from ontogr_mem.retrieval.errors import (
    KeywordIndexError,
    KeywordSearchError,
    KeywordTimeoutError,
    RerankerError,
    RetrievalError,
)
from ontogr_mem.retrieval.fusion import fuse
from ontogr_mem.retrieval.keyword import KeywordHit, KeywordKind, KeywordSearch
from ontogr_mem.retrieval.models import (
    DISPUTED_MODES,
    GRAPH_DEPTH_CAP,
    LIMIT_CAP,
    MENTION_CAP,
    ContextBlock,
    EntityCard,
    EpisodeResult,
    FactResult,
    LaneRecord,
    LaneStatus,
    RetrievalRequest,
    RetrievalResult,
)
from ontogr_mem.retrieval.ranking import apply_penalties
from ontogr_mem.retrieval.rerank import Passage, Reranker
from ontogr_mem.retrieval.retriever import Retriever
from ontogr_mem.retrieval.settings import KeywordSettings, RetrievalSettings


__all__ = [
    "DISPUTED_MODES",
    "GRAPH_DEPTH_CAP",
    "LIMIT_CAP",
    "MENTION_CAP",
    "ContextBlock",
    "EntityCard",
    "EpisodeResult",
    "FactResult",
    "KeywordHit",
    "KeywordIndexError",
    "KeywordKind",
    "KeywordSearch",
    "KeywordSearchError",
    "KeywordSettings",
    "KeywordTimeoutError",
    "LaneRecord",
    "LaneStatus",
    "Passage",
    "Reranker",
    "RerankerError",
    "RetrievalError",
    "RetrievalRequest",
    "RetrievalResult",
    "RetrievalSettings",
    "Retriever",
    "apply_penalties",
    "fuse",
]
