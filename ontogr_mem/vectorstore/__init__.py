"""Vector storage: embed each text once and search by cosine similarity."""

from ontogr_mem.observability.metrics import Metrics
from ontogr_mem.vectorstore.embedder import Embedder, EmbeddingGateway
from ontogr_mem.vectorstore.episode_vectors import write_episode_vectors
from ontogr_mem.vectorstore.errors import (
    CapabilityError,
    EmbeddingError,
    EmbeddingFailed,
    EmbeddingTimeoutError,
    IndexDriftError,
    InvalidVectorError,
    PermanentEmbeddingError,
    QueryTimeoutError,
    SpaceMismatchError,
    StoreNotReadyError,
    TransientEmbeddingError,
    VectorStoreError,
)
from ontogr_mem.vectorstore.fact_vectors import FactVectorObserver
from ontogr_mem.vectorstore.models import (
    Capabilities,
    EdgeExtras,
    EmbedItem,
    FailedItem,
    FailReason,
    Hit,
    NodeExtras,
    SearchFilters,
    SpaceSpec,
    TemporalFilter,
    Validity,
    ValidityUpdate,
    VectorKind,
    WriteResult,
)
from ontogr_mem.vectorstore.protocol import Backend
from ontogr_mem.vectorstore.settings import VectorStoreSettings
from ontogr_mem.vectorstore.vector_store import VectorStore
from ontogr_mem.vectorstore.writer import FailedWrite, VectorWriter


__all__ = [
    "Backend",
    "Capabilities",
    "CapabilityError",
    "EdgeExtras",
    "EmbedItem",
    "Embedder",
    "EmbeddingError",
    "EmbeddingFailed",
    "EmbeddingGateway",
    "EmbeddingTimeoutError",
    "FactVectorObserver",
    "FailReason",
    "FailedItem",
    "FailedWrite",
    "Hit",
    "IndexDriftError",
    "InvalidVectorError",
    "Metrics",
    "NodeExtras",
    "PermanentEmbeddingError",
    "QueryTimeoutError",
    "SearchFilters",
    "SpaceMismatchError",
    "SpaceSpec",
    "StoreNotReadyError",
    "TemporalFilter",
    "TransientEmbeddingError",
    "Validity",
    "ValidityUpdate",
    "VectorKind",
    "VectorStore",
    "VectorStoreError",
    "VectorStoreSettings",
    "VectorWriter",
    "WriteResult",
    "write_episode_vectors",
]
