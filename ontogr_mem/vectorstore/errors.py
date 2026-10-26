"""Errors raised by the vector store."""

from ontogr_mem.errors import AgentMemoryError


class VectorStoreError(AgentMemoryError):
    """Base class for every error the vector store raises."""


class SpaceMismatchError(VectorStoreError):
    """The configured, stored or probed embedding spaces disagree."""


class IndexDriftError(VectorStoreError):
    """A vector index is missing, invalid, still building, or has another shape."""


class QueryTimeoutError(VectorStoreError):
    """A statement exceeded its time limit."""


class InvalidVectorError(VectorStoreError):
    """A vector has the wrong length, a non-finite value or zero norm."""


class StoreNotReadyError(VectorStoreError):
    """`ensure_ready` has not succeeded, so reads and writes are refused."""


class CapabilityError(VectorStoreError):
    """The backend cannot answer this request on this kind."""


class EmbeddingError(VectorStoreError):
    """Base class for embedding failures."""


class TransientEmbeddingError(EmbeddingError):
    """Retryable provider failure such as a rate limit or a connection reset."""


class PermanentEmbeddingError(EmbeddingError):
    """Non-retryable provider failure such as bad credentials or a bad request."""


class EmbeddingTimeoutError(EmbeddingError):
    """A provider call exceeded its time limit. It is never retried."""


class EmbeddingFailed(EmbeddingError):
    """A strict embedding request failed for one of its input texts.

    Attributes:
        index: Index of the first input text without a vector.
    """

    def __init__(self, index: int, detail: str) -> None:
        """Store the failed text index and its outcome detail."""
        super().__init__(f"embedding failed for text {index}: {detail}")
        self.index = index
