"""Errors raised by the retrieval component."""

from ontogr_mem.errors import AgentMemoryError


class RetrievalError(AgentMemoryError):
    """Base class for every error the retrieval component raises."""


class RerankerError(RetrievalError):
    """Optional passage scoring could not complete."""


class KeywordSearchError(RetrievalError):
    """Keyword search could not complete."""


class KeywordTimeoutError(KeywordSearchError):
    """Keyword search exceeded its query timeout."""


class KeywordIndexError(KeywordSearchError):
    """A required fulltext index is missing or not ready."""
