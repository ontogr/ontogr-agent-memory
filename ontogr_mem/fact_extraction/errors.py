"""Errors raised by fact extraction."""

from ontogr_mem.errors import AgentMemoryError


class FactExtractionError(AgentMemoryError):
    """Base class for fact extraction failures."""


class UnknownKindError(FactExtractionError, ValueError):
    """The episode kind has no fact extraction path. It never falls back."""
