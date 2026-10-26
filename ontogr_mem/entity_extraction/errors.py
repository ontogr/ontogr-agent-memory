"""Errors raised by entity extraction."""

from ontogr_mem.errors import AgentMemoryError


class ExtractionError(AgentMemoryError):
    """Base class for extraction failures."""


class UnknownKindError(ExtractionError, ValueError):
    """The episode kind has no extraction path. It never falls back."""


class ParseFailedError(ExtractionError):
    """A model reply stayed invalid after the repair budget."""
