# ruff: noqa: N818
"""Errors raised by the ontology manager."""

from ontogr_mem.errors import AgentMemoryError


class OntologyError(AgentMemoryError):
    """Base class for ontology-manager errors."""


class VersionConflict(OntologyError):
    """A commit used a stale base version; re-read the catalog and re-apply."""

    def __init__(self, base_version: int) -> None:
        """Remember which base version lost the race."""
        super().__init__(f"stale base version {base_version}")
        self.base_version = base_version


class InvalidChange(OntologyError):
    """A change set would produce an invalid catalog.

    ``item`` names the conflicting thing; ``hint`` says what to do instead.
    """

    def __init__(self, message: str, item: str = "", hint: str = "") -> None:
        """Build a message that names the item and the way out."""
        full = message
        if item:
            full += f" [{item}]"
        if hint:
            full += f"; {hint}"
        super().__init__(full)
        self.item = item
        self.hint = hint


class MissingVersionError(OntologyError):
    """A requested ontology version does not exist."""


class ProposalDecidedError(OntologyError):
    """A proposal that was accepted, rejected or superseded cannot be rewritten."""


class DenseIdError(OntologyError):
    """A dense type id is outside the projection that issued it."""
