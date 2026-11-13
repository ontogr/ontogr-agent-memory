"""Model failures exposed to memory components."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ontogr_mem.errors import AgentMemoryError


if TYPE_CHECKING:
    from ontogr_mem.llm.run import TokenUsage


class ModelFailure(AgentMemoryError):  # noqa: N818
    """A model operation failed, with any usage collected before failure."""

    def __init__(self, message: str, *, usage: TokenUsage | None = None) -> None:
        """Store the public message and optional unpriced token usage."""
        super().__init__(message)
        self.usage = usage


class ModelNotConfiguredError(ModelFailure):
    """The requested kind of work has no configured model."""


class ModelCallFailedError(ModelFailure):
    """The provider call failed after any provider-owned retries."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        usage: TokenUsage | None = None,
    ) -> None:
        """Store an optional HTTP status and collected usage."""
        super().__init__(message, usage=usage)
        self.status_code = status_code


class ModelOutputError(ModelFailure):
    """The reply never validated or ended at its output limit."""

    def __init__(
        self,
        message: str,
        *,
        truncated: bool = False,
        usage: TokenUsage | None = None,
    ) -> None:
        """Store whether the output limit cut the reply short."""
        super().__init__(message, usage=usage)
        self.truncated = truncated
