"""Errors raised by the package."""

from collections.abc import Sequence


class AgentMemoryError(Exception):
    """Root of every error the package raises."""


class StoreError(AgentMemoryError):
    """Base class for every error a store raises."""


class AmbiguousNameError(AgentMemoryError):
    """A name resolves to more than one entity.

    Attributes:
        code: Stable error code for this condition.
        ids: Distinct matching entity ids in sorted order.
    """

    code = "ambiguous_name"

    def __init__(self, ids: Sequence[str]) -> None:
        """Build the error with the distinct matching entity ids."""
        self.ids = tuple(sorted(set(ids)))
        super().__init__(f"name matches multiple entities: {', '.join(self.ids)}")


class MissingExtraError(AgentMemoryError):
    """An optional dependency is not installed.

    Attributes:
        extra: Name of the extra that provides the missing dependency.
    """

    def __init__(self, extra: str) -> None:
        """Build the error with the install command for the extra.

        Args:
            extra: Name of the extra to install.
        """
        super().__init__(
            f'this feature needs the "{extra}" extra: '
            f'pip install "agent-memory[{extra}]"'
        )
        self.extra = extra
