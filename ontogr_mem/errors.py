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
    """One or more optional extras are not installed.

    Attributes:
        extras: Names of the extras that are not installed.
    """

    def __init__(
        self,
        extras: Sequence[str],
        *,
        features: Sequence[str] = (),
        needed: Sequence[str] = (),
        subcommand: str = "",
    ) -> None:
        """Build the error with the commands that provide the missing extras.

        Args:
            extras: Extras that are not installed.
            features: Features that need the missing extras.
            needed: Every extra the command needs. Defaults to ``extras``.
            subcommand: Command placed after the uvx invocation, such as mcp.
        """
        self.extras = tuple(extras)
        selected = tuple(needed) if needed else self.extras
        lines = [f"missing extras: {', '.join(self.extras)}"]
        if features:
            lines.append(f"needed for: {', '.join(features)}")
        lines.append(f'install with: uv add "ontogr-mem[{",".join(self.extras)}]"')
        invocation = "ontogr-mem" + (f" {subcommand}" if subcommand else "")
        lines.append(
            f'or run with: uvx --from "ontogr-mem[{",".join(selected)}]" {invocation}'
        )
        super().__init__("\n".join(lines))
