"""Ordered list of stages with unique names."""

from collections.abc import Sequence

from ontogr_mem.consolidation.errors import RegistryError
from ontogr_mem.consolidation.stage import Stage


def validate_stages(stages: Sequence[Stage]) -> tuple[Stage, ...]:
    """Return stages in run order after checking their names are unique.

    Raises:
        RegistryError: If two stages share a name.
    """
    names = [stage.name for stage in stages]
    if len(set(names)) != len(names):
        raise RegistryError(f"duplicate stage names: {names}")
    return tuple(stages)
