"""Lazy import of optional dependencies."""

import importlib
from types import ModuleType

from ontogr_mem.errors import MissingExtraError


def import_extra(module: str, extra: str) -> ModuleType:
    """Import an optional dependency or raise an error with the install hint.

    Args:
        module: Dotted name of the module to import.
        extra: Name of the extra that provides the module.

    Returns:
        The imported module.

    Raises:
        MissingExtraError: If the module itself is not installed.
        ModuleNotFoundError: If the module is installed but one of its own
            dependencies is missing.
    """
    try:
        return importlib.import_module(module)
    except ModuleNotFoundError as exc:
        if exc.name not in {module, module.partition(".")[0]}:
            raise
        raise MissingExtraError(extra) from exc
