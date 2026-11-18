"""OntoGr Agent Memory: durable, traceable memory for AI agents."""

from importlib import import_module
from importlib.metadata import PackageNotFoundError, version
from typing import Any


# Source checkouts may have no installed package metadata.
try:
    __version__ = version("ontogr-mem")
except PackageNotFoundError:  # pragma: no cover
    __version__ = "0.0.0.dev0"

__all__ = [
    "EntityPage",
    "Memory",
    "Modality",
    "OntologyView",
    "StaleEpisodePage",
    "__version__",
]


def __getattr__(name: str) -> Any:
    """Load public client exports only when they are requested."""
    if name in __all__ and name != "__version__":
        client = import_module("ontogr_mem.client")
        return getattr(client, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
