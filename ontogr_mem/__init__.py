"""OntoGr Agent Memory: durable, traceable memory for AI agents."""

from importlib.metadata import PackageNotFoundError, version

from ontogr_mem.client import (
    EntityPage,
    Memory,
    Modality,
    OntologyView,
    StaleEpisodePage,
)


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
