"""FalkorDB backend of the graph store. Needs the `falkordb` extra."""

from ontogr_mem._extras import import_extra


import_extra("falkordb", "falkordb")
import_extra("redis", "falkordb")

from ontogr_mem.graphstore.falkordb.settings import FalkorDBSettings
from ontogr_mem.graphstore.falkordb.store import FalkorDBGraphStore


__all__ = ["FalkorDBGraphStore", "FalkorDBSettings"]
