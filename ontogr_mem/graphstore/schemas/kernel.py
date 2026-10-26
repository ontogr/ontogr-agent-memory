"""Labels the graph store itself keeps in every tenant graph."""

from ontogr_mem.graphstore.indices import SchemaPart


KERNEL = SchemaPart(name="kernel", labels=("__Meta",))
"""Schema part of the graph store kernel.

The `__Meta` label holds the metadata a component writes for itself, such as
the vector space of a tenant. It is declared here because every tenant graph
gets it.
"""
