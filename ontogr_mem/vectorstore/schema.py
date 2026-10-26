"""Graph schema requirements for objects that carry vectors."""

from ontogr_mem.graphstore.indices import SchemaPart


VECTOR = SchemaPart(
    name="vector",
    labels=("Entity", "Episode"),
    edge_keys={"FACT": ("fact_id",)},
)
