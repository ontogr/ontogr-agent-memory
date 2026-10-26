"""A schema part with the labels the graph store tests write."""

from ontogr_mem.graphstore import RangeIndex, SchemaPart, UniqueConstraint


TEST_PART = SchemaPart(
    name="test",
    labels=("Entity", "Slot", "Ledger", "Item", "Person", "Org"),
    ranges=(
        RangeIndex("Entity", "name"),
        RangeIndex("FACT", "valid_at", edge=True),
    ),
    uniques=(UniqueConstraint("Entity", "email"),),
    edge_keys={"FACT": ("fact_key",), "MENTIONS": ()},
)

DRIFT_PART = SchemaPart(
    name="drift",
    labels=("Drift",),
    ranges=(RangeIndex("Drift", "name"),),
    uniques=(UniqueConstraint("Drift", "email"),),
)
"""A schema only the drift tests use, so dropping its rules cannot disturb others."""
