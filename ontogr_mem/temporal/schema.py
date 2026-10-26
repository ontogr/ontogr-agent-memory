"""Graph schema declarations for temporal storage."""

SCHEMA_NAME = "temporal"
LABELS = (
    "Slot",
    "LedgerChain",
    "Observation",
    "LedgerEntry",
    "Supersession",
    "FactVerdict",
    "FactDecision",
    "Conflict",
)
RANGES = (
    ("Observation", "episode_id", False),
    ("Observation", "slot_key", False),
    ("LedgerEntry", "chain", False),
    ("Supersession", "slot_key", False),
    ("Conflict", "slot_key", False),
    ("FactVerdict", "fact_a_id", False),
    ("FactVerdict", "fact_b_id", False),
    ("FACT", "slot_key", True),
    ("FACT", "fact_id", True),
    ("FACT", "valid_from", True),
    ("FACT", "expired_at", True),
)
REQUIRES = ("episodes", "resolution")
