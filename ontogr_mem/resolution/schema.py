"""Graph schema declarations for entity resolution storage."""

SCHEMA_NAME = "resolution"
LABELS = (
    "Entity",
    "Alias",
    "Mention",
    "EntityVerdict",
    "MergeRecord",
    "ResolverCounter",
)
RANGES = (
    ("Entity", "uri", False),
    ("Entity", "strict_key", False),
    ("Entity", "status", False),
    ("Alias", "strict_key", False),
    ("Alias", "loose_key", False),
    ("Alias", "entity_id", False),
    ("Mention", "episode_id", False),
    ("Mention", "entity_id", False),
    ("EntityVerdict", "a_id", False),
    ("EntityVerdict", "b_id", False),
    ("MergeRecord", "survivor_id", False),
    ("MergeRecord", "absorbed_id", False),
    ("MergeRecord", "state", False),
)
REQUIRES: tuple[str, ...] = ()
