"""Graph schema declarations for episode storage."""

SCHEMA_NAME = "episodes"
LABELS = ("Episode", "Source", "Actor", "EpisodeMeta")
RANGES = (("Episode", "occurred_at", False),)
REQUIRES: tuple[str, ...] = ()
