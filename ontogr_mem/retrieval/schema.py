"""Optional fulltext indexes used by the keyword retrieval lane."""

SCHEMA_NAME = "keyword"
REQUIRES = ("episodes", "resolution", "temporal")
FULLTEXTS = (
    ("Alias", ("text",)),
    ("Observation", ("statement", "quote")),
    ("Episode", ("content",)),
)
