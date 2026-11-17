"""Graph schema for ontology versions."""

SCHEMA_NAME = "ontology"
LABELS = ("OntologyHead", "OntologySnapshot", "OntologyChangeSet")
RANGES: tuple[tuple[str, str, bool], ...] = ()
REQUIRES = ("episodes",)
