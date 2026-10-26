"""Graph schema declarations for ontology storage."""

SCHEMA_NAME = "ontology"
LABELS = (
    "OntologyHead",
    "OntologySnapshot",
    "OntologyChangeSet",
    "OntologyUnmatched",
    "OntologyProposal",
)
RANGES = (
    ("OntologyUnmatched", "episode_id", False),
    ("OntologyProposal", "status", False),
)
REQUIRES = ("episodes",)
