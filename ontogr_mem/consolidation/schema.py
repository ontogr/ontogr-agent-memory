"""Graph schema declarations for consolidation storage."""

SCHEMA_NAME = "consolidation"
LABELS = (
    "ConsWatermark",
    "ConsLease",
    "ConsCounter",
    "ConsFeedbackEvent",
    "ConsFeedbackWeight",
    "ConsAccessEvent",
    "ConsSessionEntry",
    "ConsProposal",
    "ConsScore",
    "ConsArchiveRecord",
    "ConsCommunity",
    "ConsBucket",
)
RANGES = (
    ("ConsFeedbackEvent", "applied", False),
    ("ConsFeedbackEvent", "target_id", False),
    ("ConsAccessEvent", "target_id", False),
    ("ConsSessionEntry", "session_id", False),
    ("ConsProposal", "status", False),
    ("ConsProposal", "session_id", False),
    ("ConsScore", "kind", False),
    ("ConsArchiveRecord", "target_kind", False),
    ("ConsArchiveRecord", "state", False),
    ("ConsBucket", "level", False),
)
REQUIRES: tuple[str, ...] = ()
