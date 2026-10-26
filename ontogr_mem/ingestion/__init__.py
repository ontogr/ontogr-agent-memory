"""Ingestion: turn typed inputs into stored episodes."""

from ontogr_mem.ingestion.connectors import (
    ChatConnector,
    ChatMessage,
    Connector,
    DocumentConnector,
    DocumentInput,
    EventConnector,
    EventInput,
    RecordConnector,
    RecordInput,
    ToolOutputConnector,
    ToolOutputInput,
    UnknownKindError,
    connector_for,
    default_registry,
    doc_id,
)
from ontogr_mem.ingestion.ids import content_hash, episode_id, normalize, seal
from ontogr_mem.ingestion.pipeline import Handoff, IngestionPipeline
from ontogr_mem.ingestion.store import EpisodeStore
from ontogr_mem.ingestion.validate import (
    DraftValidationError,
    validate_batch,
)


__all__ = [
    "ChatConnector",
    "ChatMessage",
    "Connector",
    "connector_for",
    "DocumentConnector",
    "DocumentInput",
    "DraftValidationError",
    "EpisodeStore",
    "EventConnector",
    "EventInput",
    "Handoff",
    "IngestionPipeline",
    "RecordConnector",
    "RecordInput",
    "ToolOutputConnector",
    "ToolOutputInput",
    "UnknownKindError",
    "content_hash",
    "default_registry",
    "doc_id",
    "episode_id",
    "normalize",
    "seal",
    "validate_batch",
]
