"""Connectors: pure mappings from typed input to episode drafts."""

from ontogr_mem.ingestion.connectors.base import Connector
from ontogr_mem.ingestion.connectors.chat import ChatConnector, ChatMessage
from ontogr_mem.ingestion.connectors.document import (
    DocumentConnector,
    DocumentInput,
    doc_id,
)
from ontogr_mem.ingestion.connectors.event import EventConnector, EventInput
from ontogr_mem.ingestion.connectors.record import RecordConnector, RecordInput
from ontogr_mem.ingestion.connectors.registry import (
    UnknownKindError,
    connector_for,
    default_registry,
)
from ontogr_mem.ingestion.connectors.tool_output import (
    ToolOutputConnector,
    ToolOutputInput,
)


__all__ = [
    "ChatConnector",
    "ChatMessage",
    "Connector",
    "connector_for",
    "DocumentConnector",
    "DocumentInput",
    "EventConnector",
    "EventInput",
    "RecordConnector",
    "RecordInput",
    "ToolOutputConnector",
    "ToolOutputInput",
    "UnknownKindError",
    "default_registry",
    "doc_id",
]
