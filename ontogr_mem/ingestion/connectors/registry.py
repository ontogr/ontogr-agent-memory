"""Registry that maps an episode kind to its connector."""

from collections.abc import Mapping
from typing import Any

from ontogr_mem.data_models import Kind
from ontogr_mem.errors import AgentMemoryError
from ontogr_mem.ingestion.connectors.base import Connector
from ontogr_mem.ingestion.connectors.chat import ChatConnector
from ontogr_mem.ingestion.connectors.document import DocumentConnector
from ontogr_mem.ingestion.connectors.event import EventConnector
from ontogr_mem.ingestion.connectors.record import RecordConnector
from ontogr_mem.ingestion.connectors.tool_output import ToolOutputConnector


class UnknownKindError(AgentMemoryError, KeyError):
    """No connector is registered for a kind."""


def connector_for(
    connectors: Mapping[Kind, Connector[Any]], kind: Kind | str
) -> Connector[Any]:
    """Return a connector, refusing unknown or unregistered kinds.

    Raises:
        UnknownKindError: If `kind` is not known or has no connector.
    """
    try:
        key = Kind(kind)
    except ValueError:
        raise UnknownKindError(f"unknown kind: {kind!r}") from None
    try:
        return connectors[key]
    except KeyError:
        raise UnknownKindError(f"no connector registered for kind: {key}") from None


def default_registry(
    overrides: Mapping[Kind, Connector[Any]] | None = None,
) -> dict[Kind, Connector[Any]]:
    """Build the registry of built-in connectors; `overrides` replace them."""
    connectors: dict[Kind, Connector[Any]] = {
        Kind.MESSAGE: ChatConnector(),
        Kind.DOCUMENT: DocumentConnector(),
        Kind.TOOL_OUTPUT: ToolOutputConnector(),
        Kind.EVENT: EventConnector(),
        Kind.RECORD: RecordConnector(),
    }
    connectors.update(overrides or {})
    return connectors
