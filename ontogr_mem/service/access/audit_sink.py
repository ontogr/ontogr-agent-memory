"""Audit events and adapters for in-memory and chained audit storage."""

import json
from dataclasses import dataclass
from datetime import datetime
from typing import Literal, Protocol

from ontogr_mem.relational_store.audit.models import (
    AuditAction,
    AuditDetailValue,
    AuditEntry,
)
from ontogr_mem.relational_store.audit.models import (
    AuditOutcome as ChainOutcome,
)
from ontogr_mem.relational_store.audit.store import AuditLog


AuditOutcome = Literal["allowed", "denied", "ok"]


@dataclass(frozen=True)
class AuditEvent:
    """A record of an authorization decision or access-layer change."""

    actor_id: str
    action: str
    outcome: AuditOutcome
    tenant: str | None
    target: str
    detail: dict[str, object]
    at: datetime


class AuditSink(Protocol):
    """Append access events to an audit destination."""

    async def append(self, event: AuditEvent) -> None:
        """Store one audit event."""
        ...


class InMemoryAuditSink:
    """Collect audit events in memory for tests and transient runtimes."""

    def __init__(self) -> None:
        """Create an empty event collection."""
        self._events: list[AuditEvent] = []

    @property
    def events(self) -> list[AuditEvent]:
        """Return a copy of the events appended so far."""
        return list(self._events)

    async def append(self, event: AuditEvent) -> None:
        """Append one event to the in-memory collection."""
        self._events.append(event)


class ChainedAuditSink:
    """Adapt access events to the durable audit chain."""

    def __init__(self, audit_log: AuditLog) -> None:
        """Create a sink backed by an audit log."""
        self._audit_log = audit_log

    async def append(self, event: AuditEvent) -> None:
        """Store an access event as a hash-chained audit entry."""
        action = _CHAIN_ACTIONS.get(event.action, AuditAction.AUTHORIZE)
        detail = {name: _detail_value(value) for name, value in event.detail.items()}
        detail["event_action"] = event.action
        await self._audit_log.append(
            AuditEntry(
                action=action,
                outcome=ChainOutcome(event.outcome),
                principal_id=event.actor_id,
                tenant=event.tenant,
                target=event.target,
                trace_id="",
                detail=detail,
            )
        )


_CHAIN_ACTIONS = {
    "grant.added": AuditAction.GRANT_ADDED,
    "grant.removed": AuditAction.GRANT_REMOVED,
    "export": AuditAction.EXPORT,
    "forget": AuditAction.FORGET,
    "erase": AuditAction.ERASE,
    "authorize": AuditAction.AUTHORIZE,
}


def _detail_value(value: object) -> AuditDetailValue:
    """Keep detail values scalar by JSON-encoding structured ones."""
    if value is None or isinstance(value, (str, int, bool)):
        return value
    return json.dumps(value, sort_keys=True, separators=(",", ":"))
