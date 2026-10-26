"""Audit event vocabulary and immutable row models."""

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from types import MappingProxyType
from typing import TypeAlias

from ontogr_mem.data_models.tenant import validate_tenant_id


AuditDetailValue: TypeAlias = str | int | bool | None


class AuditAction(StrEnum):
    """Action recorded by an audit event."""

    AUTHORIZE = "authorize"
    GRANT_ADDED = "grant_added"
    GRANT_REMOVED = "grant_removed"
    EXPORT = "export"
    FORGET = "forget"
    ERASE = "erase"


class AuditOutcome(StrEnum):
    """Outcome recorded by an audit event."""

    ALLOWED = "allowed"
    DENIED = "denied"
    OK = "ok"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class AuditEntry:
    """An immutable audit event with caller identity and non-content detail.

    The detail mapping is copied on creation. ``principal_id`` must not be
    empty, and a non-None Tenant id must be valid.
    """

    action: AuditAction
    outcome: AuditOutcome
    principal_id: str
    tenant: str | None = None
    target: str = ""
    acting_for: str | None = None
    trace_id: str = ""
    detail: Mapping[str, AuditDetailValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate the principal and optional Tenant id."""
        if not self.principal_id:
            raise ValueError("principal_id must not be empty")
        if self.tenant is not None:
            validate_tenant_id(self.tenant)
        object.__setattr__(self, "detail", MappingProxyType(dict(self.detail)))


@dataclass(frozen=True, slots=True)
class AuditRow:
    """One persisted event and its position in the audit chain."""

    seq: int
    recorded_at: datetime
    entry: AuditEntry
    previous_hash: str
    row_hash: str
