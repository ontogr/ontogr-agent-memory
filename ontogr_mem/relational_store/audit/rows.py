"""Shared row decoding for audit log adapters."""

import json
from typing import Any

from ontogr_mem.data_models.timestamps import from_micros
from ontogr_mem.relational_store.audit.models import (
    AuditAction,
    AuditEntry,
    AuditOutcome,
    AuditRow,
)


AUDIT_COLUMNS = (
    "seq, recorded_at, action, outcome, principal_id, tenant, acting_for, "
    "target, trace_id, detail, previous_hash, row_hash"
)


def audit_row_from_row(row: Any) -> AuditRow:
    """Rebuild an immutable audit row from selected columns by position."""
    recorded_at = from_micros(row[1])
    if recorded_at is None:
        raise ValueError("audit timestamp cannot be an open end")
    detail = json.loads(row[9])
    if not isinstance(detail, dict):
        raise ValueError("stored audit detail must be a JSON object")
    entry = AuditEntry(
        action=AuditAction(row[2]),
        outcome=AuditOutcome(row[3]),
        principal_id=row[4],
        tenant=row[5],
        acting_for=row[6],
        target=row[7],
        trace_id=row[8],
        detail=detail,
    )
    return AuditRow(
        seq=row[0],
        recorded_at=recorded_at,
        entry=entry,
        previous_hash=row[10],
        row_hash=row[11],
    )
