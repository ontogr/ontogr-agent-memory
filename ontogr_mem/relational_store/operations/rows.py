"""Shared row decoding for operation store adapters."""

from datetime import datetime
from typing import Any

from ontogr_mem.data_models.timestamps import from_micros
from ontogr_mem.data_models.usage import OperationUsage
from ontogr_mem.relational_store.operations.models import (
    OperationKind,
    OperationOutcome,
    OperationRecord,
)


OPERATION_COLUMNS = (
    "id, parent_id, kind, principal_id, tenant, session_id, trace_id, "
    "started_at, ended_at, outcome, error_class, error, tokens_in, tokens_out, "
    "cost_usd"
)


def _datetime(value: int) -> datetime:
    """Return a finite UTC timestamp, refusing the open-end sentinel."""
    result = from_micros(value)
    if result is None:
        raise ValueError("operation timestamp cannot be an open end")
    return result


def record_from_row(row: Any) -> OperationRecord:
    """Decode an operation record from selected columns by position."""
    ended_at = from_micros(int(row[8]))
    return OperationRecord(
        id=str(row[0]),
        parent_id=None if row[1] is None else str(row[1]),
        kind=OperationKind(str(row[2])),
        principal_id=str(row[3]),
        tenant=None if row[4] is None else str(row[4]),
        session_id=None if row[5] is None else str(row[5]),
        trace_id=str(row[6]),
        started_at=_datetime(int(row[7])),
        ended_at=ended_at,
        outcome=OperationOutcome(str(row[9])),
        error_class=None if row[10] is None else str(row[10]),
        error=None if row[11] is None else str(row[11]),
        usage=OperationUsage(
            tokens_in=int(row[12]),
            tokens_out=int(row[13]),
            cost_usd=None if row[14] is None else float(row[14]),
        ),
    )
