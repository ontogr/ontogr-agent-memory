"""Hash links detect edits and deletions in the middle of the chain.

A database owner can rewrite every later hash and leave a chain that verifies.
Deleting the last row is also undetectable. Audit rows hold ids, actions and
counts, never memory text.
"""

import hashlib
import json
from collections.abc import Mapping, Sequence
from datetime import datetime
from typing import Final

from ontogr_mem.data_models.timestamps import to_micros
from ontogr_mem.relational_store.audit.models import (
    AuditDetailValue,
    AuditEntry,
    AuditRow,
)


GENESIS_HASH: Final = "0" * 64


def canonical_detail(detail: Mapping[str, AuditDetailValue]) -> str:
    """Encode detail with stable key ordering and compact JSON separators."""
    return json.dumps(dict(detail), sort_keys=True, separators=(",", ":"))


def chain_hash(previous_hash: str, recorded_at: datetime, entry: AuditEntry) -> str:
    """Hash the previous link and the canonical stored fields of one entry.

    Raises:
        ValueError: If ``recorded_at`` is not timezone-aware.
    """
    fields = [
        previous_hash,
        to_micros(recorded_at),
        entry.action.value,
        entry.outcome.value,
        entry.principal_id,
        entry.tenant,
        entry.acting_for,
        entry.target,
        entry.trace_id,
        canonical_detail(entry.detail),
    ]
    encoded = json.dumps(fields, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def first_broken_seq(rows: Sequence[AuditRow]) -> int | None:
    """Return the first row with an invalid sequence, previous link, or own hash."""
    previous_hash = GENESIS_HASH
    for expected_seq, row in enumerate(rows, start=1):
        if row.seq != expected_seq:
            return row.seq
        if row.previous_hash != previous_hash:
            return row.seq
        if chain_hash(previous_hash, row.recorded_at, row.entry) != row.row_hash:
            return row.seq
        previous_hash = row.row_hash
    return None
