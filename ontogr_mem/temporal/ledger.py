"""A tamper-evident hash chain for fact changes.

Each slot has its own chain. Episode events form one chain per tenant. An entry
stores a checksum computed from its own fields and the previous entry's
checksum, so changing, inserting or deleting a past entry breaks every later
link.

The functions here are pure and take the chain head explicitly. A caller that
appends must first prove the head is still the chain's last entry.
"""

import hashlib
import json
from collections.abc import Mapping, Sequence
from datetime import datetime
from typing import Any

from ontogr_mem.data_models.ledger import LedgerEntry, LedgerOp
from ontogr_mem.data_models.timestamps import to_micros


GENESIS = "0" * 64
"""The checksum a first entry links to."""

TENANT_CHAIN = "tenant"
"""Chain key of the episode events of a tenant."""


def slot_chain(slot_key: str) -> str:
    """Return the chain key of a slot's fact history."""
    return f"slot:{slot_key}"


def payload_hash(payload: Mapping[str, Any]) -> str:
    """Return the SHA-256 of the payload as canonical JSON.

    Keys are sorted and separators are compact, so two mappings with the same
    content hash the same. The payload itself is not stored in the entry.

    Args:
        payload: Fields the entry commits to.

    Returns:
        Hex SHA-256 of the canonical JSON form.
    """
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(blob.encode()).hexdigest()


def checksum(  # noqa: PLR0917
    seq: int,
    ts: datetime,
    op: LedgerOp,
    subject_id: str,
    actor: str,
    payload_digest: str,
    prev_checksum: str,
) -> str:
    """Return the checksum of one entry.

    Time is hashed as integer microseconds, so the value does not depend on how
    a datetime is formatted.

    Args:
        seq: Position in the chain.
        ts: When the entry was written.
        op: What the entry records.
        subject_id: Fact id or slot key the entry is about.
        actor: Who performed the write.
        payload_digest: Output of ``payload_hash`` for the entry payload.
        prev_checksum: Checksum of the previous entry, or ``GENESIS``.

    Returns:
        Hex SHA-256 over the entry fields and the previous checksum.
    """
    parts = [
        seq,
        to_micros(ts),
        op.value,
        subject_id,
        actor,
        payload_digest,
        prev_checksum,
    ]
    blob = json.dumps(parts, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(blob.encode()).hexdigest()


def link(  # noqa: PLR0917
    chain: str,
    head: tuple[int, str],
    ts: datetime,
    op: LedgerOp,
    subject_id: str,
    actor: str,
    payload: Mapping[str, Any],
) -> LedgerEntry:
    """Build the entry that follows ``head``.

    Args:
        chain: Chain key: a slot key or a tenant chain key.
        head: Last ``(seq, checksum)`` of the chain, or ``(0, GENESIS)``.
        ts: When the entry was written.
        op: What the entry records.
        subject_id: Fact id or slot key the entry is about.
        actor: Who performed the write.
        payload: Fields the entry commits to.

    Returns:
        An entry with sequence ``head[0] + 1`` linked to ``head[1]``.
    """
    seq = head[0] + 1
    prev_checksum = head[1]
    digest = payload_hash(payload)
    return LedgerEntry(
        chain=chain,
        seq=seq,
        ts=ts,
        op=op,
        subject_id=subject_id,
        actor=actor,
        payload_hash=digest,
        prev_checksum=prev_checksum,
        checksum=checksum(
            seq,
            ts,
            op,
            subject_id,
            actor,
            digest,
            prev_checksum,
        ),
    )


def verify(entries: Sequence[LedgerEntry]) -> list[str]:
    """Return problems in one chain, oldest entry first.

    An empty list means the chain is intact. Three checks run per entry: the
    sequence follows the previous one, ``prev_checksum`` links to the previous
    entry, and ``checksum`` matches a fresh computation.

    Args:
        entries: Entries of one chain, in any order.

    Returns:
        A readable problem for each broken link, or an empty list.
    """
    problems: list[str] = []
    previous_seq, previous_checksum = 0, GENESIS
    for entry in sorted(entries, key=lambda item: item.seq):
        if entry.seq != previous_seq + 1:
            problems.append(f"{entry.chain}: seq {entry.seq} follows {previous_seq}")
        if entry.prev_checksum != previous_checksum:
            problems.append(f"{entry.chain}#{entry.seq}: prev_checksum does not link")
        expected = checksum(
            entry.seq,
            entry.ts,
            entry.op,
            entry.subject_id,
            entry.actor,
            entry.payload_hash,
            entry.prev_checksum,
        )
        if entry.checksum != expected:
            problems.append(f"{entry.chain}#{entry.seq}: checksum does not match")
        previous_seq, previous_checksum = entry.seq, entry.checksum
    return problems
