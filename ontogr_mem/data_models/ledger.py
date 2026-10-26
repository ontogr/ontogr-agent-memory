"""Ledger rows: the hash chain that records every write of the temporal layer.

An entry holds ids and hashes only, never episode text.
"""

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum


class LedgerOp(StrEnum):
    """What a ledger entry records."""

    FACT_ADDED = "fact_added"
    FACT_VERSIONED = "fact_versioned"
    FACT_SUPERSEDED = "fact_superseded"
    FACT_WITHDRAWN = "fact_withdrawn"
    EPISODE_COMMITTED = "episode_committed"
    EPISODE_RETRACTED = "episode_retracted"


@dataclass(frozen=True, slots=True)
class LedgerEntry:
    """One link of a hash chain.

    ``checksum`` covers this entry and ``prev_checksum``, so changing an earlier
    entry breaks the check for every later one. An entry holds ids and hashes
    only, never episode text.

    Attributes:
        chain: Chain key: a slot key or a tenant chain key.
        seq: Position in the chain, starting at 1.
        ts: When the entry was written.
        op: What the entry records.
        subject_id: Fact id or slot key the entry is about.
        actor: Who performed the write.
        payload_hash: Hash of the payload the entry commits to.
        prev_checksum: Checksum of the previous entry, or ``GENESIS``.
        checksum: Hash of this entry and ``prev_checksum``.
    """

    chain: str
    seq: int
    ts: datetime
    op: LedgerOp
    subject_id: str
    actor: str
    payload_hash: str
    prev_checksum: str
    checksum: str

    @property
    def id(self) -> str:
        """Identity of the entry inside its chain."""
        return f"{self.chain}:{self.seq}"
