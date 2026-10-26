"""LedgerStore operations for graph-backed temporal storage."""

from collections.abc import Sequence

from ontogr_mem.data_models.ledger import LedgerEntry
from ontogr_mem.graphstore.cypher import TenantScope
from ontogr_mem.graphstore.errors import ConstraintViolationError
from ontogr_mem.graphstore.executor import QuerySpec
from ontogr_mem.graphstore.protocol import GraphStore, ScopedRunner
from ontogr_mem.graphstore.types import GraphHandle, Guard, Node
from ontogr_mem.temporal.ledger import GENESIS

from .rows import (
    _LEDGER_COLUMNS,
    _columns,
    _ledger_from,
    _ledger_props,
)
from .scope import (
    LEDGER_CHAIN,
    LEDGER_ENTRY,
)


class LedgerStore:
    """Tenant-bound ledger store persistence operations."""

    def __init__(
        self,
        store: GraphStore,
        handle: GraphHandle,
        runner: ScopedRunner,
        scope: TenantScope,
    ) -> None:
        """Bind standalone ledger operations to one tenant's graph."""
        self._store = store
        self._handle = handle
        self._runner = runner
        self._scope = scope

    def _map(self, body: str = "") -> str:
        return self._scope.pattern(body)

    async def ledger_entries(self, chain: str) -> list[LedgerEntry]:
        """Return the entries of a chain, ordered by sequence number."""
        rows = await self._runner.run(
            QuerySpec(
                f"MATCH (n:{LEDGER_ENTRY} {self._map('chain: $chain')}) "
                f"RETURN {_columns('n', _LEDGER_COLUMNS)} ORDER BY n.seq",
                {"chain": chain},
                read_only=True,
                idempotent=True,
                tag="temporal_ledger_entries",
            )
        )
        return [_ledger_from(row) for row in rows.rows]

    async def ledger_head(self, chain: str) -> tuple[int, str]:
        """Return the last ``(seq, checksum)`` of a standalone chain."""
        rows = await self._runner.run(
            QuerySpec(
                f"MATCH (c:{LEDGER_CHAIN} {self._map('id: $chain')}) "
                "RETURN c.ledger_seq AS seq, c.ledger_head AS head",
                {"chain": chain},
                read_only=True,
                idempotent=True,
                tag="temporal_ledger_head",
            )
        )
        if not rows.rows:
            return (0, GENESIS)
        return (int(rows.rows[0]["seq"]), str(rows.rows[0]["head"]))

    async def append_ledger(
        self, entries: Sequence[LedgerEntry], base_seq: int
    ) -> bool:
        """Append entries to a standalone chain when its head still has `base_seq`."""
        if not entries:
            return True
        chain = entries[0].chain
        if any(entry.chain != chain for entry in entries):
            msg = "ledger entries must all belong to one chain"
            raise ValueError(msg)
        last = max(entries, key=lambda entry: entry.seq)
        guard = Guard(
            label=LEDGER_CHAIN,
            id=chain,
            expect={"ledger_seq": base_seq},
            set={"ledger_seq": last.seq, "ledger_head": last.checksum},
            create_with={"ledger_seq": 0, "ledger_head": GENESIS},
        )
        nodes = [
            Node(id=entry.id, label=LEDGER_ENTRY, props=_ledger_props(entry))
            for entry in entries
        ]
        try:
            return await self._store.guarded_write(self._handle, guard, nodes)
        except ConstraintViolationError:
            return False
