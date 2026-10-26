"""Hash-chain verification against rows produced by the memory adapter."""

from dataclasses import replace
from datetime import datetime

import pytest

from ontogr_mem.relational_store.audit import AuditOutcome
from ontogr_mem.relational_store.audit.chain import canonical_detail, first_broken_seq
from ontogr_mem.relational_store.memory.audit import MemoryAuditLog
from tests.contracts.audit_log import make_entry
from tests.support.clock import FixedClock


class TestAuditChain:
    """The chain identifies the first row changed or removed."""

    @pytest.mark.parametrize(
        ("edit", "expected"),
        [
            ("outcome", 2),
            ("delete", 3),
            ("swap", 3),
            ("recorded_at", 1),
            ("detail", 2),
        ],
    )
    async def test_reports_first_broken_row(self, edit: str, expected: int):
        """An edit or deletion is reported at the first broken link."""
        log = MemoryAuditLog(FixedClock())
        for index in range(1, 4):
            await log.append(make_entry(target=f"target-{index}", detail={"n": index}))
        rows = await log.list_rows(limit=3)
        rows.reverse()

        if edit == "outcome":
            rows[1] = replace(
                rows[1],
                entry=replace(rows[1].entry, outcome=AuditOutcome.DENIED),
            )
        elif edit == "delete":
            del rows[1]
        elif edit == "swap":
            rows[1], rows[2] = rows[2], rows[1]
        elif edit == "recorded_at":
            rows[0] = replace(rows[0], recorded_at=rows[0].recorded_at.replace(day=22))
        else:
            rows[1] = replace(
                rows[1],
                entry=replace(rows[1].entry, detail={"n": 200}),
            )

        assert first_broken_seq(rows) == expected

    async def test_intact_and_empty_chains_have_no_broken_row(self):
        """An intact chain and an empty sequence both verify cleanly."""
        log = MemoryAuditLog(FixedClock())
        await log.append(make_entry())

        assert first_broken_seq(await log.list_rows(limit=1)) is None
        assert first_broken_seq([]) is None

    async def test_first_row_must_link_to_genesis(self):
        """A first row linked to another hash is the first broken row."""
        log = MemoryAuditLog(FixedClock())
        row = await log.append(make_entry())

        assert first_broken_seq([replace(row, previous_hash="f" * 64)]) == 1

    async def test_renumbered_rows_break_sequence_continuity(self):
        """Rows must retain consecutive positions even when hash links are intact."""
        log = MemoryAuditLog(FixedClock())
        for index in range(1, 4):
            await log.append(make_entry(target=f"target-{index}"))

        rows = await log.list_rows(limit=3)
        rows.reverse()
        renumbered_rows = [replace(row, seq=row.seq + 10) for row in rows]

        assert first_broken_seq(renumbered_rows) == 11

    async def test_deleted_final_row_is_not_detected(self):
        """A missing tail remains undetectable until a later row is appended."""
        log = MemoryAuditLog(FixedClock())
        for index in range(1, 4):
            await log.append(make_entry(target=f"target-{index}"))

        rows = await log.list_rows(limit=3)
        rows = [row for row in rows if row.seq != 3]

        assert first_broken_seq(list(reversed(rows))) is None

    def test_canonical_detail_does_not_depend_on_mapping_order(self):
        """Equivalent detail mappings have the same canonical JSON."""
        assert canonical_detail({"a": 1, "b": True}) == canonical_detail(
            {"b": True, "a": 1}
        )

    async def test_naive_clock_time_is_rejected_without_appending(self):
        """A naive datetime is rejected before it can become a chain row."""
        log = MemoryAuditLog(FixedClock(datetime(2026, 10, 21, 9, 0)))

        with pytest.raises(ValueError, match="datetime must have a timezone"):
            await log.append(make_entry())

        assert await log.list_rows() == []
