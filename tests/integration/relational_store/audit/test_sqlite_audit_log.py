"""Audit-chain persistence and owner edits on file-backed SQLite databases."""

import sqlite3

import pytest

from ontogr_mem.data_models.timestamps import from_micros
from ontogr_mem.errors import StoreError
from ontogr_mem.relational_store.audit import AuditLog, AuditOutcome
from ontogr_mem.relational_store.audit.chain import chain_hash
from ontogr_mem.relational_store.sqlite.audit import SqliteAuditLog
from tests.contracts.audit_log import AuditLogContract, make_entry
from tests.support.clock import FixedClock


class TestSqliteAuditLog(AuditLogContract):
    """SQLite keeps the contract across a live file-backed connection."""

    @pytest.fixture
    def clock(self) -> FixedClock:
        """Return the fixed time source for one contract case."""
        return FixedClock()

    @pytest.fixture
    def connection(self, tmp_path):
        """Open an isolated SQLite file in autocommit mode."""
        connection = sqlite3.connect(
            str(tmp_path / "audit.sqlite3"), isolation_level=None
        )
        try:
            yield connection
        finally:
            connection.close()

    @pytest.fixture
    def audit_log(self, connection, clock: FixedClock) -> AuditLog:
        """Return a SQLite audit log on the fixture's isolated database."""
        return SqliteAuditLog(connection, clock)

    async def test_rows_survive_reopening_the_database(self, tmp_path):
        """Rows and their chain links remain intact after closing and reopening."""
        path = str(tmp_path / "restart.sqlite3")
        first_connection = sqlite3.connect(path, isolation_level=None)
        first_log = SqliteAuditLog(first_connection, FixedClock())
        try:
            for index in range(1, 4):
                await first_log.append(make_entry(target=f"target-{index}"))
            before = await first_log.list_rows(limit=3)
        finally:
            first_connection.close()

        reopened_connection = sqlite3.connect(path, isolation_level=None)
        try:
            reopened = SqliteAuditLog(reopened_connection, FixedClock())
            assert await reopened.list_rows(limit=3) == before
            assert await reopened.verify() is None
        finally:
            reopened_connection.close()

    async def test_external_outcome_edit_is_detected(self, tmp_path):
        """An owner edit without recomputing hashes breaks the edited row."""
        path = str(tmp_path / "edited.sqlite3")
        connection = sqlite3.connect(path, isolation_level=None)
        log = SqliteAuditLog(connection, FixedClock())
        try:
            for index in range(1, 4):
                await log.append(make_entry(target=f"target-{index}"))
            owner = sqlite3.connect(path)
            try:
                owner.execute("UPDATE audit_rows SET outcome = 'ok' WHERE seq = 2")
                owner.commit()
            finally:
                owner.close()

            assert await log.verify() == 2
        finally:
            connection.close()

    async def test_malformed_detail_is_reported_as_a_broken_row(self, tmp_path):
        """A row that cannot be decoded is still reported by verification."""
        path = str(tmp_path / "malformed.sqlite3")
        connection = sqlite3.connect(path, isolation_level=None)
        log = SqliteAuditLog(connection, FixedClock())
        try:
            for index in range(1, 4):
                await log.append(make_entry(target=f"target-{index}"))
            owner = sqlite3.connect(path, isolation_level=None)
            try:
                owner.execute("UPDATE audit_rows SET detail = '[]' WHERE seq = 2")
            finally:
                owner.close()

            assert await log.verify() == 2
        finally:
            connection.close()

    async def test_external_middle_row_deletion_is_detected(self, tmp_path):
        """Deleting a middle row breaks the following row's previous hash."""
        path = str(tmp_path / "deleted.sqlite3")
        connection = sqlite3.connect(path, isolation_level=None)
        log = SqliteAuditLog(connection, FixedClock())
        try:
            for index in range(1, 4):
                await log.append(make_entry(target=f"target-{index}"))
            owner = sqlite3.connect(path)
            try:
                owner.execute("DELETE FROM audit_rows WHERE seq = 2")
                owner.commit()
            finally:
                owner.close()

            assert await log.verify() == 3
        finally:
            connection.close()

    async def test_renumbered_rows_break_sequence_continuity(self, tmp_path):
        """Renumbering rows without changing their order breaks verification."""
        path = str(tmp_path / "renumbered.sqlite3")
        connection = sqlite3.connect(path, isolation_level=None)
        log = SqliteAuditLog(connection, FixedClock())
        try:
            for index in range(1, 4):
                await log.append(make_entry(target=f"target-{index}"))
            connection.execute("UPDATE audit_rows SET seq = seq + 100")

            assert await log.verify() == 101
        finally:
            connection.close()

    async def test_external_final_row_deletion_is_not_detected(self, tmp_path):
        """Deleting the final row cannot be detected from the remaining chain."""
        path = str(tmp_path / "deleted-tail.sqlite3")
        connection = sqlite3.connect(path, isolation_level=None)
        log = SqliteAuditLog(connection, FixedClock())
        try:
            for index in range(1, 4):
                await log.append(make_entry(target=f"target-{index}"))
            connection.execute("DELETE FROM audit_rows WHERE seq = 3")

            assert await log.verify() is None
            await log.append(make_entry(target="target-4"))

            assert await log.verify() == 4
        finally:
            connection.close()

    async def test_owner_who_rewrites_every_later_hash_passes_verify(self, tmp_path):
        """An owner can edit a row and recompute every later hash undetected."""
        path = str(tmp_path / "rewritten.sqlite3")
        clock = FixedClock()
        connection = sqlite3.connect(path, isolation_level=None)
        log = SqliteAuditLog(connection, clock)
        try:
            for index in range(1, 4):
                await log.append(make_entry(target=f"target-{index}"))

            rows = connection.execute(
                "SELECT seq, recorded_at, previous_hash, row_hash FROM audit_rows "
                "ORDER BY seq"
            ).fetchall()
            row_one_hash = rows[0][3]
            second_entry = make_entry(target="target-2", outcome=AuditOutcome.DENIED)
            second_time = from_micros(rows[1][1])
            assert second_time is not None
            second_hash = chain_hash(row_one_hash, second_time, second_entry)
            connection.execute(
                "UPDATE audit_rows SET outcome = ?, previous_hash = ?, row_hash = ? "
                "WHERE seq = 2",
                (AuditOutcome.DENIED.value, row_one_hash, second_hash),
            )

            third_entry = make_entry(target="target-3")
            third_time = from_micros(rows[2][1])
            assert third_time is not None
            third_hash = chain_hash(second_hash, third_time, third_entry)
            connection.execute(
                "UPDATE audit_rows SET previous_hash = ?, row_hash = ? WHERE seq = 3",
                (second_hash, third_hash),
            )

            assert await log.verify() is None
        finally:
            connection.close()

    async def test_append_on_a_closed_connection_raises_store_error(self, tmp_path):
        """A closed database connection becomes a chained store failure."""
        connection = sqlite3.connect(
            str(tmp_path / "closed.sqlite3"), isolation_level=None
        )
        log = SqliteAuditLog(connection, FixedClock())
        connection.close()

        with pytest.raises(StoreError) as error:
            await log.append(make_entry())

        assert isinstance(error.value.__cause__, sqlite3.Error)
