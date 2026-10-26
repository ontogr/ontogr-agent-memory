"""Operation rows persist in file-backed SQLite databases."""

import sqlite3
from collections.abc import AsyncIterator

import pytest

from ontogr_mem.data_models.clock import Clock
from ontogr_mem.errors import StoreError
from ontogr_mem.relational_store import (
    OperationKind,
    OperationOutcome,
    OperationStore,
)
from ontogr_mem.relational_store.sqlite.operations import SqliteOperationStore
from tests.contracts.operation_store import OperationStoreContract
from tests.support.clock import FixedClock


class TestSqliteOperationStore(OperationStoreContract):
    """SQLite preserves operation contract behavior and restart outcomes."""

    @pytest.fixture
    def clock(self) -> Clock:
        """Return a stable clock for stored operation timestamps."""
        return FixedClock()

    @pytest.fixture
    async def operation_store(
        self, tmp_path, clock: Clock
    ) -> AsyncIterator[OperationStore]:
        """Open an isolated SQLite file for one contract case."""
        connection = sqlite3.connect(
            str(tmp_path / "operations.sqlite3"), isolation_level=None
        )
        store = SqliteOperationStore(connection, clock)
        try:
            yield store
        finally:
            connection.close()

    async def test_rows_left_running_are_failed_when_the_store_opens(
        self, tmp_path, clock: Clock
    ) -> None:
        """Reopening the file marks unfinished rows interrupted, once."""
        path = str(tmp_path / "restart.sqlite3")
        connection = sqlite3.connect(path, isolation_level=None)
        first = SqliteOperationStore(connection, clock)
        running_id = await first.start(
            kind=OperationKind.WRITE,
            principal_id="owner",
            tenant="acme",
        )
        finished_id = await first.start(
            kind=OperationKind.RETRIEVE,
            principal_id="owner",
            tenant="acme",
        )
        await first.finish(finished_id, outcome=OperationOutcome.OK)
        connection.close()

        reopened_connection = sqlite3.connect(path, isolation_level=None)
        try:
            reopened = SqliteOperationStore(reopened_connection, clock)

            running = await reopened.get(running_id)
            finished = await reopened.get(finished_id)

            assert running is not None
            assert (running.outcome, running.error_class) == (
                OperationOutcome.FAILED,
                "Interrupted",
            )
            assert running.error == "operation was still running when the store opened"
            assert running.ended_at == clock.now()
            assert finished is not None
            assert (finished.outcome, finished.ended_at) == (
                OperationOutcome.OK,
                clock.now(),
            )
        finally:
            reopened_connection.close()

    async def test_closed_connection_raises_store_error_with_sqlite_cause(
        self, tmp_path, clock: Clock
    ) -> None:
        """SQLite failures retain their driver error as the chained cause."""
        connection = sqlite3.connect(
            str(tmp_path / "closed.sqlite3"), isolation_level=None
        )
        store = SqliteOperationStore(connection, clock)
        connection.close()

        with pytest.raises(StoreError) as error:
            await store.start(
                kind=OperationKind.WRITE,
                principal_id="owner",
                tenant="acme",
            )

        assert isinstance(error.value.__cause__, sqlite3.Error)
