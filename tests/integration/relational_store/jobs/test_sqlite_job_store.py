"""Exercise the job contract on real SQLite files that remain in pytest tmp dirs."""

import asyncio
import sqlite3
from dataclasses import replace

import pytest

from ontogr_mem.data_models.jobs import JobProgress, JobState
from ontogr_mem.relational_store.sqlite import SqliteJobStore
from tests.contracts.erase_journal import (
    EraseJournalContract,
    make_erasure,
)
from tests.contracts.job_store import JobStoreContract, make_job


class TestSqliteJobStore(JobStoreContract):
    """SQLite keeps the same storage semantics across threads and reopenings."""

    @pytest.fixture
    async def make_store(self, tmp_path):
        """Close every isolated database opened by the contract factory."""
        stores = []

        async def make():
            store = SqliteJobStore(str(tmp_path / f"state-{len(stores)}.sqlite3"))
            stores.append(store)
            return store

        yield make
        for store in stores:
            await store.close()

    async def test_jobs_survive_reopening_the_file(self, tmp_path):
        """Terminal results and interrupted attempts persist across reopenings."""
        path = str(tmp_path / "restart.sqlite3")
        first = replace(make_job(), status=JobState.DONE, result={"ok": True})
        second = replace(
            make_job("j2"),
            idempotency_key=None,
            status=JobState.RUNNING,
            attempts=1,
        )
        store = SqliteJobStore(path)
        try:
            await store.create_job(first)
            await store.create_job(make_job("j2", "tenant_b"))
            await store.save_job(replace(second, tenant_id="tenant_b"))
        finally:
            await store.close()

        reopened = SqliteJobStore(path)
        try:
            assert await reopened.get_job("j1") == first
            assert await reopened.get_job("j2") == replace(second, tenant_id="tenant_b")
            assert await reopened.requeue_running() == [
                replace(second, tenant_id="tenant_b", status=JobState.QUEUED)
            ]
        finally:
            await reopened.close()

    async def test_job_progress_survives_reopening_the_file(self, tmp_path):
        """A progress update remains readable after the SQLite store reopens."""
        path = str(tmp_path / "progress.sqlite3")
        store = SqliteJobStore(path)
        job = make_job()
        progress = JobProgress(stage="export", done=100, total=250)
        try:
            await store.create_job(job)
            await store.set_job_progress(job.job_id, progress)
        finally:
            await store.close()

        reopened = SqliteJobStore(path)
        try:
            stored = await reopened.get_job(job.job_id)
            assert stored is not None
            assert stored.progress == progress
        finally:
            await reopened.close()

    async def test_concurrent_creates_with_one_key_make_one_job(self, store):
        """Competing requests have one winner and return the same stored id."""
        results = await asyncio.gather(
            *(store.create_job(make_job(f"j{i}")) for i in range(20))
        )

        assert sum(not replay for _, replay in results) == 1
        assert sum(replay for _, replay in results) == 19
        assert len({job.job_id for job, _ in results}) == 1
        assert await store.count_jobs(["tenant_a"], [JobState.QUEUED]) == 1

    def test_a_database_error_is_not_wrapped(self, tmp_path):
        """An invalid database path remains a sqlite3 error."""
        with pytest.raises(sqlite3.OperationalError):
            SqliteJobStore(str(tmp_path / "missing" / "state.sqlite3"))

    async def test_close_is_idempotent(self, tmp_path):
        """Explicit closure and fixture cleanup may both close a store."""
        store = SqliteJobStore(str(tmp_path / "closed.sqlite3"))
        await store.close()
        await store.close()

        with pytest.raises(sqlite3.ProgrammingError):
            await store.get_job("j1")


class TestSqliteEraseJournal(EraseJournalContract):
    """The SQLite state file retains running erase rows after restart."""

    @pytest.fixture
    async def make_store(self, tmp_path):
        """Close every state connection opened by a contract case."""
        stores = []

        async def make():
            store = SqliteJobStore(str(tmp_path / f"erase-{len(stores)}.sqlite3"))
            stores.append(store)
            return store

        yield make
        for store in stores:
            await store.close()

    async def test_running_row_survives_reopening_the_state_file(self, tmp_path):
        """A process restart leaves the journal ready for recovery."""
        path = str(tmp_path / "erase-recovery.sqlite3")
        store = SqliteJobStore(path)
        record = make_erasure()
        try:
            await store.create(record)
            await store.step(record.erasure_id, "retract")
        finally:
            await store.close()

        reopened = SqliteJobStore(path)
        try:
            stored = await reopened.get(record.erasure_id)
            assert stored is not None
            assert stored.status.value == "running"
            assert stored.last_step == "retract"
            assert await reopened.running(record.tenant) == [stored]
        finally:
            await reopened.close()
