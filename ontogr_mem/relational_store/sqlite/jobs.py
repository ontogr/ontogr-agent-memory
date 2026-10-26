"""SQLite job persistence with serialized worker-thread access."""

import asyncio
import sqlite3
import threading
from collections.abc import Callable, Sequence
from datetime import datetime
from typing import TypeVar

from ontogr_mem.data_models.jobs import Job, JobProgress, JobState
from ontogr_mem.data_models.tenant import validate_tenant_id
from ontogr_mem.data_models.timestamps import to_micros
from ontogr_mem.relational_store.engine import open_sqlite_connection
from ontogr_mem.relational_store.jobs.erase_journal import (
    ErasureRecord,
    ErasureStatus,
)
from ontogr_mem.relational_store.jobs.errors import IdempotencyConflictError
from ontogr_mem.relational_store.jobs.rows import (
    dumps,
    erasure_from_row,
    job_from_row,
)
from ontogr_mem.relational_store.schema.runner import ensure_sqlite_schema


T = TypeVar("T")


def _values(job: Job) -> tuple[object, ...]:
    """Snapshot mutable fields before handing a write to a worker thread."""
    return (
        job.tenant_id,
        job.op,
        dumps(job.payload),
        job.request_hash,
        job.idempotency_key,
        job.status.value,
        job.attempts,
        None if job.result is None else dumps(job.result),
        None if job.error is None else dumps(job.error),
        to_micros(job.created_at),
        to_micros(job.updated_at),
        job.trace_id,
        None if job.progress is None else job.progress.stage,
        None if job.progress is None else job.progress.done,
        None if job.progress is None else job.progress.total,
        job.job_id,
    )


class SqliteJobStore:
    """Keep durable jobs in one SQLite connection guarded by a thread lock.

    Args:
        path: Database file path; its parent directory must already exist.
    """

    def __init__(self, path: str) -> None:
        """Open the database and apply its schema migrations."""
        self._lock = threading.Lock()
        self._closed = False
        self._db = open_sqlite_connection(path, autocommit=False)
        self._db.row_factory = sqlite3.Row
        try:
            ensure_sqlite_schema(self._db)
        except BaseException:
            self._db.close()
            raise

    async def create_job(self, job: Job) -> tuple[Job, bool]:
        """Atomically insert a job or replay a matching Tenant request key."""
        values = _values(job)

        def create(db: sqlite3.Connection) -> tuple[Job, bool]:
            # Reserve the writer before reading, including across connections.
            db.execute("BEGIN IMMEDIATE")
            if job.idempotency_key is not None:
                row = db.execute(
                    "SELECT * FROM jobs WHERE tenant_id = ? AND idempotency_key = ?",
                    (job.tenant_id, job.idempotency_key),
                ).fetchone()
                if row is not None:
                    stored = job_from_row(row)
                    if stored.request_hash != job.request_hash:
                        raise IdempotencyConflictError(
                            "idempotency key has a different request"
                        )
                    return stored, True
            db.execute(
                "INSERT INTO jobs (tenant_id, op, payload, request_hash, "
                "idempotency_key, status, attempts, result, error, created_at, "
                "updated_at, trace_id, progress_stage, progress_done, "
                "progress_total, job_id) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                values,
            )
            row = db.execute(
                "SELECT * FROM jobs WHERE job_id = ?", (job.job_id,)
            ).fetchone()
            return job_from_row(row), False

        return await asyncio.to_thread(lambda: self._locked(create))

    async def get_job(self, job_id: str) -> Job | None:
        """Return a detached job or None for an unknown id."""

        def get(db: sqlite3.Connection) -> Job | None:
            row = db.execute(
                "SELECT * FROM jobs WHERE job_id = ?", (job_id,)
            ).fetchone()
            return None if row is None else job_from_row(row)

        return await asyncio.to_thread(lambda: self._locked(get))

    async def save_job(self, job: Job) -> None:
        """Replace an existing record without storing its replay response flag."""
        values = _values(job)

        def save(db: sqlite3.Connection) -> None:
            cursor = db.execute(
                "UPDATE jobs SET tenant_id = ?, op = ?, payload = ?, "
                "request_hash = ?, idempotency_key = ?, status = ?, attempts = ?, "
                "result = ?, error = ?, created_at = ?, updated_at = ?, trace_id = ? "
                "WHERE job_id = ?",
                values[:12] + (values[15],),
            )
            if cursor.rowcount == 0:
                raise KeyError(job.job_id)

        await asyncio.to_thread(lambda: self._locked(save))

    async def set_job_progress(self, job_id: str, progress: JobProgress) -> None:
        """Persist monotonic progress inside a serialized SQLite transaction."""

        def update(db: sqlite3.Connection) -> None:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute(
                "SELECT progress_done FROM jobs WHERE job_id = ?", (job_id,)
            ).fetchone()
            if row is None:
                raise KeyError(job_id)
            previous_done = row["progress_done"]
            if previous_done is not None and progress.done < previous_done:
                raise ValueError("job progress cannot move backwards")
            db.execute(
                "UPDATE jobs SET progress_stage = ?, progress_done = ?, "
                "progress_total = ? WHERE job_id = ?",
                (progress.stage, progress.done, progress.total, job_id),
            )

        await asyncio.to_thread(lambda: self._locked(update))

    async def requeue_running(self) -> list[Job]:
        """Requeue interrupted work and return queued jobs by creation time."""

        def requeue(db: sqlite3.Connection) -> list[Job]:
            db.execute("UPDATE jobs SET status = 'queued' WHERE status = 'running'")
            return [
                job_from_row(row)
                for row in db.execute(
                    "SELECT * FROM jobs WHERE status = 'queued' "
                    "ORDER BY created_at, seq"
                )
            ]

        return await asyncio.to_thread(lambda: self._locked(requeue))

    async def count_jobs(
        self, tenant_ids: Sequence[str], statuses: Sequence[JobState]
    ) -> int:
        """Count only jobs matching both selected Tenants and states."""
        tenants = tuple(set(tenant_ids))
        states = tuple({status.value for status in statuses})
        if not tenants or not states:
            return 0

        def count(db: sqlite3.Connection) -> int:
            tenant_slots = ",".join("?" for _ in tenants)
            state_slots = ",".join("?" for _ in states)
            row = db.execute(
                f"SELECT COUNT(*) FROM jobs WHERE tenant_id IN ({tenant_slots}) "
                f"AND status IN ({state_slots})",
                tenants + states,
            ).fetchone()
            return row[0]

        return await asyncio.to_thread(lambda: self._locked(count))

    async def tenant_ids(self) -> list[str]:
        """Return sorted distinct Tenants with persisted jobs."""

        def tenants(db: sqlite3.Connection) -> list[str]:
            return [
                row[0]
                for row in db.execute(
                    "SELECT DISTINCT tenant_id FROM jobs ORDER BY tenant_id"
                )
            ]

        return await asyncio.to_thread(lambda: self._locked(tenants))

    async def delete_tenant(self, tenant_id: str) -> int:
        """Delete a Tenant's jobs and their idempotency keys."""
        validate_tenant_id(tenant_id)

        def delete(db: sqlite3.Connection) -> int:
            cursor = db.execute("DELETE FROM jobs WHERE tenant_id = ?", (tenant_id,))
            return cursor.rowcount

        return await asyncio.to_thread(lambda: self._locked(delete))

    async def create(self, record: ErasureRecord) -> str:
        """Store a running erase row and return its id."""
        if record.status is not ErasureStatus.RUNNING:
            raise ValueError("new erase journal rows must be running")

        def create(db: sqlite3.Connection) -> str:
            db.execute(
                "INSERT INTO erasures (erasure_id, tenant, mode, principal_id, "
                "selector_kind, target, episode_ids, fact_ids, observation_ids, "
                "entity_ids, created_at, status, last_step, finished_at, residue, "
                "counts) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, ?, ?)",
                (
                    record.erasure_id,
                    record.tenant,
                    record.mode,
                    record.principal_id,
                    record.selector_kind,
                    record.target,
                    dumps(record.episode_ids),
                    dumps(record.fact_ids),
                    dumps(record.observation_ids),
                    dumps(record.entity_ids),
                    to_micros(record.created_at),
                    record.status.value,
                    record.last_step,
                    dumps(record.residue),
                    dumps(record.counts),
                ),
            )
            return record.erasure_id

        return await asyncio.to_thread(lambda: self._locked(create))

    async def step(self, erasure_id: str, step: str) -> None:
        """Record the last completed step of a running erase row."""
        if not step:
            raise ValueError("erase step must not be empty")

        def update(db: sqlite3.Connection) -> None:
            cursor = db.execute(
                "UPDATE erasures SET last_step = ? "
                "WHERE erasure_id = ? AND status = 'running'",
                (step, erasure_id),
            )
            if cursor.rowcount:
                return
            row = db.execute(
                "SELECT status FROM erasures WHERE erasure_id = ?", (erasure_id,)
            ).fetchone()
            if row is None:
                raise KeyError(erasure_id)
            raise ValueError("erase journal row is already finished")

        await asyncio.to_thread(lambda: self._locked(update))

    async def finish(
        self,
        erasure_id: str,
        status: ErasureStatus,
        residue: tuple[str, ...],
        counts: tuple[tuple[str, int], ...],
        at: datetime,
    ) -> None:
        """Close one running row with its final outcome."""
        if status is ErasureStatus.RUNNING:
            raise ValueError("erase journal finish requires a terminal status")
        if any(count < 0 for _, count in counts):
            raise ValueError("erase counts must not be negative")

        def update(db: sqlite3.Connection) -> None:
            cursor = db.execute(
                "UPDATE erasures SET status = ?, residue = ?, counts = ?, "
                "finished_at = ? WHERE erasure_id = ? AND status = 'running'",
                (
                    status.value,
                    dumps(residue),
                    dumps(counts),
                    to_micros(at),
                    erasure_id,
                ),
            )
            if cursor.rowcount:
                return
            row = db.execute(
                "SELECT status FROM erasures WHERE erasure_id = ?", (erasure_id,)
            ).fetchone()
            if row is None:
                raise KeyError(erasure_id)
            raise ValueError("erase journal row is already finished")

        await asyncio.to_thread(lambda: self._locked(update))

    async def running(self, tenant: str | None = None) -> list[ErasureRecord]:
        """Return running rows oldest first, optionally filtered by Tenant."""

        def query(db: sqlite3.Connection) -> list[ErasureRecord]:
            if tenant is None:
                rows = db.execute(
                    "SELECT * FROM erasures WHERE status = 'running' "
                    "ORDER BY created_at, seq"
                )
            else:
                rows = db.execute(
                    "SELECT * FROM erasures WHERE status = 'running' "
                    "AND tenant = ? ORDER BY created_at, seq",
                    (tenant,),
                )
            return [erasure_from_row(row) for row in rows]

        return await asyncio.to_thread(lambda: self._locked(query))

    async def get(self, erasure_id: str) -> ErasureRecord | None:
        """Return one erase row or None when its id is unknown."""

        def query(db: sqlite3.Connection) -> ErasureRecord | None:
            row = db.execute(
                "SELECT * FROM erasures WHERE erasure_id = ?", (erasure_id,)
            ).fetchone()
            return None if row is None else erasure_from_row(row)

        return await asyncio.to_thread(lambda: self._locked(query))

    async def close(self) -> None:
        """Close the connection under the same lock as all database calls."""
        await asyncio.to_thread(self._close)

    def _locked(self, operation: Callable[[sqlite3.Connection], T]) -> T:
        """Run one call in a transaction while holding the connection lock."""
        with self._lock, self._db:
            return operation(self._db)

    def _close(self) -> None:
        """Release the connection exactly once while holding the thread lock."""
        with self._lock:
            if not self._closed:
                self._db.close()
                self._closed = True
