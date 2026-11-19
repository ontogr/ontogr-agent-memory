"""PostgreSQL job persistence over a shared connection pool."""

from collections.abc import AsyncGenerator, Callable, Sequence
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from ontogr_mem.data_models.clock import Clock
from ontogr_mem.data_models.jobs import Job, JobProgress, JobState
from ontogr_mem.data_models.tenant import validate_tenant_id
from ontogr_mem.data_models.timestamps import to_micros
from ontogr_mem.errors import StoreError
from ontogr_mem.relational_store.engine import pg_connection
from ontogr_mem.relational_store.jobs.erase_journal import (
    ErasureRecord,
    ErasureStatus,
)
from ontogr_mem.relational_store.jobs.errors import (
    IdempotencyConflictError,
    LeaseLostError,
)
from ontogr_mem.relational_store.jobs.rows import (
    JOB_COLUMNS,
    dumps,
    erasure_from_row,
    job_from_row,
    job_values,
)


if TYPE_CHECKING:
    import asyncpg


_SELECT_JOB = f"SELECT seq, {JOB_COLUMNS} FROM jobs"
_INSERT_JOB = (
    f"INSERT INTO jobs ({JOB_COLUMNS}) VALUES "
    "($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16) "
    "ON CONFLICT (tenant_id, idempotency_key) DO NOTHING "
    f"RETURNING seq, {JOB_COLUMNS}"
)


class PostgresJobStore:
    """Keep durable jobs in one Postgres database behind a shared pool.

    Jobs are leased to one process at a time and claimed in tenant order.

    Args:
        pool: Process-wide connection pool; the caller runs
            ``ensure_postgres_schema`` before constructing the store.
        owner_id: Identity recorded on rows leased by this process.
        lease_seconds: Duration of each lease.
        clock: Service clock used for lease checks.
    """

    def __init__(
        self, pool: "asyncpg.Pool", clock: Clock, *, owner_id: str, lease_seconds: float
    ) -> None:
        """Bind the store to its pool without taking a connection."""
        self._pool = pool
        self._clock = clock
        self._owner_id = owner_id
        self._lease_micros = int(lease_seconds * 1_000_000)

    async def create_job(self, job: Job) -> tuple[Job, bool]:
        """Atomically insert a job or replay a matching Tenant request key."""
        async with pg_connection(self._pool) as connection, connection.transaction():
            await connection.fetchval(
                "SELECT pg_advisory_xact_lock(hashtextextended($1, 0))",
                job.tenant_id,
            )
            row = await connection.fetchrow(_INSERT_JOB, *job_values(job))
            if row is not None:
                return job_from_row(row), False
            if job.idempotency_key is None:
                raise StoreError("job insert reported no conflict and no row")
            existing = await connection.fetchrow(
                f"{_SELECT_JOB} WHERE tenant_id = $1 AND idempotency_key = $2",
                job.tenant_id,
                job.idempotency_key,
            )
            if existing is None:
                raise StoreError("conflicting job row disappeared on read")
            stored = job_from_row(existing)
            if stored.request_hash != job.request_hash:
                raise IdempotencyConflictError(
                    "idempotency key has a different request"
                )
            return stored, True

    async def get_job(self, job_id: str) -> Job | None:
        """Return a detached job or None for an unknown id."""
        async with pg_connection(self._pool) as connection:
            row = await connection.fetchrow(f"{_SELECT_JOB} WHERE job_id = $1", job_id)
            return None if row is None else job_from_row(row)

    async def save_job(self, job: Job) -> bool:
        """Replace an existing record without storing its replay flag."""
        async with pg_connection(self._pool) as connection:
            owner_filter = (
                "(owner_id = $14 AND status = 'running' AND lease_expires_at > $15)"
            )
            if job.status is JobState.QUEUED:
                owner_filter += " OR (owner_id IS NULL AND status = 'queued')"
            status = await connection.execute(
                "UPDATE jobs SET tenant_id = $1, op = $2, payload = $3, "
                "request_hash = $4, idempotency_key = $5, status = $6, "
                "attempts = $7, result = $8, error = $9, created_at = $10, "
                "updated_at = $11, trace_id = $12, "
                "owner_id = CASE WHEN $6 = 'queued' THEN NULL ELSE owner_id END, "
                "lease_expires_at = CASE WHEN $6 = 'queued' THEN NULL "
                "ELSE lease_expires_at END WHERE job_id = $13 "
                f"AND ({owner_filter})",
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
                job.job_id,
                self._owner_id,
                to_micros(self._clock.now()),
            )
            return status != "UPDATE 0"

    async def set_job_progress(self, job_id: str, progress: JobProgress) -> None:
        """Persist monotonic progress inside one serialized transaction."""
        async with pg_connection(self._pool) as connection, connection.transaction():
            row = await connection.fetchrow(
                "SELECT progress_done FROM jobs WHERE job_id = $1 "
                "AND owner_id = $2 AND status = 'running' "
                "AND lease_expires_at > $3 FOR UPDATE",
                job_id,
                self._owner_id,
                to_micros(self._clock.now()),
            )
            if row is None:
                raise LeaseLostError(job_id)
            previous_done = row["progress_done"]
            if previous_done is not None and progress.done < previous_done:
                raise ValueError("job progress cannot move backwards")
            await connection.execute(
                "UPDATE jobs SET progress_stage = $1, progress_done = $2, "
                "progress_total = $3 WHERE job_id = $4 AND owner_id = $5 "
                "AND status = 'running' AND lease_expires_at > $6",
                progress.stage,
                progress.done,
                progress.total,
                job_id,
                self._owner_id,
                to_micros(self._clock.now()),
            )

    async def queued_jobs(self) -> list[Job]:
        """Return queued jobs oldest first."""
        async with pg_connection(self._pool) as connection:
            rows = await connection.fetch(
                f"{_SELECT_JOB} WHERE status = 'queued' ORDER BY created_at, seq"
            )
            return [job_from_row(row) for row in rows]

    async def requeue_expired(self, now: datetime) -> None:
        """Requeue running jobs whose lease is missing or expired."""
        async with pg_connection(self._pool) as connection:
            await connection.execute(
                "UPDATE jobs SET status = 'queued', owner_id = NULL, "
                "lease_expires_at = NULL WHERE status = 'running' AND "
                "(lease_expires_at IS NULL OR lease_expires_at <= $1)",
                to_micros(now),
            )

    async def claim_next(
        self, tenant_id: str, now: datetime, can_run: Callable[[Job], bool]
    ) -> Job | None:
        """Claim the oldest queued job when this tenant has no live lease."""
        async with pg_connection(self._pool) as connection, connection.transaction():
            await connection.fetchval(
                "SELECT pg_advisory_xact_lock(hashtextextended($1, 0))", tenant_id
            )
            has_live_lease = await connection.fetchval(
                "SELECT EXISTS (SELECT 1 FROM jobs WHERE tenant_id = $1 "
                "AND status = 'running' AND lease_expires_at > $2) OR "
                "EXISTS (SELECT 1 FROM erasures WHERE tenant = $1 "
                "AND status = 'running' AND lease_expires_at > $2)",
                tenant_id,
                to_micros(now),
            )
            if has_live_lease:
                return None
            rows = await connection.fetch(
                "SELECT seq FROM jobs WHERE tenant_id = $1 AND status = 'queued' "
                "ORDER BY created_at, seq FOR UPDATE",
                tenant_id,
            )
            for row in rows:
                job_row = await connection.fetchrow(
                    f"{_SELECT_JOB} WHERE seq = $1", row["seq"]
                )
                job = job_from_row(job_row)
                if not can_run(job):
                    continue
                expiry = to_micros(now) + self._lease_micros
                updated = await connection.fetchrow(
                    "UPDATE jobs SET status = 'running', owner_id = $1, "
                    "lease_expires_at = $2, heartbeat_at = $3 WHERE seq = $4 "
                    f"RETURNING seq, {JOB_COLUMNS}",
                    self._owner_id,
                    expiry,
                    to_micros(now),
                    row["seq"],
                )
                return job_from_row(updated)
            return None

    async def tenant_has_live_lease(self, tenant_id: str, now: datetime) -> bool:
        """Return whether a job or erase row has a live lease for the tenant."""
        async with pg_connection(self._pool) as connection:
            return bool(
                await connection.fetchval(
                    "SELECT EXISTS (SELECT 1 FROM jobs WHERE tenant_id = $1 "
                    "AND status = 'running' AND lease_expires_at > $2) OR "
                    "EXISTS (SELECT 1 FROM erasures WHERE tenant = $1 "
                    "AND status = 'running' AND lease_expires_at > $2)",
                    tenant_id,
                    to_micros(now),
                )
            )

    async def live_lease_expires_at(
        self, tenant_id: str, now: datetime
    ) -> datetime | None:
        """Return the latest live running job lease for a tenant."""
        async with pg_connection(self._pool) as connection:
            value = await connection.fetchval(
                "SELECT max(lease_expires_at) FROM jobs WHERE tenant_id = $1 "
                "AND status = 'running' AND lease_expires_at > $2",
                tenant_id,
                to_micros(now),
            )
            return (
                None
                if value is None
                else datetime.fromtimestamp(value / 1_000_000, UTC)
            )

    @asynccontextmanager
    async def tenant_lock(self, tenant_id: str) -> AsyncGenerator[None, None]:
        """Hold the session advisory lock for one tenant."""
        async with pg_connection(self._pool) as connection:
            await connection.fetchval(
                "SELECT pg_advisory_lock(hashtextextended($1, 0))", tenant_id
            )
            try:
                yield
            finally:
                await connection.fetchval(
                    "SELECT pg_advisory_unlock(hashtextextended($1, 0))", tenant_id
                )

    async def renew_leases(self, now: datetime) -> None:
        """Renew this owner's live job and erasure leases."""
        async with pg_connection(self._pool) as connection:
            await connection.execute(
                "UPDATE jobs SET lease_expires_at = $1, heartbeat_at = $2 "
                "WHERE status = 'running' AND owner_id = $3 "
                "AND lease_expires_at > $2",
                to_micros(now) + self._lease_micros,
                to_micros(now),
                self._owner_id,
            )
            await connection.execute(
                "UPDATE erasures SET lease_expires_at = $1, heartbeat_at = $2 "
                "WHERE status = 'running' AND owner_id = $3 "
                "AND lease_expires_at > $2",
                to_micros(now) + self._lease_micros,
                to_micros(now),
                self._owner_id,
            )

    async def release_leases(self) -> None:
        """Release every job and erasure lease held by this owner."""
        async with pg_connection(self._pool) as connection:
            await connection.execute(
                "UPDATE jobs SET owner_id = NULL, lease_expires_at = NULL "
                "WHERE owner_id = $1",
                self._owner_id,
            )
            await connection.execute(
                "UPDATE erasures SET owner_id = NULL, lease_expires_at = NULL "
                "WHERE owner_id = $1",
                self._owner_id,
            )

    async def release_job(self, job_id: str) -> None:
        """Release one job lease held by this owner."""
        async with pg_connection(self._pool) as connection:
            await connection.execute(
                "UPDATE jobs SET owner_id = NULL, lease_expires_at = NULL "
                "WHERE owner_id = $1 AND job_id = $2",
                self._owner_id,
                job_id,
            )

    async def count_jobs(
        self, tenant_ids: Sequence[str], statuses: Sequence[JobState]
    ) -> int:
        """Count only jobs matching both selected Tenants and states."""
        tenants = tuple(set(tenant_ids))
        states = tuple({status.value for status in statuses})
        if not tenants or not states:
            return 0
        tenant_slots = ", ".join(f"${index}" for index in range(1, len(tenants) + 1))
        state_slots = ", ".join(
            f"${index}"
            for index in range(len(tenants) + 1, len(tenants) + len(states) + 1)
        )
        async with pg_connection(self._pool) as connection:
            return await connection.fetchval(
                f"SELECT COUNT(*) FROM jobs WHERE tenant_id IN ({tenant_slots}) "
                f"AND status IN ({state_slots})",
                *tenants,
                *states,
            )

    async def tenant_ids(self) -> list[str]:
        """Return sorted distinct Tenants with persisted jobs."""
        async with pg_connection(self._pool) as connection:
            rows = await connection.fetch(
                "SELECT DISTINCT tenant_id FROM jobs ORDER BY tenant_id"
            )
            return [row["tenant_id"] for row in rows]

    async def delete_tenant(self, tenant_id: str) -> int:
        """Delete a Tenant's jobs and their idempotency keys."""
        validate_tenant_id(tenant_id)
        async with pg_connection(self._pool) as connection:
            status = await connection.execute(
                "DELETE FROM jobs WHERE tenant_id = $1", tenant_id
            )
            return int(status.split()[-1])

    async def create(self, record: ErasureRecord) -> str:
        """Store a running erase row and return its id."""
        if record.status is not ErasureStatus.RUNNING:
            raise ValueError("new erase journal rows must be running")
        async with pg_connection(self._pool) as connection:
            await connection.execute(
                "INSERT INTO erasures (erasure_id, tenant, mode, principal_id, "
                "selector_kind, target, episode_ids, fact_ids, observation_ids, "
                "entity_ids, created_at, status, last_step, finished_at, residue, "
                "counts, owner_id, lease_expires_at, heartbeat_at) "
                "VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, "
                "$12, $13, NULL, $14, $15, $16, $17, $18)",
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
                self._owner_id,
                to_micros(record.created_at) + self._lease_micros,
                to_micros(record.created_at),
            )
            return record.erasure_id

    async def step(self, erasure_id: str, step: str) -> None:
        """Record the last completed step of a running erase row."""
        if not step:
            raise ValueError("erase step must not be empty")
        async with pg_connection(self._pool) as connection:
            await self._update_running(
                connection, erasure_id, "step", step, to_micros(self._clock.now())
            )

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
        async with pg_connection(self._pool) as connection:
            await self._update_running(
                connection,
                erasure_id,
                "finish",
                status.value,
                dumps(residue),
                dumps(counts),
                to_micros(at),
                to_micros(self._clock.now()),
            )

    async def running(self, tenant: str | None = None) -> list[ErasureRecord]:
        """Return running rows oldest first, optionally filtered by Tenant."""
        query = (
            "SELECT seq, erasure_id, tenant, mode, principal_id, selector_kind, "
            "target, episode_ids, fact_ids, observation_ids, entity_ids, "
            "created_at, status, last_step, finished_at, residue, counts "
            "FROM erasures WHERE status = 'running'"
        )
        async with pg_connection(self._pool) as connection:
            if tenant is None:
                rows = await connection.fetch(query + " ORDER BY created_at, seq")
            else:
                rows = await connection.fetch(
                    query + " AND tenant = $1 ORDER BY created_at, seq", tenant
                )
            return [erasure_from_row(row) for row in rows]

    async def claim_recoverable(
        self, tenant: str | None, now: datetime
    ) -> list[ErasureRecord]:
        """Claim expired running erase rows, optionally filtered by Tenant."""
        lease_expiry = to_micros(now) + self._lease_micros
        async with pg_connection(self._pool) as connection:
            rows = await connection.fetch(
                "WITH candidates AS (SELECT seq FROM erasures "
                "WHERE status = 'running' AND "
                "(lease_expires_at IS NULL OR lease_expires_at <= $1) AND "
                "(owner_id IS NULL OR lease_expires_at <= $1) AND "
                "($4::text IS NULL OR tenant = $4) "
                "ORDER BY created_at, seq FOR UPDATE SKIP LOCKED) "
                "UPDATE erasures SET owner_id = $2, lease_expires_at = $3, "
                "heartbeat_at = $1 WHERE seq IN (SELECT seq FROM candidates) "
                "RETURNING seq, erasure_id, tenant, mode, principal_id, "
                "selector_kind, target, episode_ids, fact_ids, observation_ids, "
                "entity_ids, created_at, status, last_step, finished_at, "
                "residue, counts",
                to_micros(now),
                self._owner_id,
                lease_expiry,
                tenant,
            )
            return [erasure_from_row(row) for row in rows]

    async def get(self, erasure_id: str) -> ErasureRecord | None:
        """Return one erase row or None when its id is unknown."""
        async with pg_connection(self._pool) as connection:
            row = await connection.fetchrow(
                "SELECT seq, erasure_id, tenant, mode, principal_id, selector_kind, "
                "target, episode_ids, fact_ids, observation_ids, entity_ids, "
                "created_at, status, last_step, finished_at, residue, counts "
                "FROM erasures WHERE erasure_id = $1",
                erasure_id,
            )
            return None if row is None else erasure_from_row(row)

    async def close(self) -> None:
        """Keep the shared pool open; it lives for the process lifetime."""

    async def _update_running(
        self,
        connection: "asyncpg.Connection",
        erasure_id: str,
        kind: str,
        *values: object,
    ) -> None:
        """Apply one step or finish update and resolve its missing-row case."""
        if kind == "step":
            status = await connection.execute(
                "UPDATE erasures SET last_step = $1 "
                "WHERE erasure_id = $2 AND status = 'running' "
                "AND owner_id = $3 AND lease_expires_at > "
                "$4",
                values[0],
                erasure_id,
                self._owner_id,
                values[1],
            )
        else:
            status = await connection.execute(
                "UPDATE erasures SET status = $1, residue = $2, counts = $3, "
                "finished_at = $4 WHERE erasure_id = $5 AND status = 'running' "
                "AND owner_id = $6 AND lease_expires_at > "
                "$7",
                values[0],
                values[1],
                values[2],
                values[3],
                erasure_id,
                self._owner_id,
                values[4],
            )
        if status != "UPDATE 0":
            return
        current_status = await connection.fetchval(
            "SELECT status FROM erasures WHERE erasure_id = $1", erasure_id
        )
        if current_status is not None and current_status != ErasureStatus.RUNNING.value:
            raise ValueError("erase journal row is already finished")
        raise LeaseLostError(erasure_id)
