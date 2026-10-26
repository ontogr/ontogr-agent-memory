"""In-memory job persistence with no durable storage."""

from collections.abc import Sequence
from copy import deepcopy
from dataclasses import replace
from datetime import datetime

from ontogr_mem.data_models.jobs import Job, JobProgress, JobState
from ontogr_mem.data_models.tenant import validate_tenant_id
from ontogr_mem.relational_store.jobs.erase_journal import (
    ErasureRecord,
    ErasureStatus,
)
from ontogr_mem.relational_store.jobs.errors import IdempotencyConflictError


class MemoryJobStore:
    """Keep detached job records for one event loop without durable storage."""

    def __init__(self) -> None:
        """Create an empty store."""
        self._jobs: dict[str, Job] = {}
        self._keys: dict[tuple[str, str], str] = {}
        self._erasures: dict[str, ErasureRecord] = {}

    async def create_job(self, job: Job) -> tuple[Job, bool]:
        """Insert a job or atomically replay a matching Tenant request key."""
        if job.idempotency_key is not None:
            key = (job.tenant_id, job.idempotency_key)
            if key in self._keys:
                stored = self._jobs[self._keys[key]]
                if stored.request_hash != job.request_hash:
                    raise IdempotencyConflictError(
                        "idempotency key has a different request"
                    )
                return deepcopy(stored), True
        if job.job_id in self._jobs:
            raise ValueError("job id already exists")
        stored = deepcopy(replace(job, replayed=False))
        self._jobs[job.job_id] = stored
        if job.idempotency_key is not None:
            self._keys[(job.tenant_id, job.idempotency_key)] = job.job_id
        return deepcopy(stored), False

    async def get_job(self, job_id: str) -> Job | None:
        """Return a detached job or None when no record exists."""
        return deepcopy(self._jobs.get(job_id))

    async def save_job(self, job: Job) -> None:
        """Replace an existing record, discarding response-only replay state."""
        previous = self._jobs[job.job_id]
        if job.idempotency_key is not None:
            key = (job.tenant_id, job.idempotency_key)
            owner = self._keys.get(key)
            if owner is not None and owner != job.job_id:
                raise IdempotencyConflictError("idempotency key belongs to another job")
        stored = deepcopy(
            replace(
                job,
                replayed=False,
                progress=job.progress
                if job.progress is not None
                else previous.progress,
            )
        )
        if previous.idempotency_key is not None:
            del self._keys[(previous.tenant_id, previous.idempotency_key)]
        self._jobs[job.job_id] = stored
        if job.idempotency_key is not None:
            self._keys[(job.tenant_id, job.idempotency_key)] = job.job_id

    async def set_job_progress(self, job_id: str, progress: JobProgress) -> None:
        """Persist monotonic progress for a known job."""
        job = self._jobs.get(job_id)
        if job is None:
            raise KeyError(job_id)
        if job.progress is not None and progress.done < job.progress.done:
            raise ValueError("job progress cannot move backwards")
        self._jobs[job_id] = replace(job, progress=progress)

    async def requeue_running(self) -> list[Job]:
        """Requeue interrupted jobs and return all queued work oldest first."""
        for job_id, job in self._jobs.items():
            if job.status == JobState.RUNNING:
                self._jobs[job_id] = replace(job, status=JobState.QUEUED)
        return deepcopy(
            sorted(
                (job for job in self._jobs.values() if job.status == JobState.QUEUED),
                key=lambda job: job.created_at,
            )
        )

    async def count_jobs(
        self, tenant_ids: Sequence[str], statuses: Sequence[JobState]
    ) -> int:
        """Count only jobs belonging to the selected Tenants and states."""
        tenants, states = set(tenant_ids), set(statuses)
        return sum(
            job.tenant_id in tenants and job.status in states
            for job in self._jobs.values()
        )

    async def tenant_ids(self) -> list[str]:
        """Return sorted distinct Tenants with stored jobs."""
        return sorted({job.tenant_id for job in self._jobs.values()})

    async def delete_tenant(self, tenant_id: str) -> int:
        """Delete a Tenant's jobs and their idempotency keys."""
        validate_tenant_id(tenant_id)
        job_ids = {
            job_id for job_id, job in self._jobs.items() if job.tenant_id == tenant_id
        }
        for job_id in job_ids:
            del self._jobs[job_id]
        self._keys = {
            key: job_id for key, job_id in self._keys.items() if key[0] != tenant_id
        }
        return len(job_ids)

    async def create(self, record: ErasureRecord) -> str:
        """Store a running erase row and return its id."""
        if record.erasure_id in self._erasures:
            raise ValueError("erase journal id already exists")
        if record.status is not ErasureStatus.RUNNING:
            raise ValueError("new erase journal rows must be running")
        self._erasures[record.erasure_id] = record
        return record.erasure_id

    async def step(self, erasure_id: str, step: str) -> None:
        """Record the last completed step of a running erase row."""
        record = self._erasures.get(erasure_id)
        if record is None:
            raise KeyError(erasure_id)
        if record.status is not ErasureStatus.RUNNING:
            raise ValueError("erase journal row is already finished")
        if not step:
            raise ValueError("erase step must not be empty")
        self._erasures[erasure_id] = replace(record, last_step=step)

    async def finish(
        self,
        erasure_id: str,
        status: ErasureStatus,
        residue: tuple[str, ...],
        counts: tuple[tuple[str, int], ...],
        at: datetime,
    ) -> None:
        """Close one running row with its final outcome."""
        record = self._erasures.get(erasure_id)
        if record is None:
            raise KeyError(erasure_id)
        if record.status is not ErasureStatus.RUNNING:
            raise ValueError("erase journal row is already finished")
        if status is ErasureStatus.RUNNING:
            raise ValueError("erase journal finish requires a terminal status")
        self._erasures[erasure_id] = replace(
            record,
            status=status,
            residue=residue,
            counts=counts,
            finished_at=at,
        )

    async def running(self, tenant: str | None = None) -> list[ErasureRecord]:
        """Return running rows oldest first, optionally filtered by Tenant."""
        return sorted(
            (
                record
                for record in self._erasures.values()
                if record.status is ErasureStatus.RUNNING
                and (tenant is None or record.tenant == tenant)
            ),
            key=lambda record: (record.created_at, record.erasure_id),
        )

    async def get(self, erasure_id: str) -> ErasureRecord | None:
        """Return one erase row or None when its id is unknown."""
        return self._erasures.get(erasure_id)

    async def close(self) -> None:
        """Do nothing because the memory store owns no external resources."""
