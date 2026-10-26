"""Async persistence contracts for service state."""

from collections.abc import Sequence
from typing import Protocol

from ontogr_mem.data_models.jobs import Job, JobProgress, JobState
from ontogr_mem.relational_store.jobs.erase_journal import EraseJournal


class JobStore(Protocol):
    """Persist jobs from all Tenants without executing their handlers."""

    async def create_job(self, job: Job) -> tuple[Job, bool]:
        """Atomically insert a job or return the stored job and a replay flag.

        A key is unique within its Tenant. An existing key with the same
        request hash returns the existing job and True. A different hash raises
        IdempotencyConflictError. Without a key, each job is inserted separately.
        Stored records and returned records do not share mutable payloads.
        The response-only `replayed` field is always False in stored records.
        """
        ...

    async def get_job(self, job_id: str) -> Job | None:
        """Return a detached record, or None for an unknown job id."""
        ...

    async def save_job(self, job: Job) -> None:
        """Replace an existing job record without changing its insertion order.

        The response-only replay flag is discarded. An unknown job id raises
        KeyError rather than inserting a new job.
        """
        ...

    async def set_job_progress(self, job_id: str, progress: JobProgress) -> None:
        """Persist progress without allowing its completed count to regress.

        An unknown job id raises KeyError. Progress totals must be at least the
        completed count; a total may be omitted when it is not known.
        """
        ...

    async def requeue_running(self) -> list[Job]:
        """Requeue running jobs and return all queued jobs oldest first.

        Creation time orders the result, with insertion order breaking ties.
        Attempts, timestamps, results and errors remain unchanged.
        """
        ...

    async def count_jobs(
        self, tenant_ids: Sequence[str], statuses: Sequence[JobState]
    ) -> int:
        """Count jobs matching both filters; either empty filter returns zero."""
        ...

    async def tenant_ids(self) -> list[str]:
        """Return sorted, distinct Tenants that own at least one job."""
        ...

    async def delete_tenant(self, tenant_id: str) -> int:
        """Delete a Tenant's jobs and idempotency keys; return removed jobs."""
        ...


class StateStore(JobStore, EraseJournal, Protocol):
    """Combine job and erase journal persistence with resource release."""

    async def close(self) -> None:
        """Release store resources; repeated closure is safe."""
        ...
