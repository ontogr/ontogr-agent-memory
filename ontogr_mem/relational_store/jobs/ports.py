"""Async persistence contracts for service state."""

from collections.abc import Callable, Sequence
from contextlib import AbstractAsyncContextManager
from datetime import datetime
from typing import Protocol

from ontogr_mem.data_models.jobs import Job, JobProgress, JobState
from ontogr_mem.relational_store.jobs.erase_journal import EraseJournal, ErasureRecord


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

    async def requeue_expired(self, now: datetime) -> None:
        """Requeue expired running jobs; recovery is the sweep's job."""
        ...

    async def queued_jobs(self) -> list[Job]:
        """Return queued jobs oldest first."""
        ...

    async def claim_next(
        self, tenant_id: str, now: datetime, can_run: Callable[[Job], bool]
    ) -> Job | None:
        """Claim the oldest runnable job for one Tenant."""
        ...

    async def tenant_has_live_lease(self, tenant_id: str, now: datetime) -> bool:
        """Return whether a running job or erase row has a live lease."""
        ...

    async def live_lease_expires_at(
        self, tenant_id: str, now: datetime
    ) -> datetime | None:
        """Return the latest live job lease expiry for one Tenant."""
        ...

    def tenant_lock(self, tenant_id: str) -> AbstractAsyncContextManager[None]:
        """Return a context manager that serializes work for one Tenant."""
        ...

    async def renew_leases(self, now: datetime) -> None:
        """Renew this store owner's running leases."""
        ...

    async def release_leases(self) -> None:
        """Release this store owner's leases."""
        ...

    async def release_job(self, job_id: str) -> None:
        """Release one job lease."""
        ...

    async def claim_recoverable(
        self, tenant: str | None, now: datetime
    ) -> list[ErasureRecord]:
        """Return expired running erase rows for recovery."""
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
