"""Shared records for background work in a Tenant."""

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from typing import Any


class JobState(StrEnum):
    """Lifecycle of a background job."""

    QUEUED = "queued"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"

    @property
    def is_terminal(self) -> bool:
        """Return whether the job has finished and will not change again."""
        return self in (JobState.DONE, JobState.FAILED)


@dataclass(frozen=True, slots=True)
class JobProgress:
    """Persisted progress for a running job.

    Attributes:
        stage: Name of the work stage currently in progress.
        done: Number of completed units in the current stage.
        total: Total units in the current stage, when known.
    """

    stage: str
    done: int
    total: int | None = None

    def __post_init__(self) -> None:
        """Reject negative counts and totals below the completed count."""
        if self.done < 0:
            raise ValueError("progress done must not be negative")
        if self.total is not None and self.total < self.done:
            raise ValueError("progress total must be at least done")


@dataclass(frozen=True, slots=True)
class Job:
    """One unit of background work for one Tenant.

    Attributes:
        job_id: Unique identifier of this submission.
        tenant_id: Tenant that owns the work.
        op: Operation whose handler performs the work.
        payload: JSON-compatible operation arguments.
        request_hash: SHA-256 of the canonical operation and payload JSON.
        status: Current lifecycle state.
        attempts: Number of attempts already started.
        created_at: Time the job was submitted, as an aware UTC datetime.
        updated_at: Time of the last state change, as an aware UTC datetime.
        trace_id: Trace identifier shared by the request and its job.
        idempotency_key: Optional request key, unique within the Tenant.
        result: JSON-compatible handler result, or None before success.
        error: Error record with `type` and `message`, or None.
        progress: Latest persisted stage and completed-unit count, if reported.
        replayed: Response-only indication of an idempotent replay; never stored.
    """

    job_id: str
    tenant_id: str
    op: str
    payload: dict[str, Any]
    request_hash: str
    status: JobState
    attempts: int
    created_at: datetime
    updated_at: datetime
    trace_id: str = ""
    idempotency_key: str | None = None
    result: dict[str, Any] | None = None
    error: dict[str, str] | None = None
    progress: JobProgress | None = None
    replayed: bool = False
