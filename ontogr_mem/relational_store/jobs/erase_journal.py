"""Persistent state for forget operations that may need crash recovery."""

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from typing import Protocol

from ontogr_mem.data_models.tenant import validate_tenant_id


class ErasureStatus(StrEnum):
    """Lifecycle state of a forget journal row."""

    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class ErasureRecord:
    """Recovery data for one forget operation, without source text.

    Attributes:
        erasure_id: Unique id of the forget operation.
        tenant: Tenant that owns the records being removed.
        mode: Forget mode, including the admin-only ``clear`` mode.
        principal_id: Caller that started the operation.
        selector_kind: Selector used to resolve the target.
        target: Digest of the resolved target set.
        episode_ids: Episodes selected by the plan.
        fact_ids: Facts affected by the plan.
        observation_ids: Observations removed by the plan.
        entity_ids: Candidate entities considered by the plan.
        created_at: Time the journal row was created.
        status: Current lifecycle state.
        last_step: Most recent completed step name.
        finished_at: Time the run reached a terminal state.
        residue: Residue names found by the final scan.
        counts: Sorted pairs of result-count names and values.
    """

    erasure_id: str
    tenant: str
    mode: str
    principal_id: str
    selector_kind: str
    target: str
    episode_ids: tuple[str, ...]
    fact_ids: tuple[str, ...]
    observation_ids: tuple[str, ...]
    entity_ids: tuple[str, ...]
    created_at: datetime
    status: ErasureStatus = ErasureStatus.RUNNING
    last_step: str = "tombstone"
    finished_at: datetime | None = None
    residue: tuple[str, ...] = ()
    counts: tuple[tuple[str, int], ...] = ()

    def __post_init__(self) -> None:
        """Validate identity and count invariants at the persistence boundary."""
        validate_tenant_id(self.tenant)
        if not all(
            (
                self.erasure_id,
                self.mode,
                self.principal_id,
                self.selector_kind,
                self.target,
                self.last_step,
            )
        ):
            raise ValueError("erase journal identifiers must not be empty")
        if any(count < 0 for _, count in self.counts):
            raise ValueError("erase counts must not be negative")


class EraseJournal(Protocol):
    """Persist and recover forget-operation tombstones."""

    async def create(self, record: ErasureRecord) -> str:
        """Store a new running row and return its id."""
        ...

    async def step(self, erasure_id: str, step: str) -> None:
        """Record the most recent completed step of a running operation."""
        ...

    async def finish(
        self,
        erasure_id: str,
        status: ErasureStatus,
        residue: tuple[str, ...],
        counts: tuple[tuple[str, int], ...],
        at: datetime,
    ) -> None:
        """Close a running row with its residue, counts and end time."""
        ...

    async def running(self, tenant: str | None = None) -> list[ErasureRecord]:
        """Return running rows oldest first, optionally filtered by Tenant."""
        ...

    async def get(self, erasure_id: str) -> ErasureRecord | None:
        """Return one journal row, or None when its id is unknown."""
        ...
