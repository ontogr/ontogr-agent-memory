"""Immutable preview and outcome values for forget operations."""

from dataclasses import dataclass
from datetime import datetime

from ontogr_mem.service.contracts import ForgetMode


@dataclass(frozen=True, slots=True)
class RemovedCounts:
    """Counts of records and derived documents removed or updated.

    Attributes:
        episodes: Episode records removed.
        facts: Facts changed by the operation.
        observations: Observation records removed.
        entities: Entity records removed.
        vector_documents: Vector documents removed or rewritten.
        cache_entries: Cache entries evicted.
        entity_text_updates: Surviving entity text records blanked.
    """

    episodes: int = 0
    facts: int = 0
    observations: int = 0
    entities: int = 0
    vector_documents: int = 0
    cache_entries: int = 0
    entity_text_updates: int = 0


@dataclass(frozen=True, slots=True)
class ForgetPreview:
    """A read-only forget plan and its short-lived confirmation token.

    Attributes:
        mode: Removal mode selected for the operation.
        selector_kind: Field used to resolve the target.
        episode_ids: Existing Episodes selected by the request.
        fact_ids: Facts with evidence in the selected Episodes.
        entity_ids: Entities considered removable by the plan.
        collateral_fact_ids: Other current Facts in touched slots.
        reopened_fact_ids: Older Facts the operation may reopen.
        confirm_token: Token bound to this caller, target, tenant, and mode.
        expires_at: UTC time after which the token cannot be used.
    """

    mode: ForgetMode
    selector_kind: str
    episode_ids: tuple[str, ...]
    fact_ids: tuple[str, ...]
    entity_ids: tuple[str, ...]
    collateral_fact_ids: tuple[str, ...]
    reopened_fact_ids: tuple[str, ...]
    confirm_token: str
    expires_at: datetime


@dataclass(frozen=True, slots=True)
class ClearTenantPreview:
    """Tenant-wide cleanup summary and the confirmation token it requires."""

    mode: str
    episode_ids: tuple[str, ...]
    fact_ids: tuple[str, ...]
    entity_ids: tuple[str, ...]
    job_count: int
    confirm_token: str
    expires_at: datetime


@dataclass(frozen=True, slots=True)
class ForgetResult:
    """Outcome of a forget or erase run.

    Attributes:
        erasure_id: Journal id for the operation.
        status: Final journal status.
        removed: Counts of removed or updated records.
        residue: Residue names found by the erase scan.
        not_cleaned: Known storage outside the operation's cleanup boundary.
        text_scanned: Whether a text residue scan ran.
        blanked_entity_ids: Surviving entities whose text was blanked.
    """

    erasure_id: str
    status: str
    removed: RemovedCounts = RemovedCounts()
    residue: tuple[str, ...] = ()
    not_cleaned: tuple[str, ...] = ()
    text_scanned: bool = False
    blanked_entity_ids: tuple[str, ...] = ()
