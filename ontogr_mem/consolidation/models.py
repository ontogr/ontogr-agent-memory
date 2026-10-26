"""Models stored by the consolidation jobs, and the results the jobs return."""

from datetime import datetime
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from ontogr_mem.data_models.extraction import Usage


ItemKind = Literal["entity", "fact"]


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Watermark(_Model):
    """How far a stage has processed. Advanced only after the stage fully succeeds."""

    key: str
    value: str
    updated_at: datetime


class FeedbackEvent(_Model):
    """One rating (0 to 1) of a memory item. ``implicit`` counts at half strength."""

    event_id: str
    target_id: str
    rating: float = Field(ge=0.0, le=1.0)
    implicit: bool = False
    created_at: datetime
    applied: bool = False
    attempts: int = 0


class FeedbackWeight(_Model):
    """Streaming feedback weight of an item; 0.5 is neutral."""

    target_id: str
    weight: float = 0.5
    applied_ids: list[str] = Field(default_factory=list)
    last_update_at: datetime


class AccessEvent(_Model):
    """A read or mention of an item, used for recency and frequency."""

    event_id: str
    target_id: str
    at: datetime


class SessionEntry(_Model):
    """One entry of an agent session that reflection may learn from."""

    entry_id: str
    session_id: str
    seq: int
    text: str
    helpfulness: int = 0
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    created_at: datetime


class ProposalStatus(StrEnum):
    """Life of a reflection proposal."""

    PENDING = "pending"
    ACCEPTED = "accepted"
    REJECTED = "rejected"


class Proposal(_Model):
    """A lesson that reflection proposes to add. Applied only when auto-accepted."""

    proposal_id: str
    session_id: str
    text: str
    source_entry_ids: list[str]
    status: ProposalStatus = ProposalStatus.PENDING
    reason: str = ""
    created_at: datetime


class Score(_Model):
    """Importance of one item and the three terms it is made of."""

    target_id: str
    kind: ItemKind
    created_at: datetime
    score: float
    recency: float
    frequency: float
    salience: float
    computed_at: datetime


class ArchiveState(StrEnum):
    """Whether an archived item is still restorable."""

    ARCHIVED = "archived"
    RESTORED = "restored"


class ArchiveRecord(_Model):
    """An item moved to the cold state. The source item is kept until retention ends."""

    target_id: str
    target_kind: ItemKind
    archived_at: datetime
    reason: str
    state: ArchiveState = ArchiveState.ARCHIVED


class Community(_Model):
    """A group of entities that are densely connected, with a summary."""

    community_id: str
    member_ids: list[str]
    summary: str = ""
    description: str = ""
    content_hash: str = ""


class Bucket(_Model):
    """A node of the summary hierarchy above the communities."""

    bucket_id: str
    level: int
    child_ids: list[str]
    summary: str = ""
    content_hash: str = ""


class LessonVerdict(StrEnum):
    """The writer's decision on one proposed lesson."""

    ACCEPT = "accept"
    ALREADY_KNOWN = "already_known"
    NOT_DURABLE = "not_durable"
    UNSUPPORTED = "unsupported"


class StageStatus(StrEnum):
    """Outcome of one stage in one run."""

    OK = "ok"
    SKIPPED = "skipped"
    PARTIAL = "partial"
    DRY_RUN = "dry_run"
    FAILED = "failed"


class StageResult(_Model):
    """What a stage did. ``watermark`` is stored only when ``status`` is OK."""

    stage: str
    status: StageStatus
    counts: dict[str, int] = Field(default_factory=dict)
    notes: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    usage: Usage = Field(default_factory=Usage)
    watermark: str | None = None


class RunStatus(StrEnum):
    """Outcome of a whole run."""

    OK = "ok"
    PARTIAL = "partial"
    FAILED = "failed"
    LOCKED = "locked"
    LEASE_LOST = "lease_lost"


class RunResult(_Model):
    """Per-stage results of one runner call.

    ``stages`` holds the results of every pass, in pass order, so a failure of
    an earlier pass stays visible. ``reruns`` says how many passes followed the
    first one.
    """

    run_id: str
    status: RunStatus
    stages: list[StageResult] = Field(default_factory=list)
    reruns: int = 0
    # True when a rerun was asked for but the cap was reached. The request is kept
    # and the next run makes one extra pass.
    rerun_pending: bool = False

    def stage(self, name: str) -> StageResult:
        """Return the result of stage ``name`` from the last pass.

        Raises:
            KeyError: If the run has no result for ``name``.
        """
        for result in reversed(self.stages):
            if result.stage == name:
                return result
        raise KeyError(name)
