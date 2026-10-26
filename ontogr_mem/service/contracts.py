"""The request and result models every verb shares."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime
from enum import StrEnum
from typing import Annotated, Any, Literal, TypeVar

from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)

from ontogr_mem.data_models.jobs import JobProgress, JobState
from ontogr_mem.data_models.tenant import validate_tenant_id
from ontogr_mem.data_models.timestamps import require_utc
from ontogr_mem.service.errors import (
    ErrorCode,
    InvalidArgumentError,
    PayloadTooLargeError,
    ServiceError,
)
from ontogr_mem.service.limits import MAX_CONTENT_BYTES, MAX_RESULTS


ModelT = TypeVar("ModelT", bound=BaseModel)


class ContractModel(BaseModel):
    """Base for every contract model: unknown fields are errors."""

    model_config = ConfigDict(extra="forbid")


TenantId = Annotated[str, AfterValidator(validate_tenant_id)]
"""A tenant id: 1 to 60 letters, digits, dashes or underscores."""

UtcDatetime = Annotated[datetime, AfterValidator(require_utc)]
"""A timezone-aware datetime; one without a timezone fails validation."""


def _exactly_one(**named: object) -> None:
    """Raise unless exactly one of the named values is set.

    Args:
        **named: Field name and value pairs, at most one of them set.

    Raises:
        ValueError: If the count of set values is not one.
    """
    if sum(value is not None for value in named.values()) != 1:
        msg = f"name exactly one of {' and '.join(named)}"
        raise ValueError(msg)


class MemoryKind(StrEnum):
    """What add_memory was given."""

    MESSAGE = "message"
    TEXT = "text"
    FACT = "fact"


class ForgetMode(StrEnum):
    """How much forget_memory removes.

    INVALIDATE closes the fact now and keeps its history. DELETE removes the
    observations and the stored episode. MEMORY_ONLY removes the derived facts
    and keeps the episode record. ERASE deletes and then scans the graph, the
    indexes and the model cache for residue, failing when it finds any.
    """

    INVALIDATE = "invalidate"
    DELETE = "delete"
    MEMORY_ONLY = "memory_only"
    ERASE = "erase"


class ScopedRequest(ContractModel):
    """A request within one tenant; ``None`` means the principal's tenant."""

    tenant: TenantId | None = None


class WriteRequest(ScopedRequest):
    """A request that queues a job and may wait for it."""

    wait_seconds: float = Field(default=0.0, ge=0.0, le=60.0)


class AddMemory(WriteRequest):
    """Store one observation.

    ``kind`` decides which fields the request needs: free text for message
    and text, one triple for fact. Content is limited to
    ``MAX_CONTENT_BYTES`` UTF-8 bytes.
    """

    kind: MemoryKind = MemoryKind.TEXT
    content: str | None = Field(
        default=None, description="Free text for kind=text|message."
    )
    subject: str | None = Field(default=None, description="kind=fact: subject name.")
    relation: str | None = Field(
        default=None, description="kind=fact: the relation label."
    )
    object: str | None = Field(default=None, description="kind=fact: object name.")
    statement: str | None = Field(
        default=None, description="kind=fact: the fact as a sentence."
    )
    valid_from: UtcDatetime | None = None
    valid_to: UtcDatetime | None = None
    reference_time: UtcDatetime | None = Field(
        default=None, description="When the event happened. Defaults to now."
    )
    single_valued: bool | None = Field(
        default=None,
        description="kind=fact: true when a new object replaces the old.",
    )
    speaker: str | None = None
    source_description: str = ""
    session_id: str | None = None
    thread: str | None = None
    idempotency_key: str | None = Field(default=None, max_length=128)

    @field_validator("content")
    @classmethod
    def _content_size(cls, value: str | None) -> str | None:
        """Reject content over the byte cap before anything else runs."""
        if value is not None and len(value.encode()) > MAX_CONTENT_BYTES:
            msg = f"content is over {MAX_CONTENT_BYTES} bytes"
            raise PayloadTooLargeError(msg)
        return value

    @model_validator(mode="after")
    def _fields_match_kind(self) -> AddMemory:
        """Check the per-kind fields and the order of the valid window.

        Returns:
            The same request.

        Raises:
            ValueError: If the fields do not match the kind or the window
                does not run forward.
        """
        if self.kind is MemoryKind.FACT:
            missing = [
                name
                for name, value in (
                    ("subject", self.subject),
                    ("relation", self.relation),
                    ("object", self.object),
                )
                if not value
            ]
            if missing:
                msg = f"kind=fact needs {', '.join(missing)}"
                raise ValueError(msg)
        elif not (self.content and self.content.strip()):
            msg = f"kind={self.kind.value} needs non-empty content"
            raise ValueError(msg)
        if self.valid_from and self.valid_to and self.valid_to <= self.valid_from:
            msg = "valid_to must be after valid_from"
            raise ValueError(msg)
        return self


class UpdateMemory(WriteRequest):
    """Replace a fact: close the old version and assert the new one."""

    fact_id: str
    object: str | None = Field(default=None, description="The new object name.")
    statement: str | None = None
    effective_at: UtcDatetime | None = Field(
        default=None, description="When the change took effect. Defaults to now."
    )
    idempotency_key: str | None = Field(default=None, max_length=128)


class InvalidateMemory(WriteRequest):
    """Close one current fact while retaining its version history."""

    fact_id: str
    at: UtcDatetime | None = None
    reason: str = ""
    idempotency_key: str | None = Field(default=None, max_length=128)


class ForgetMemory(WriteRequest):
    """Close or remove memory. Hard modes need a dry run, then the token."""

    fact_id: str | None = None
    episode_id: str | None = None
    mode: ForgetMode = ForgetMode.INVALIDATE
    dry_run: bool = False
    confirm_token: str | None = None
    reason: str = ""

    @model_validator(mode="after")
    def _one_target(self) -> ForgetMemory:
        """Check that exactly one target is named.

        Returns:
            The same request.

        Raises:
            ValueError: If the target count is not one.
        """
        _exactly_one(fact_id=self.fact_id, episode_id=self.episode_id)
        return self


class ConsolidateMemory(WriteRequest):
    """Run the consolidation stages for a tenant as a background job."""


class ClearTenant(WriteRequest):
    """Remove everything in a tenant. Needs a dry run, then its token."""

    dry_run: bool = False
    confirm_token: str | None = None


class MemoryFeedback(ScopedRequest):
    """A score on one entity or on a fact's two entities."""

    entity_id: str | None = None
    fact_id: str | None = None
    score: float = Field(ge=-1.0, le=1.0)
    note: str = ""

    @model_validator(mode="after")
    def _one_target(self) -> MemoryFeedback:
        """Check that exactly one target is named.

        Returns:
            The same request.

        Raises:
            ValueError: If the target count is not one.
        """
        _exactly_one(entity_id=self.entity_id, fact_id=self.fact_id)
        return self


class SearchMemory(ContractModel):
    """Recall facts, entities and a ready-to-prompt context.

    The time fields are the retrieval request's filters: ``valid_at`` and
    ``known_at`` ask about one instant, ``overlaps_start`` with
    ``overlaps_end`` about a window. ``session_id`` limits facts to episodes
    associated with that session.
    """

    tenant: TenantId | None = None
    tenants: list[TenantId] = Field(default_factory=list)
    query: str = Field(min_length=1, max_length=2000)
    valid_at: UtcDatetime | None = None
    overlaps_start: UtcDatetime | None = None
    overlaps_end: UtcDatetime | None = None
    session_id: str | None = None
    known_at: UtcDatetime | None = None
    graph_depth: int = Field(default=1, ge=1)
    disputed: Literal["flag", "exclude"] = "flag"
    mentions: list[str] = Field(default_factory=list)
    include_history: bool = False
    budget_tokens: int = Field(default=2000, ge=100, le=32000)
    limit: int = Field(default=10, ge=1, le=MAX_RESULTS)
    format: Literal["json", "prompt", "both"] = "both"

    @model_validator(mode="after")
    def _overlaps_window(self) -> SearchMemory:
        """Check that the overlap window has two ends in order.

        Returns:
            The same request.

        Raises:
            ValueError: If one end is missing or the ends are out of order.
        """
        if (self.overlaps_start is None) != (self.overlaps_end is None):
            msg = "overlaps_start and overlaps_end must be given together"
            raise ValueError(msg)
        if (
            self.overlaps_start is not None
            and self.overlaps_end is not None
            and self.overlaps_start >= self.overlaps_end
        ):
            msg = "overlaps_start must be before overlaps_end"
            raise ValueError(msg)
        if self.tenant is not None and self.tenants:
            msg = "name either tenant or tenants, not both"
            raise ValueError(msg)
        return self


class GetEntity(ScopedRequest):
    """Look up one entity by id or name, with its facts."""

    entity_id: str | None = None
    name: str | None = None
    include_history: bool = False
    max_facts: int = Field(default=25, ge=1, le=MAX_RESULTS)

    @model_validator(mode="after")
    def _one_target(self) -> GetEntity:
        """Check that exactly one target is named.

        Returns:
            The same request.

        Raises:
            ValueError: If the target count is not one.
        """
        _exactly_one(entity_id=self.entity_id, name=self.name)
        return self


class ListEpisodes(ScopedRequest):
    """Page through stored episodes, newest first."""

    session_id: str | None = None
    cursor: str | None = None
    limit: int = Field(default=20, ge=1, le=MAX_RESULTS)


class ErrorInfo(ContractModel):
    """The wire form of a service error."""

    code: ErrorCode
    message: str
    hint: str = ""

    @classmethod
    def from_error(cls, error: ServiceError) -> ErrorInfo:
        """Build the wire form from a raised service error.

        Args:
            error: The error to translate.

        Returns:
            The same code, message and hint as a model.
        """
        return cls(code=error.code, message=error.message, hint=error.hint)


class WriteReceipt(ContractModel):
    """What a write returns at once: a job to follow."""

    job_id: str
    status: JobState
    tenant: TenantId
    idempotent_replay: bool = False
    trace_id: str = ""
    result: dict[str, Any] | None = Field(
        default=None,
        description=(
            "Payload of the write that ran; add_memory, update_memory, "
            "forget_memory, consolidate_memory and clear_tenant fill their "
            "own keys."
        ),
    )
    error: ErrorInfo | None = None


class JobView(ContractModel):
    """A job's state visible to an authorized caller.

    Attributes:
        job_id: Unique identifier of the submission.
        tenant: Tenant that owns the job.
        op: Operation performed by the job.
        status: Current job lifecycle state.
        attempts: Number of attempts already started.
        result: Result returned by completed work, if any.
        error: Failure details, if the job failed.
        trace_id: Trace id recorded when the job was submitted.
        progress: Latest persisted stage and count, if reported.
        created_at: Time the job was submitted.
        updated_at: Time of the last job state change.
    """

    job_id: str
    tenant: TenantId
    op: str
    status: JobState
    attempts: int = 0
    result: dict[str, Any] | None = Field(
        default=None,
        description=(
            "Payload of the verb named by op; each write verb fills its own "
            "keys once the job is done."
        ),
    )
    error: ErrorInfo | None = None
    trace_id: str = ""
    progress: JobProgress | None = None
    created_at: UtcDatetime
    updated_at: UtcDatetime


class ForgetResult(ContractModel):
    """The outcome of a forget, or of its dry run."""

    mode: ForgetMode | Literal["clear"]
    dry_run: bool
    fact_ids: list[str] = Field(default_factory=list)
    episode_ids: list[str] = Field(default_factory=list)
    collateral_fact_ids: list[str] = Field(default_factory=list)
    confirm_token: str | None = None
    receipt: WriteReceipt | None = None
    not_cleaned: list[str] = Field(
        default_factory=list,
        description="Residue erase found and could not clean; empty in other modes.",
    )


class EntityRef(ContractModel):
    """An entity named in a fact."""

    entity_id: str
    name: str = ""


class FactOut(ContractModel):
    """One fact with its validity window and provenance."""

    fact_id: str
    tenant: TenantId
    statement: str
    subject: EntityRef
    relation: str = ""
    object: EntityRef
    valid_from: UtcDatetime | None = None
    valid_to: UtcDatetime | None = None
    recorded_at: UtcDatetime | None = None
    expired_at: UtcDatetime | None = None
    status: str = "active"
    disputed: bool = False
    counterpart_ids: list[str] = Field(default_factory=list)
    evidence_count: int = 0
    episode_ids: list[str] = Field(default_factory=list)
    score: float = 0.0


class EntityOut(ContractModel):
    """A canonical entity."""

    entity_id: str
    tenant: TenantId
    name: str
    type_label: str = ""
    summary: str = ""
    aliases: list[str] = Field(default_factory=list)


class MemoryContext(ContractModel):
    """Search output. ``state`` separates "nothing stored" from "not ready"."""

    state: Literal["found", "none", "indexing", "degraded"]
    facts: list[FactOut] = Field(default_factory=list)
    entities: list[EntityOut] = Field(default_factory=list)
    rendered: str | None = None
    tokens_used: int = 0
    degraded: list[str] = Field(default_factory=list)
    trace_id: str = ""


class EntityView(ContractModel):
    """An entity and the facts that touch it."""

    entity: EntityOut
    facts: list[FactOut] = Field(default_factory=list)


class EpisodeOut(ContractModel):
    """One stored episode's metadata."""

    episode_id: str
    tenant: TenantId
    kind: str
    session_id: str | None = None
    thread: str | None = None
    source_description: str = ""
    job_id: str = ""
    created_at: UtcDatetime


class EpisodePage(ContractModel):
    """A page of episodes and the cursor for the next one."""

    episodes: list[EpisodeOut]
    next_cursor: str | None = None


class StatusOut(ContractModel):
    """Service health."""

    status: Literal["ok", "degraded"]
    backend: str
    llm_available: bool
    tenants: list[str] = Field(default_factory=list)
    queued_jobs: int = 0
    detail: dict[str, Any] = Field(
        default_factory=dict,
        description="Extra health fields the status verb chooses to expose.",
    )


def validation_error(errors: Sequence[Mapping[str, Any]]) -> ServiceError:
    """Turn pydantic error dicts into one typed service error.

    Args:
        errors: The error dicts of a pydantic ``ValidationError``.

    Returns:
        PayloadTooLargeError when any field was over its byte limit, so the
        caller shrinks the content first; otherwise InvalidArgumentError.
        Both carry every field path and message.
    """
    text = "; ".join(
        f"{'.'.join(str(part) for part in error['loc']) or 'request'}: {error['msg']}"
        for error in errors
    )
    hint = "Fix the named fields and call again."
    too_large = any(
        isinstance(error.get("ctx", {}).get("error"), PayloadTooLargeError)
        for error in errors
    )
    if too_large:
        return PayloadTooLargeError(text, hint=hint)
    return InvalidArgumentError(text, hint=hint)


def validate_request(model: type[ModelT], data: Mapping[str, Any]) -> ModelT:
    """Parse a raw request mapping into its contract model.

    Args:
        model: The contract model of the verb.
        data: The raw request fields.

    Returns:
        The validated model.

    Raises:
        InvalidArgumentError: If the fields do not validate.
        PayloadTooLargeError: If a field is over its byte limit and the
            validation failed.
    """
    try:
        return model.model_validate(dict(data))
    except ValidationError as exc:
        raise validation_error(exc.errors()) from None
