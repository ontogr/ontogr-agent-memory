"""Execute memory write jobs against one tenant runtime."""

from collections.abc import Mapping
from datetime import datetime, timedelta
from typing import Any

from ontogr_mem.data_models import (
    UNKNOWN_TYPE_ID,
    Actor,
    Episode,
    EpisodeDraft,
    EpisodeMeta,
    ExtractedEntity,
    Kind,
    Method,
    Role,
    Source,
    Status,
)
from ontogr_mem.data_models.fact_versions import FactStatus
from ontogr_mem.data_models.facts import (
    ExtractedFact,
    FactKind,
    Modality,
    Polarity,
    Precision,
    TimeBasis,
)
from ontogr_mem.data_models.jobs import Job
from ontogr_mem.data_models.type_catalog import TypeView
from ontogr_mem.entity_extraction import EntityExtraction
from ontogr_mem.entity_extraction.extractors import LLMExtractor, RecordMapper
from ontogr_mem.fact_extraction import FactExtraction, InMemoryFactStore
from ontogr_mem.fact_extraction.extractor import LLMFactExtractor
from ontogr_mem.fact_extraction.ids import assign_ids
from ontogr_mem.ingestion import IngestionPipeline, seal
from ontogr_mem.llm import LLMError
from ontogr_mem.ontology.extraction_items import stabilize_facts
from ontogr_mem.ontology.models import Ontology, Relation
from ontogr_mem.request_context import current_context
from ontogr_mem.service.connectors import draft_registry
from ontogr_mem.service.contracts import (
    AddMemory,
    InvalidateMemory,
    MemoryKind,
    UpdateMemory,
)
from ontogr_mem.service.errors import (
    ConflictError,
    InternalError,
    InvalidArgumentError,
    LLMUnavailableError,
    ObjectNotFoundError,
)
from ontogr_mem.service.formatting import default_statement
from ontogr_mem.service.relations import ensure_relation
from ontogr_mem.service.runtime import TenantRuntime
from ontogr_mem.temporal.query import FactView, TemporalQuery
from ontogr_mem.temporal.reader import FactReader
from ontogr_mem.temporal.report import Outcome, TemporalReport
from ontogr_mem.vectorstore.episode_vectors import write_episode_vectors


_LLM_HINT = "Retry, or write a structured fact with kind=fact."
_MIN_VALIDITY = timedelta(seconds=1)


class _StaticRefs:
    """Resolve episode-local references to existing entity ids."""

    def __init__(self, refs: Mapping[str, str]) -> None:
        self._refs = refs

    async def entity_id_for_ref(self, episode_id: str, local_id: str) -> str | None:
        """Return the mapped id. The job already fixed the episode id."""
        return self._refs.get(local_id)


async def add_memory(
    tenant: TenantRuntime, job: Job, payload: Mapping[str, Any]
) -> dict[str, Any]:
    """Store a structured fact or extract facts from a text episode."""
    request = AddMemory.model_validate(payload["request"])
    principal_id = str(payload["principal_id"])
    if request.kind is not MemoryKind.FACT and tenant.llm is None:
        raise LLMUnavailableError(
            "a language model is required for text memory", hint=_LLM_HINT
        )
    try:
        if request.kind is MemoryKind.FACT:
            episode, facts = await _prepare_fact(tenant, job, request, principal_id)
        else:
            episode, facts = await _prepare_text(tenant, job, request, principal_id)
    except LLMError as exc:
        raise LLMUnavailableError(str(exc), hint=_LLM_HINT) from exc

    typed = await stabilize_facts(
        tenant.ontology, facts, await tenant.ontology.extraction_view()
    )
    if typed.dropped:
        raise InvalidArgumentError("one or more extracted facts were refused")
    ontology = await tenant.ontology.catalog()
    tenant.refresh_relation_labels(ontology)
    report = await tenant.engine(ontology, tenant.resolver.store).ingest(
        typed.typed, episode
    )
    _settled(report)
    return await _finish_episode(tenant, episode, request)


async def _prepare_fact(
    tenant: TenantRuntime,
    job: Job,
    request: AddMemory,
    principal_id: str,
) -> tuple[Episode, list[ExtractedFact]]:
    """Build a typed fact, resolve its endpoints, and ingest its episode."""
    if request.subject is None or request.relation is None or request.object is None:
        raise InvalidArgumentError("kind=fact needs subject, relation and object")
    relation = await ensure_relation(
        tenant.ontology, request.relation, single_valued=request.single_valued
    )
    ontology = await tenant.ontology.catalog()
    tenant.refresh_relation_labels(ontology)
    projection = await tenant.ontology.extraction_view()
    subject_type, object_type = _endpoint_types(ontology, relation, projection.view)
    relation_type = projection.relation_ids.index(relation.type_id)
    occurred_at = request.reference_time or job.created_at
    statement = request.statement or default_statement(
        request.subject, relation.label, request.object
    )
    draft = _draft(
        job,
        request,
        principal_id,
        Kind.EVENT,
        occurred_at,
        content={
            "subject": request.subject,
            "relation": relation.label,
            "object": request.object,
            "statement": statement,
        },
    )
    episode = await _ingest_draft(tenant, draft)
    entities = [
        ExtractedEntity(
            local_id=local_id,
            name=name,
            type_id=entity_type,
            confidence=1.0,
            method=Method.RECORD,
            episode_id=episode.id,
        )
        for local_id, name, entity_type in (
            ("s", request.subject, subject_type),
            ("o", request.object, object_type),
        )
    ]
    await tenant.resolver.resolve(entities)
    valid_from = request.valid_from or request.reference_time or occurred_at
    time_basis = (
        TimeBasis.STATED if request.valid_from is not None else TimeBasis.REFERENCE
    )
    fact = ExtractedFact(
        fact_id="",
        text=statement,
        entity_refs=("s", "o"),
        relation_type_id=relation_type,
        relation_label=None,
        fact_kind=FactKind.STATE,
        polarity=Polarity.AFFIRMED,
        modality=Modality.ACTUAL,
        asserted_by=principal_id,
        valid_from=valid_from,
        valid_to=request.valid_to,
        precision=Precision.DAY if request.valid_from is not None else Precision.TIME,
        time_expression=None,
        time_basis=time_basis,
        evidence_quote=statement,
        evidence_span=None,
        confidence=1.0,
        method=Method.RECORD,
        episode_id=episode.id,
    )
    return episode, assign_ids(episode.id, [fact])


async def _prepare_text(
    tenant: TenantRuntime,
    job: Job,
    request: AddMemory,
    principal_id: str,
) -> tuple[Episode, list[ExtractedFact]]:
    """Extract entities and facts from a text or message episode."""
    if tenant.llm is None or request.content is None:
        raise LLMUnavailableError(
            "a language model is required for text memory", hint=_LLM_HINT
        )
    kind = Kind.MESSAGE if request.kind.value == "message" else Kind.DOCUMENT
    occurred_at = request.reference_time or job.created_at
    episode = await _ingest_draft(
        tenant,
        _draft(job, request, principal_id, kind, occurred_at, content=request.content),
    )
    projection = await tenant.ontology.extraction_view()
    entities = await EntityExtraction(
        [LLMExtractor(tenant.llm)],
        RecordMapper({}),
        projection.view.catalog,
        previous=tenant.stores.episodes,
    ).extract([episode])
    entity_result = entities.results[0]
    if entity_result.status in {Status.ERROR, Status.PARSE_FAILED}:
        raise LLMUnavailableError(entity_result.detail, hint=_LLM_HINT)
    await tenant.resolver.resolve(entity_result.entities)
    fact_report = await FactExtraction(
        LLMFactExtractor(tenant.llm),
        InMemoryFactStore(),
        projection.view,
        previous=tenant.stores.episodes,
    ).extract([episode], {episode.id: entity_result.entities})
    fact_result = fact_report.results[0]
    if fact_result.status in {Status.ERROR, Status.PARSE_FAILED}:
        raise LLMUnavailableError(fact_result.detail, hint=_LLM_HINT)
    return episode, fact_result.facts


# The update's steps run in order and share one episode, so the flow stays whole.
async def update_memory(  # noqa: PLR0915
    tenant: TenantRuntime, job: Job, payload: Mapping[str, Any]
) -> dict[str, Any]:
    """Close a current fact and assert its replacement in one episode."""
    request = UpdateMemory.model_validate(payload["request"])
    current = await _latest_fact(
        tenant.reader,
        request.fact_id,
        tenant.clock.now(),
    )
    if current is None:
        raise ObjectNotFoundError("fact not found")
    if current.status is not FactStatus.ACTIVE:
        raise ConflictError("fact is closed, not active")
    effective = request.effective_at
    if effective is None:
        now = tenant.clock.now()
        effective = max(now, current.valid_from + _MIN_VALIDITY)
    elif effective <= current.valid_from:
        raise InvalidArgumentError("effective_at must be after valid_from")
    ontology = await tenant.ontology.catalog()
    relation = ontology.relation(current.relation_type_id)
    if relation is None:
        raise InternalError("fact relation is missing from the ontology")
    names = await _names(tenant, [current.subject_id, current.object_id])
    subject_name = names.get(current.subject_id, current.subject_id)
    old_object_name = names.get(current.object_id, current.object_id)
    object_name = request.object
    if object_name is None:
        raise InvalidArgumentError("update needs a new object")
    old_statement = current.statement or default_statement(
        subject_name, relation.label, old_object_name
    )
    statement = request.statement or default_statement(
        subject_name, relation.label, object_name
    )
    draft = _draft(
        job,
        request,
        str(payload["principal_id"]),
        Kind.EVENT,
        effective,
        content={"op": "update", "fact_id": request.fact_id, "object": object_name},
    )
    episode = await _ingest_draft(tenant, draft)
    projection = await tenant.ontology.extraction_view()
    _, object_type = _endpoint_types(ontology, relation, projection.view)
    relation_type = projection.relation_ids.index(relation.type_id)
    new_entity = ExtractedEntity(
        local_id="n",
        name=object_name,
        type_id=object_type,
        confidence=1.0,
        method=Method.RECORD,
        episode_id=episode.id,
    )
    await tenant.resolver.resolve([new_entity])
    new_id = await tenant.resolver.store.entity_id_for_ref(episode.id, "n")
    if new_id is None:
        raise InternalError("new object reference did not resolve")
    end = _fact_for_update(
        episode,
        old_statement,
        ("s", "o"),
        relation_type,
        effective,
        polarity=Polarity.NEGATED,
    )
    assertion = _fact_for_update(
        episode,
        statement,
        ("s", "n"),
        relation_type,
        effective,
        polarity=Polarity.AFFIRMED,
    )
    facts = assign_ids(episode.id, [end, assertion])
    stabilized = await stabilize_facts(
        tenant.ontology, facts, await tenant.ontology.extraction_view()
    )
    if stabilized.dropped:
        raise InvalidArgumentError("one or more update facts were refused")
    refs = _StaticRefs({"s": current.subject_id, "o": current.object_id, "n": new_id})
    report = await tenant.engine(ontology, refs).ingest(stabilized.typed, episode)
    _settled(report)
    closed = await _latest_fact(tenant.reader, request.fact_id, tenant.clock.now())
    if closed is None:
        raise InternalError("the updated fact is missing after temporal reconciliation")
    try:
        ensure_closed(closed)
    except InternalError as exc:
        if report.counts.get(Outcome.UNMATCHED_END.value, 0):
            raise InternalError(f"{exc} with unmatched_end") from exc
        raise
    output = await _finish_episode(tenant, episode, request)
    return {**output, "closed_fact_id": request.fact_id}


async def invalidate_memory(
    tenant: TenantRuntime, job: Job, payload: Mapping[str, Any]
) -> dict[str, Any]:
    """Close one active fact and retain its history in a new episode."""
    request = InvalidateMemory.model_validate(payload["request"])
    principal_id = str(payload["principal_id"])
    versions = await tenant.stores.temporal_store.facts_by_ids([request.fact_id])
    current = max(
        (fact for fact in versions if fact.current),
        key=lambda fact: fact.version,
        default=None,
    )
    if current is None:
        raise ObjectNotFoundError("fact not found")
    if current.status is not FactStatus.ACTIVE:
        raise ConflictError("fact is closed, not active")

    end = request.at or tenant.clock.now()
    applied_end = max(end, current.valid_from)
    draft = _draft(
        job,
        request,
        principal_id,
        Kind.EVENT,
        applied_end,
        content={
            "op": "invalidate",
            "fact_ids": [request.fact_id],
            "reason": request.reason,
        },
    )
    episode = await _ingest_draft(tenant, draft)
    projection = await tenant.ontology.extraction_view()
    try:
        dense_relation_id = projection.relation_ids.index(current.relation_type_id)
    except ValueError:
        raise InternalError("fact relation is missing from the ontology") from None

    subject_ref, object_ref = "s0", "o0"
    statement = request.reason or f"Invalidated fact {request.fact_id}"
    extracted = ExtractedFact(
        fact_id="",
        text=statement,
        entity_refs=(subject_ref, object_ref),
        relation_type_id=dense_relation_id,
        relation_label=None,
        fact_kind=FactKind.STATE,
        polarity=Polarity.NEGATED,
        modality=Modality.ACTUAL,
        asserted_by=principal_id,
        valid_from=applied_end,
        valid_to=None,
        precision=Precision.TIME,
        time_expression=None,
        time_basis=TimeBasis.STATED,
        evidence_quote=statement,
        evidence_span=None,
        confidence=1.0,
        method=Method.RECORD,
        episode_id=episode.id,
    )
    stabilized = await stabilize_facts(
        tenant.ontology, assign_ids(episode.id, [extracted]), projection
    )
    if stabilized.dropped:
        raise InvalidArgumentError("invalidation fact was refused")
    refs = _StaticRefs({subject_ref: current.subject_id, object_ref: current.object_id})
    report = await tenant.engine(await tenant.ontology.catalog(), refs).ingest(
        stabilized.typed, episode
    )
    _settled(report)
    closed_versions = await tenant.stores.temporal_store.facts_by_ids([request.fact_id])
    closed = max(
        (fact for fact in closed_versions if fact.current),
        key=lambda fact: fact.version,
        default=None,
    )
    if closed is None or closed.status is FactStatus.ACTIVE:
        raise InternalError("fact is still active after invalidation")
    output = await _finish_episode(tenant, episode, request)
    return {
        **output,
        "closed_fact_ids": [request.fact_id],
        "applied_end": applied_end.isoformat(),
    }


async def run_consolidation(tenant: TenantRuntime) -> dict[str, Any]:
    """Run the tenant's consolidation stages and return their public summary."""
    result = await (await tenant.consolidation()).run()
    return {
        "status": result.status.value,
        "stages": [
            {
                "stage": stage.stage,
                "status": stage.status.value,
                "notes": stage.notes,
                "errors": stage.errors,
            }
            for stage in result.stages
        ],
        "rerun_pending": result.rerun_pending,
    }


def _fact_for_update(
    episode: Episode,
    statement: str,
    refs: tuple[str, str],
    relation_type: int,
    effective: datetime,
    *,
    polarity: Polarity,
) -> ExtractedFact:
    """Build one stated-end or assertion fact for an update."""
    return ExtractedFact(
        fact_id="",
        text=statement,
        entity_refs=refs,
        relation_type_id=relation_type,
        relation_label=None,
        fact_kind=FactKind.STATE,
        polarity=polarity,
        modality=Modality.ACTUAL,
        asserted_by=None,
        valid_from=effective,
        valid_to=None,
        precision=Precision.TIME,
        time_expression=None,
        time_basis=TimeBasis.STATED,
        evidence_quote=statement,
        evidence_span=None,
        confidence=1.0,
        method=Method.RECORD,
        episode_id=episode.id,
    )


async def _latest_fact(
    reader: FactReader, fact_id: str, now: datetime
) -> FactView | None:
    """Return the newest version of a fact, including closed history."""
    facts = await reader.by_ids([fact_id], TemporalQuery(now=now, include_history=True))
    if not facts:
        return None
    return max(facts, key=lambda fact: fact.version)


def _endpoint_types(
    ontology: Ontology, relation: Relation, view: TypeView
) -> tuple[int, int]:
    """Return dense entity ids for a relation's first endpoint pattern."""
    if not relation.patterns:
        return UNKNOWN_TYPE_ID, UNKNOWN_TYPE_ID
    pattern = relation.patterns[0]
    source = ontology.entity(pattern.src)
    target = ontology.entity(pattern.tgt)
    if source is None or target is None:
        return UNKNOWN_TYPE_ID, UNKNOWN_TYPE_ID
    source_type = view.catalog.entity_by_label(source.label)
    target_type = view.catalog.entity_by_label(target.label)
    return (
        source_type.id if source_type is not None else UNKNOWN_TYPE_ID,
        target_type.id if target_type is not None else UNKNOWN_TYPE_ID,
    )


def ensure_closed(view: FactView) -> None:
    """Reject an update whose old fact remains active after reconciliation."""
    if view.status is FactStatus.ACTIVE:
        raise InternalError("fact is still active")


def _draft(
    job: Job,
    request: AddMemory | UpdateMemory | InvalidateMemory,
    principal_id: str,
    kind: Kind,
    occurred_at: datetime,
    *,
    content: str | dict[str, Any],
) -> EpisodeDraft:
    """Build the episode envelope for a service write."""
    context = current_context()
    add = request if isinstance(request, AddMemory) else None
    return EpisodeDraft(
        kind=kind,
        occurred_at=occurred_at,
        source=Source(
            system="agent-memory",
            ref=job.job_id,
            connector="agent-interface",
            description=add.source_description if add is not None else "",
        ),
        actor=Actor(
            id=(add.speaker if add is not None and add.speaker else principal_id),
            role=Role.USER,
            display_name=add.speaker or "" if add is not None else "",
        ),
        content=content,
        metadata={"trace_id": context.trace_id if context is not None else ""},
    )


async def _ingest_draft(tenant: TenantRuntime, draft: EpisodeDraft) -> Episode:
    """Seal and store one service-created episode."""
    episode = seal(draft, ingested_at=tenant.clock.now())
    await IngestionPipeline(
        tenant.stores.episodes, draft_registry(), tenant.clock
    ).ingest(draft.kind, [draft])
    return episode


async def _finish_episode(
    tenant: TenantRuntime,
    episode: Episode,
    request: AddMemory | UpdateMemory | InvalidateMemory,
) -> dict[str, Any]:
    """Store episode metadata and write its episode vector."""
    facts = await tenant.stores.temporal_store.facts_of_episode(episode.id)
    fact_ids = list(dict.fromkeys(fact.fact_id for fact in facts))
    add = request if isinstance(request, AddMemory) else None
    await tenant.stores.episodes.write_episode_meta(
        EpisodeMeta(
            episode_id=episode.id,
            session_id=add.session_id if add is not None else None,
            thread=add.thread if add is not None else None,
            fact_ids=tuple(fact_ids),
        )
    )
    await write_episode_vectors(tenant.writer, [episode])
    return {
        "episode_id": episode.id,
        "episode_ids": [episode.id],
        "fact_ids": fact_ids,
    }


async def _names(tenant: TenantRuntime, entity_ids: list[str]) -> dict[str, str]:
    """Return canonical names for existing entities."""
    entities = await tenant.stores.resolver_store.get_entities(entity_ids)
    return {entity.entity_id: entity.canonical_name for entity in entities}


def _settled(report: TemporalReport) -> None:
    """Turn an unsuccessful temporal report into a service error."""
    if report.skipped:
        reason, count = next(iter(report.skipped.items()))
        raise InvalidArgumentError(f"fact was skipped: {reason} ({count})")
    if not report.ok:
        unmatched = report.counts.get(Outcome.UNMATCHED_END.value, 0)
        detail = " with unmatched_end" if unmatched else ""
        raise InternalError(f"temporal reconciliation failed{detail}")
