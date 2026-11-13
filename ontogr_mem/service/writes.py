"""Execute memory write jobs against one tenant runtime."""

from collections.abc import Mapping, Sequence
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
    TemporalStatus,
    TypeCatalog,
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
from ontogr_mem.entity_extraction.extractors import (
    EntityExtractor,
    Extracted,
    LLMExtractor,
    RecordMapper,
)
from ontogr_mem.fact_extraction import FactExtraction, InMemoryFactStore
from ontogr_mem.fact_extraction.extractor import LLMFactExtractor
from ontogr_mem.fact_extraction.ids import assign_ids
from ontogr_mem.fact_extraction.models import FactDropReason
from ontogr_mem.fact_extraction.text import text_of
from ontogr_mem.ingestion import IngestionPipeline, seal
from ontogr_mem.llm import ModelFailure, ModelRole
from ontogr_mem.ontology.extraction_items import stabilize_facts
from ontogr_mem.ontology.models import UNKNOWN_ID, Mode, Ontology, Reason, Relation
from ontogr_mem.ontology.validate import (
    Context,
    EntityItem,
    ExtractedItems,
    RelationItem,
)
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
from ontogr_mem.service.sourced_facts import (
    BindInput,
    Drop,
    bind_host_facts,
    ground_host_facts,
)
from ontogr_mem.temporal.query import FactView, TemporalQuery
from ontogr_mem.temporal.reader import FactReader
from ontogr_mem.temporal.report import Outcome, TemporalReport
from ontogr_mem.vectorstore.episode_vectors import write_episode_vectors


_LLM_HINT = "Configure both extraction roles, or use write_facts with source text."
_MIN_VALIDITY = timedelta(seconds=1)
# Strict mode drops a relation whose endpoint types do not fit its patterns.
_PATTERN_REASONS = frozenset({Reason.PATTERN_MISMATCH, Reason.DIRECTION_SWAPPED})


class _HostExtractor(EntityExtractor):
    """Return the source-grounded endpoint entities prepared for this job."""

    def __init__(self, entities: Sequence[ExtractedEntity]) -> None:
        self._entities = list(entities)

    async def extract(
        self,
        episode: Episode,
        types: TypeCatalog,
        previous: Sequence[Episode] = (),
    ) -> Extracted:
        """Return the prepared entities without reading the episode."""
        return Extracted(list(self._entities))


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
    text_write = request.kind in {MemoryKind.TEXT, MemoryKind.MESSAGE}
    if text_write and not (
        tenant.models.configured(ModelRole.ENTITY_EXTRACTION)
        and tenant.models.configured(ModelRole.FACT_EXTRACTION)
    ):
        raise LLMUnavailableError(
            "both extraction models are required for text memory", hint=_LLM_HINT
        )
    dropped: list[dict[str, Any]] = []
    try:
        if request.kind is MemoryKind.FACT:
            episode, facts = await _prepare_fact(tenant, job, request, principal_id)
        elif request.kind is MemoryKind.SOURCED_FACTS:
            episode, facts, dropped = await _prepare_sourced_facts(
                tenant, job, request, principal_id
            )
        else:
            episode, facts = await _prepare_text(tenant, job, request, principal_id)
    except ModelFailure as exc:
        raise LLMUnavailableError(str(exc), hint=_LLM_HINT) from exc

    typed = await stabilize_facts(
        tenant.ontology, facts, await tenant.ontology.extraction_view()
    )
    if request.kind is MemoryKind.FACT:
        for fact in typed.dropped:
            if fact.relation_type_id is not None or not fact.relation_label:
                raise InvalidArgumentError("one or more extracted facts were refused")
            dropped.append(
                {
                    "index": 0,
                    "reason": "unknown_relation",
                    "detail": "relation recorded for review; fact not stored",
                }
            )
    ontology = await tenant.ontology.catalog()
    tenant.refresh_relation_labels(ontology)
    report = await tenant.engine(ontology, tenant.resolver.store).ingest(
        typed.typed, episode, ontology_refused=len(typed.dropped)
    )
    text_write = request.kind in {MemoryKind.TEXT, MemoryKind.MESSAGE}
    sourced = request.kind is MemoryKind.SOURCED_FACTS
    _settled(report, allow_skips=text_write or sourced)
    state = await tenant.stores.temporal_store.episode_state(episode.id)
    return await _finish_episode(
        tenant,
        episode,
        request,
        ontology_refused=state.ontology_refused if state is not None else None,
        skipped=state.skipped if state is not None else None,
        dropped=dropped if sourced or request.kind is MemoryKind.FACT else None,
    )


async def _prepare_fact(
    tenant: TenantRuntime,
    job: Job,
    request: AddMemory,
    principal_id: str,
) -> tuple[Episode, list[ExtractedFact]]:
    """Build a typed fact, resolve its endpoints, and ingest its episode."""
    if request.subject is None or request.relation is None or request.object is None:
        raise InvalidArgumentError("kind=fact needs subject, relation and object")
    ontology = await tenant.ontology.catalog()
    relation: Relation | None = None
    if (
        ontology.relation_by_label(request.relation) is not None
        or ontology.relation_mode is not Mode.PREFERRED
    ):
        relation = await ensure_relation(
            tenant.ontology, request.relation, single_valued=request.single_valued
        )
        ontology = await tenant.ontology.catalog()
    tenant.refresh_relation_labels(ontology)
    projection = await tenant.ontology.extraction_view()
    if relation is None:
        subject_type, object_type = UNKNOWN_TYPE_ID, UNKNOWN_TYPE_ID
        relation_type = None
    else:
        subject_type, object_type = _endpoint_types(ontology, relation, projection.view)
        relation_type = projection.relation_ids.index(relation.type_id)
    relation_label = request.relation if relation is None else relation.label
    occurred_at = request.reference_time or job.created_at
    statement = request.statement or default_statement(
        request.subject, relation_label, request.object
    )
    draft = _draft(
        job,
        request,
        principal_id,
        Kind.EVENT,
        occurred_at,
        content={
            "subject": request.subject,
            "relation": relation_label,
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
        relation_label=request.relation if relation is None else None,
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
    if (
        not (
            tenant.models.configured(ModelRole.ENTITY_EXTRACTION)
            and tenant.models.configured(ModelRole.FACT_EXTRACTION)
        )
        or request.content is None
    ):
        raise LLMUnavailableError(
            "both extraction models are required for text memory", hint=_LLM_HINT
        )
    kind = Kind.MESSAGE if request.kind.value == "message" else Kind.DOCUMENT
    occurred_at = request.reference_time or job.created_at
    episode = await _ingest_draft(
        tenant,
        _draft(job, request, principal_id, kind, occurred_at, content=request.content),
    )
    state = await tenant.stores.temporal_store.episode_state(episode.id)
    if (
        state is not None
        and state.status is TemporalStatus.COMMITTED
        and state.committed_hash == episode.content_hash
    ):
        return episode, []
    projection = await tenant.ontology.extraction_view()
    entities = await EntityExtraction(
        [LLMExtractor(tenant.models)],
        RecordMapper({}),
        projection.view.catalog,
        previous=tenant.stores.episodes,
    ).extract([episode])
    entity_result = entities.results[0]
    if entity_result.status in {Status.ERROR, Status.PARSE_FAILED}:
        raise LLMUnavailableError(entity_result.detail, hint=_LLM_HINT)
    await tenant.resolver.resolve(entity_result.entities)
    fact_report = await FactExtraction(
        LLMFactExtractor(tenant.models),
        InMemoryFactStore(),
        projection.view,
        previous=tenant.stores.episodes,
    ).extract([episode], {episode.id: entity_result.entities})
    fact_result = fact_report.results[0]
    if fact_result.status in {Status.ERROR, Status.PARSE_FAILED}:
        raise LLMUnavailableError(fact_result.detail, hint=_LLM_HINT)
    return episode, fact_result.facts


# The steps share the validation report and projection, so the flow stays whole.
async def _prepare_sourced_facts(  # noqa: PLR0912, PLR0915
    tenant: TenantRuntime,
    job: Job,
    request: AddMemory,
    principal_id: str,
) -> tuple[Episode, list[ExtractedFact], list[dict[str, Any]]]:
    """Ground host facts in their source text and bind them without a model."""
    if request.content is None:
        raise InvalidArgumentError("kind=sourced_facts needs content")
    occurred_at = request.reference_time or job.created_at
    episode = await _ingest_draft(
        tenant,
        _draft(
            job,
            request,
            principal_id,
            Kind.DOCUMENT,
            occurred_at,
            content=request.content,
        ),
    )
    grounded = ground_host_facts(text_of(episode), request.facts or [])
    drops: list[Drop] = list(grounded.drops)
    bound_facts: list[ExtractedFact] = []
    if grounded.facts:
        # One call keeps the entity and relation review rows together.
        report = await tenant.ontology.process_extraction(
            ExtractedItems(
                entities=tuple(
                    EntityItem(
                        local_id=endpoint.ref,
                        raw_type=endpoint.type_label,
                        quote=endpoint.name,
                    )
                    for endpoint in grounded.endpoints
                ),
                relations=tuple(
                    RelationItem(
                        source=item.subject_ref,
                        target=item.object_ref,
                        raw_label=item.fact.relation,
                        quote=item.fact.evidence_quote,
                    )
                    for item in grounded.facts
                ),
            ),
            Context(episode.id),
        )
        ontology = await tenant.ontology.catalog()
        tenant.refresh_relation_labels(ontology)
        projection = await tenant.ontology.extraction_view()
        endpoint_verdicts = {
            endpoint.ref: verdict
            for endpoint, verdict in zip(
                grounded.endpoints, report.entities, strict=True
            )
        }
        relation_detail = (
            "relation recorded for review; fact not stored"
            if ontology.relation_mode is Mode.PREFERRED
            else "relation is not in the ontology"
        )
        survivors: list[tuple[BindInput, int]] = []
        for item, verdict in zip(grounded.facts, report.relations, strict=True):
            subject = endpoint_verdicts[item.subject_ref]
            target = endpoint_verdicts[item.object_ref]
            # A relation is dropped with its endpoint, so name the endpoint first.
            if not (subject.kept and target.kept):
                drops.append(
                    Drop(
                        item.index,
                        FactDropReason.UNKNOWN_ENTITY_TYPE,
                        "entity type is not in the ontology",
                    )
                )
                continue
            if not verdict.kept and _PATTERN_REASONS & set(verdict.reasons):
                drops.append(
                    Drop(
                        item.index,
                        FactDropReason.WRONG_DIRECTION,
                        "endpoint types do not fit the relation",
                    )
                )
                continue
            if (
                not verdict.kept
                or verdict.type_id is None
                or verdict.type_id not in projection.relation_ids
            ):
                drops.append(
                    Drop(item.index, FactDropReason.UNKNOWN_RELATION, relation_detail)
                )
                continue
            relation = ontology.relation(verdict.type_id)
            subject_id = _known_type(subject.type_id)
            object_id = _known_type(target.type_id)
            if (
                relation is not None
                and subject_id is not None
                and object_id is not None
                and not ontology.pattern_allows(relation, subject_id, object_id)
                and not (
                    relation.symmetric
                    and ontology.pattern_allows(relation, object_id, subject_id)
                )
            ):
                drops.append(
                    Drop(
                        item.index,
                        FactDropReason.WRONG_DIRECTION,
                        "endpoint types do not fit the relation",
                    )
                )
                continue
            survivors.append(
                (
                    BindInput(
                        item.index,
                        item.fact,
                        item.subject_ref,
                        item.object_ref,
                        projection.relation_ids.index(verdict.type_id),
                    ),
                    verdict.type_id,
                )
            )
        if survivors:
            used = {
                ref
                for bind_input, _ in survivors
                for ref in (bind_input.subject_ref, bind_input.object_ref)
            }
            entities: list[ExtractedEntity] = []
            for endpoint in grounded.endpoints:
                if endpoint.ref not in used:
                    continue
                stable = _known_type(endpoint_verdicts[endpoint.ref].type_id)
                unknown = stable is None or stable not in projection.entity_ids
                entities.append(
                    ExtractedEntity(
                        local_id=endpoint.ref,
                        name=endpoint.name,
                        type_id=(
                            UNKNOWN_TYPE_ID
                            if unknown
                            else projection.entity_ids.index(stable)
                        ),
                        confidence=1.0,
                        method=Method.RECORD,
                        episode_id=episode.id,
                        type_label=endpoint.type_label if unknown else None,
                    )
                )
            extraction = await EntityExtraction(
                [_HostExtractor(entities)], RecordMapper({}), projection.view.catalog
            ).extract([episode])
            result = extraction.results[0]
            if result.status in {Status.ERROR, Status.PARSE_FAILED}:
                raise InternalError(result.detail)
            bound = bind_host_facts(
                episode,
                [bind_input for bind_input, _ in survivors],
                result.entities,
                projection.view,
            )
            drops.extend(bound.drops)
            state = await tenant.stores.temporal_store.episode_state(episode.id)
            if (
                state is not None
                and state.status is TemporalStatus.COMMITTED
                and state.committed_hash == episode.content_hash
            ):
                # The earlier run stored its facts; only the drops are reported.
                return episode, [], _drop_results(drops)
            # The resolver raises on an episode it has already resolved.
            if not await tenant.resolver.store.episode_resolution_state(episode.id):
                await tenant.resolver.resolve(result.entities, allow_model=False)
            bound_facts = list(bound.facts)
    return episode, bound_facts, _drop_results(drops)


def _known_type(stable_id: int | None) -> int | None:
    """Return a stable entity type id, or None for a missing or Unknown type."""
    return None if stable_id is None or stable_id == UNKNOWN_ID else stable_id


def _drop_results(drops: Sequence[Drop]) -> list[dict[str, Any]]:
    """Return the drops as plain mappings ordered by input index."""
    return [drop.as_result() for drop in sorted(drops, key=lambda drop: drop.index)]


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
            role=add.role if add is not None else Role.USER,
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
    *,
    ontology_refused: int | None = None,
    skipped: Mapping[str, int] | None = None,
    dropped: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Store episode metadata, outcome counts and its episode vector."""
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
    result = {
        "episode_id": episode.id,
        "episode_ids": [episode.id],
        "fact_ids": fact_ids,
    }
    if add is not None:
        result["ontology_refused"] = ontology_refused
        result["skipped"] = dict(skipped) if skipped is not None else None
        if dropped is not None:
            result["dropped"] = [dict(item) for item in dropped]
    return result


async def _names(tenant: TenantRuntime, entity_ids: list[str]) -> dict[str, str]:
    """Return canonical names for existing entities."""
    entities = await tenant.stores.resolver_store.get_entities(entity_ids)
    return {entity.entity_id: entity.canonical_name for entity in entities}


def _settled(report: TemporalReport, *, allow_skips: bool = False) -> None:
    """Turn an unsuccessful temporal report into a service error."""
    if report.skipped and not allow_skips:
        reason, count = next(iter(report.skipped.items()))
        raise InvalidArgumentError(f"fact was skipped: {reason} ({count})")
    if not report.ok:
        unmatched = report.counts.get(Outcome.UNMATCHED_END.value, 0)
        detail = " with unmatched_end" if unmatched else ""
        raise InternalError(f"temporal reconciliation failed{detail}")
