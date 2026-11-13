"""Run the three model-backed consolidation stages with disposable local ports."""

from typing import Any

import pytest
from pydantic import BaseModel
from pydantic_ai import models

from ontogr_mem.consolidation.agents import writer_agent
from ontogr_mem.consolidation.config import ConsolidationSettings
from ontogr_mem.consolidation.models import (
    Bucket,
    Community,
    LessonVerdict,
    Proposal,
    SessionEntry,
    StageResult,
    StageStatus,
    Watermark,
)
from ontogr_mem.consolidation.stage import StageContext
from ontogr_mem.consolidation.stages.communities import CommunitiesStage
from ontogr_mem.consolidation.stages.descriptions import DescriptionsStage
from ontogr_mem.consolidation.stages.reflection import ReflectionStage
from ontogr_mem.consolidation.text import quote
from ontogr_mem.data_models.fact_versions import FactStatus
from ontogr_mem.llm import LLMConfig, ModelRole, RoleModels, build_models, run_agent
from ontogr_mem.temporal.query import FactView
from tests.unit.consolidation.stage_fakes import (
    NOW,
    MemoryResolution,
    MemoryStore,
    MemoryTemporal,
    entity,
)


pytestmark = [pytest.mark.slow, pytest.mark.enable_socket]

_LONG_DESCRIPTION = "Founded in Berlin in 2010. " * 100
_LESSON = "I prefer concise replies."
_ENTRY_TEXT = "Please keep future replies concise."


class _StageStore(MemoryStore):
    """Add community and bucket documents to the shared in-memory stage store."""

    def __init__(self, *documents: BaseModel) -> None:
        """Create an empty stage store."""
        super().__init__(*documents)
        self._group_documents: dict[tuple[type[BaseModel], str], BaseModel] = {}

    async def list(self, model: type[Any]) -> list[Any]:
        """Return group documents or delegate to the shared stage fake."""
        if model in (Community, Bucket):
            return [
                document
                for (document_model, _), document in sorted(
                    self._group_documents.items(), key=lambda item: item[0][1]
                )
                if document_model is model
            ]
        return await super().list(model)

    async def get(self, model: type[Any], document_id: str) -> Any | None:
        """Return one group document or delegate to the shared stage fake."""
        if model in (Community, Bucket):
            return self._group_documents.get((model, document_id))
        return await super().get(model, document_id)

    async def put(self, document: BaseModel) -> None:
        """Store group documents locally and delegate other records."""
        if isinstance(document, Community):
            identifier = document.community_id
        elif isinstance(document, Bucket):
            identifier = document.bucket_id
        else:
            await super().put(document)
            return
        self._group_documents[(type(document), identifier)] = document

    async def delete(self, model: type[Any], document_id: str) -> None:
        """Delete a community or bucket document when its input is stale."""
        if model not in (Community, Bucket):
            raise TypeError(f"unsupported document: {model.__name__}")
        self._group_documents.pop((model, document_id), None)


def _context(
    models_by_role: RoleModels,
    store: _StageStore,
    resolution: MemoryResolution,
    temporal: MemoryTemporal,
) -> StageContext:
    """Build a stage context over disposable in-memory ports."""
    return StageContext(
        run_id="live-consolidation-check",
        clock=lambda: NOW,
        store=store,
        settings=ConsolidationSettings(_env_file=None),
        models=models_by_role,
        resolution=resolution,
        temporal=temporal,
    )


def _request_count(result: StageResult) -> int:
    """Return model requests recorded by one stage."""
    return sum(bucket.calls for bucket in result.usage.buckets.values())


async def test_live_consolidation_stages_use_the_configured_model() -> None:
    """Summaries, communities and reflection succeed with the configured model."""
    config = LLMConfig()
    if config.resolve(ModelRole.CONSOLIDATION).model is None:
        pytest.skip("the consolidation model is not configured")

    person = entity("person", [_LONG_DESCRIPTION])
    company = entity("company", ["Operates in Berlin."])
    resolution = MemoryResolution(person, company)
    temporal = MemoryTemporal(
        (
            FactView(
                fact_id="employment-fact",
                version=1,
                slot_key="employment",
                subject_id="person",
                object_id="company",
                relation_type_id=1,
                valid_from=NOW,
                valid_to=None,
                valid_from_precision=None,
                valid_to_precision=None,
                recorded_at=NOW,
                expired_at=None,
                status=FactStatus.ACTIVE,
                evidence=(),
                statement="The person works at the company.",
            ),
        )
    )
    entry = SessionEntry(
        entry_id="live-entry",
        session_id="live-session",
        seq=1,
        text=_ENTRY_TEXT,
        helpfulness=1,
        confidence=1.0,
        created_at=NOW,
    )
    store = _StageStore(entry)

    with models.override_allow_model_requests(True):
        role_models = build_models(config)
        try:
            context = _context(role_models, store, resolution, temporal)
            description_result = await DescriptionsStage().run(context)
            if description_result.status is not StageStatus.OK:
                pytest.fail(
                    "description stage did not complete successfully", pytrace=False
                )
            summary = resolution.entities["person"].summary
            if not summary.strip() or resolution.entities["person"].summary_stale:
                pytest.fail(
                    "description stage did not save a fresh summary", pytrace=False
                )

            community_result = await CommunitiesStage().run(context)
            if community_result.status is not StageStatus.OK:
                pytest.fail(
                    "community stage did not complete successfully", pytrace=False
                )
            communities = await store.list(Community)
            if not any(
                len(item.member_ids) == 2
                and item.description.strip()
                and item.content_hash
                for item in communities
            ):
                pytest.fail(
                    "community stage did not save a described community", pytrace=False
                )

            reflection_result = await ReflectionStage().run(context)
            if (
                reflection_result.status is not StageStatus.OK
                or reflection_result.errors
            ):
                pytest.fail(
                    "reflection stage did not complete successfully", pytrace=False
                )
            watermark = await store.get(Watermark, "reflection|live-session")
            if watermark is None or watermark.value != "1":
                pytest.fail(
                    "reflection stage did not advance its watermark", pytrace=False
                )

            writer_requests = 0
            proposals = await store.list(Proposal)
            if not any(proposal.source_entry_ids for proposal in proposals):
                prompt = (
                    f"Lesson: {quote(_LESSON)}\n"
                    f"Source entries: {quote([_ENTRY_TEXT])}\n"
                    f"Known lessons: {quote([])}"
                )
                writer_result = await run_agent(
                    writer_agent,
                    ModelRole.CONSOLIDATION,
                    prompt,
                    models=role_models,
                    deps=None,
                )
                if not isinstance(writer_result.output.verdict, LessonVerdict):
                    pytest.fail("writer did not return a typed verdict", pytrace=False)
                writer_requests = writer_result.usage.requests

            print(
                " ".join(
                    (
                        f"descriptions={description_result.status.value}"
                        f" requests={_request_count(description_result)}",
                        f"communities={community_result.status.value}"
                        f" requests={_request_count(community_result)}",
                        f"reflection={reflection_result.status.value}"
                        f" requests={_request_count(reflection_result)}",
                        f"writer requests={writer_requests}",
                    )
                )
            )
        finally:
            await role_models.aclose()
