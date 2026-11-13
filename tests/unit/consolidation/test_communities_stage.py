"""Offline community-stage behavior with typed, scripted model responses."""

from datetime import UTC, datetime
from typing import TypeVar

import pytest
from pydantic import BaseModel

import ontogr_mem.consolidation.stages.communities as communities_module
from ontogr_mem.consolidation.agents import (
    COMMUNITY_DESCRIPTION_SYSTEM,
    COMMUNITY_SYSTEM,
)
from ontogr_mem.consolidation.config import CommunitySettings, ConsolidationSettings
from ontogr_mem.consolidation.graph import EntityGraph
from ontogr_mem.consolidation.models import Bucket, Community, StageStatus
from ontogr_mem.consolidation.stage import StageContext
from ontogr_mem.consolidation.stages.communities import CommunitiesStage, community_id
from ontogr_mem.data_models.entities import Entity
from ontogr_mem.data_models.extraction import Usage
from ontogr_mem.llm import ModelCallFailedError, ModelRole, RoleModels
from ontogr_mem.llm.run import TokenUsage
from tests.support.scripted_model import (
    ScriptedCall,
    scripted_model,
    scripted_role_models,
)


NOW = datetime(2030, 1, 1, tzinfo=UTC)


M = TypeVar("M", bound=BaseModel)


class _MemoryStore:
    """Store the community and bucket documents used by this stage."""

    def __init__(self, *documents: Community | Bucket) -> None:
        self.documents: dict[tuple[type[BaseModel], str], BaseModel] = {}
        for document in documents:
            self._put(document)

    async def get(self, model: type[M], doc_id: str) -> M | None:
        document = self.documents.get((model, doc_id))
        return None if document is None else model.model_validate(document)

    async def put(self, document: BaseModel) -> None:
        self._put(document)

    async def list(self, model: type[M]) -> list[M]:
        return [
            model.model_validate(document)
            for (document_model, _), document in sorted(
                self.documents.items(), key=lambda item: item[0][1]
            )
            if document_model is model
        ]

    async def delete(self, model: type[BaseModel], doc_id: str) -> None:
        self.documents.pop((model, doc_id), None)

    def _put(self, document: BaseModel) -> None:
        if isinstance(document, Community):
            document_id = document.community_id
        elif isinstance(document, Bucket):
            document_id = document.bucket_id
        else:
            raise TypeError(f"unsupported document: {type(document).__name__}")
        self.documents[(type(document), document_id)] = document


def _graph() -> EntityGraph:
    """Return one three-entity community for summary reduction tests."""
    entities = {
        entity_id: Entity(
            entity_id=entity_id,
            type_label="Thing",
            canonical_name=entity_id,
            strict_key=entity_id,
            loose_key=entity_id,
            descriptions=[f"{entity_id} detail"],
            created_at=NOW,
            updated_at=NOW,
        )
        for entity_id in "abc"
    }
    neighbours = {entity_id: {} for entity_id in entities}
    for left, right in (("a", "b"), ("a", "c"), ("b", "c")):
        neighbours[left][right] = 1
        neighbours[right][left] = 1
    return EntityGraph(entities, neighbours, ())


def _context(models: RoleModels, store: _MemoryStore | None = None) -> StageContext:
    """Build a stage context with an offline store and fixed clock."""
    return StageContext(
        run_id="communities-test",
        clock=lambda: NOW,
        store=store or _MemoryStore(),
        settings=ConsolidationSettings(
            community=CommunitySettings(summary_max_chars=16)
        ),
        models=models,
    )


def _answer(call: ScriptedCall) -> dict[str, str]:
    """Return the typed output for the matching community request."""
    if call.system == COMMUNITY_SYSTEM:
        return {"summary": "A sentence. Extra text."}
    if call.system == COMMUNITY_DESCRIPTION_SYSTEM:
        return {"description": "A connected group."}
    raise AssertionError(f"unexpected agent instructions: {call.system}")


class TestCommunitiesStage:
    """Typed summaries preserve community reuse and failure behavior."""

    async def test_reduces_and_describes_then_reuses_unchanged_content(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Pair reduction truncates summaries and reuses unchanged content."""
        graph = _graph()

        async def build_graph(_ctx: StageContext) -> EntityGraph:
            return graph

        monkeypatch.setattr(communities_module, "build_entity_graph", build_graph)
        model, calls = scripted_model(answer=_answer)
        models = scripted_role_models(consolidation=model)
        store = _MemoryStore()

        first = await CommunitiesStage().run(_context(models, store))
        communities = await store.list(Community)

        assert [
            (item.member_ids, item.summary, item.description) for item in communities
        ] == [(["a", "b", "c"], "A sentence.", "A connected group.")]
        assert first.status is StageStatus.OK
        assert first.watermark is not None
        assert next(iter(first.usage.buckets.values())).calls == 3
        assert len(calls.calls) == 3

        second = await CommunitiesStage().run(_context(models, store))

        assert second.status is StageStatus.OK
        assert second.usage.buckets == {}
        assert await store.list(Community) == communities
        assert calls.calls and len(calls.calls) == 3

    async def test_failed_summary_keeps_old_text_and_counts_failed_requests(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A provider failure keeps old community text and leaves it stale."""
        graph = _graph()

        async def build_graph(_ctx: StageContext) -> EntityGraph:
            return graph

        monkeypatch.setattr(communities_module, "build_entity_graph", build_graph)
        model, calls = scripted_model(
            [ModelCallFailedError("provider down", usage=TokenUsage(17, 4, 2))]
        )
        existing = Community(
            community_id=community_id(["a", "b", "c"]),
            member_ids=["a", "b", "c"],
            summary="Last good summary.",
            description="Last good description.",
            content_hash="old-content",
        )
        store = _MemoryStore(existing)

        result = await CommunitiesStage().run(
            _context(scripted_role_models(consolidation=model), store)
        )

        stored = await store.get(Community, existing.community_id)
        assert result.status is StageStatus.PARTIAL
        assert result.watermark is None
        assert stored is not None
        assert (stored.summary, stored.description, stored.content_hash) == (
            "Last good summary.",
            "Last good description.",
            "",
        )
        usage = result.usage.buckets[f"communities/{ModelRole.CONSOLIDATION.value}"]
        assert (usage.input_tokens, usage.output_tokens, usage.calls) == (17, 4, 2)
        assert len(calls.calls) == 1

    async def test_invalid_summary_reasks_and_counts_all_requests(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Three malformed summaries keep old text and hold the watermark."""
        graph = _graph()

        async def build_graph(_ctx: StageContext) -> EntityGraph:
            return graph

        monkeypatch.setattr(communities_module, "build_entity_graph", build_graph)
        model, calls = scripted_model(answer=lambda _: {"summary": ""})
        existing = Community(
            community_id=community_id(["a", "b", "c"]),
            member_ids=["a", "b", "c"],
            summary="Last good summary.",
            description="Last good description.",
            content_hash="old-content",
        )
        store = _MemoryStore(existing)

        result = await CommunitiesStage().run(
            _context(scripted_role_models(consolidation=model), store)
        )

        stored = await store.get(Community, existing.community_id)
        assert result.status is StageStatus.PARTIAL
        assert result.watermark is None
        assert stored is not None
        assert (stored.summary, stored.description, stored.content_hash) == (
            "Last good summary.",
            "Last good description.",
            "",
        )
        assert (
            result.usage.buckets[f"communities/{ModelRole.CONSOLIDATION.value}"].calls
            == 3
        )
        assert len(calls.calls) == 3

    async def test_other_configured_role_does_not_enable_community_summaries(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An entity-extraction model does not configure community summaries."""
        graph = _graph()

        async def build_graph(_ctx: StageContext) -> EntityGraph:
            return graph

        monkeypatch.setattr(communities_module, "build_entity_graph", build_graph)
        model, calls = scripted_model(answer=_answer)
        store = _MemoryStore()

        result = await CommunitiesStage().run(
            _context(scripted_role_models(entity_extraction=model), store)
        )

        communities = await store.list(Community)
        assert result.status is StageStatus.PARTIAL
        assert result.watermark is None
        assert result.notes == [
            "no consolidation model configured: communities stored without summaries"
        ]
        assert len(communities) == 1 and communities[0].summary == ""
        assert calls.calls == []
        assert result.usage == Usage()
