"""Integration tests for the Memory client on the compose FalkorDB service.

Each test opens a client on an isolated tenant with in-memory state and scripted
models, and erases the tenant from FalkorDB afterwards. The tests fail when the
service is unreachable at MEMORY_FALKORDB_HOST and MEMORY_FALKORDB_PORT.
"""

import asyncio
import hashlib
import json
import random
import re
from collections.abc import AsyncGenerator
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from pydantic_ai.exceptions import ModelHTTPError
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    ToolCallPart,
    UserPromptPart,
)
from pydantic_ai.models.concurrency import ConcurrencyLimitedModel
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.profiles import ModelProfile

from ontogr_mem.client import Memory
from ontogr_mem.data_models import Episode, EpisodeMeta, Kind, Role
from ontogr_mem.data_models.jobs import JobState
from ontogr_mem.graphstore.falkordb import FalkorDBGraphStore
from ontogr_mem.llm import LLMConfig, ModelRole, RoleModels, build_models
from ontogr_mem.relational_store.config import StateSettings
from ontogr_mem.relational_store.jobs.errors import EphemeralStateError
from ontogr_mem.service.bootstrap import ServiceSettings
from ontogr_mem.service.contracts import ForgetMode, HostFact
from ontogr_mem.service.errors import (
    ConflictError,
    InvalidArgumentError,
    LLMUnavailableError,
    ObjectNotFoundError,
    PayloadTooLargeError,
)
from ontogr_mem.service.limits import MAX_CONTENT_BYTES
from tests.support.embedder import MockEmbedder
from tests.support.falkordb import falkordb_test_settings
from tests.support.graph_backend import GraphBackend
from tests.support.scripted_model import (
    ScriptedCall,
    scripted_model,
    scripted_role_models,
)
from tests.support.tenants import new_tenant


@pytest.fixture(autouse=True)
async def falkordb_environment(
    monkeypatch: pytest.MonkeyPatch, falkordb_store: FalkorDBGraphStore
) -> AsyncGenerator[None]:
    """Point the service settings at the compose FalkorDB for each test."""
    settings = falkordb_test_settings()
    monkeypatch.setenv("MEMORY_FALKORDB_HOST", settings.host)
    monkeypatch.setenv("MEMORY_FALKORDB_PORT", str(settings.port))
    yield


@asynccontextmanager
async def open_memory(
    tenant: str, models: RoleModels | None = None
) -> AsyncGenerator[Memory]:
    """Open a Memory client on FalkorDB with ephemeral state.

    Args:
        tenant: Tenant the client writes to and reads from.
        models: Model pool for the client. Defaults to scripted models that
            answer no calls, which is enough for structured writes.
    """
    async with Memory.open(
        tenant=tenant,
        service_settings=ServiceSettings(backend="falkordb", _env_file=None),
        state_settings=StateSettings(
            backend="memory", allow_ephemeral=True, _env_file=None
        ),
        models=scripted_role_models() if models is None else models,
        embedder=MockEmbedder(),
    ) as memory:
        yield memory


DIMENSIONS = 8


def _seeded_vector(text: str) -> list[float]:
    """Return a vector of DIMENSIONS components seeded by the text."""
    seed = int.from_bytes(hashlib.sha256(text.encode()).digest(), "big")
    rng = random.Random(seed)
    # Positive components keep every vector non-zero, which the embedder rejects.
    return [rng.uniform(0.1, 1.0) for _ in range(DIMENSIONS)]


class _FakeEmbeddings:
    """Answers embeddings.create with one seeded row per input text."""

    async def create(self, **request: Any) -> SimpleNamespace:
        """Return rows in input order for the request texts."""
        rows = [
            SimpleNamespace(index=index, embedding=_seeded_vector(text))
            for index, text in enumerate(request["input"])
        ]
        return SimpleNamespace(data=rows)


class FakeEmbeddingClient:
    """Stands in for AsyncOpenAI and counts close calls."""

    def __init__(self) -> None:
        """Create the client with no closes recorded."""
        self.embeddings = _FakeEmbeddings()
        self.close_count = 0

    async def close(self) -> None:
        """Count one close call."""
        self.close_count += 1


@pytest.fixture
def fake_embedding_client(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> FakeEmbeddingClient:
    """Route the default embedder to a fake endpoint in an empty directory.

    The empty working directory keeps any project `.env` from supplying
    embedding settings.
    """
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("MEMORY_EMBEDDING_MODEL", "test-model")
    monkeypatch.setenv("MEMORY_EMBEDDING_BASE_URL", "http://embeddings.invalid/v1")
    monkeypatch.setenv("MEMORY_EMBEDDING_DIMENSIONS", str(DIMENSIONS))
    client = FakeEmbeddingClient()
    monkeypatch.setattr(
        "ontogr_mem.vectorstore.builtin.endpoint.AsyncOpenAI", lambda **kwargs: client
    )
    return client


def open_default_memory(
    tenant: str, state_settings: StateSettings | None = None
) -> AbstractAsyncContextManager[Memory]:
    """Open a Memory client that selects its own embedder and owns its models.

    Args:
        tenant: Tenant the client writes to and reads from.
        state_settings: State settings. Defaults to ephemeral in-memory state.
    """
    if state_settings is None:
        state_settings = StateSettings(
            backend="memory", allow_ephemeral=True, _env_file=None
        )
    return Memory.open(
        tenant=tenant,
        service_settings=ServiceSettings(backend="falkordb", _env_file=None),
        state_settings=state_settings,
        llm_config=LLMConfig.model_construct(),
    )


def _judge_model(models: RoleModels) -> Any:
    """Return the provider model behind the judge role."""
    wrapped = models.model_for(ModelRole.JUDGE)
    assert isinstance(wrapped, ConcurrencyLimitedModel)
    return wrapped.wrapped


def _prompt_block(prompt: str, tag: str) -> str:
    """Return one JSON prompt block's body."""
    match = re.search(rf"<{tag}>\n(.*?)\n</{tag}>", prompt, re.DOTALL)
    if match is None:
        raise AssertionError(f"missing {tag} block")
    return match.group(1)


def _hydra_answer(call: ScriptedCall) -> object:
    """Return the scripted extraction answer for the "I own Hydra." sentence.

    OWNS is part of the default ontology, so text extraction can attach it
    without declaring a relation first.
    """
    prompt = call.prompt
    if "Task: list the entities" in prompt:
        entity_types = json.loads(_prompt_block(prompt, "entity_types"))
        type_id = next(
            item["id"] for item in entity_types if item["label"].casefold() == "thing"
        )
        return {"entities": [{"name": "Hydra", "type_id": type_id, "confidence": 1.0}]}

    relation_types = json.loads(_prompt_block(prompt, "relation_types"))
    relation_id = next(
        (item["id"] for item in relation_types if item["label"].casefold() == "owns"),
        None,
    )
    return {
        "facts": [
            {
                "text": "The speaker owns Hydra",
                "entity_refs": ["actor", "e0"],
                "relation_type_id": relation_id,
                "relation_label": "OWNS",
                "fact_kind": "state",
                "polarity": "affirmed",
                "modality": "actual",
                "valid_from": None,
                "valid_to": None,
                "precision": None,
                "time_expression": None,
                "evidence_quote": "I own Hydra.",
                "confidence": 1.0,
            }
        ]
    }


def _own_hydra_models() -> RoleModels:
    """Return a model pool that extracts one owning fact from one sentence."""
    model, _ = scripted_model(answer=_hydra_answer)
    return scripted_role_models(model)


def _held_hydra_models(release: asyncio.Event) -> RoleModels:
    """Return a model pool that answers only after ``release`` is set.

    Holding the reply keeps the write job queued or running, so a test can
    read the search state while indexing is still in progress.
    """

    async def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        prompts = [
            part.content
            for message in messages
            if isinstance(message, ModelRequest)
            for part in message.parts
            if isinstance(part, UserPromptPart)
            if isinstance(part.content, str)
        ]
        call = ScriptedCall(
            info.instructions or "",
            prompts[-1] if prompts else "",
            deepcopy(messages),
            deepcopy(info),
        )
        await release.wait()
        encoded = json.dumps(_hydra_answer(call))
        return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, encoded)])

    profile = ModelProfile(
        default_structured_output_mode="tool",
        supports_json_schema_output=True,
        supports_json_object_output=True,
    )
    return scripted_role_models(FunctionModel(respond, profile=profile))


async def _stored_episode(
    falkordb_store: FalkorDBGraphStore, tenant: str, episode_id: str
) -> tuple[Episode, EpisodeMeta]:
    """Return a stored episode and its metadata for one tenant.

    The client has no episode read, so the episode port of the same FalkorDB
    store that the client writes to is used.

    Raises:
        AssertionError: If the episode or its metadata is missing.
    """
    stores = await GraphBackend(store=falkordb_store, name="falkordb").stores_for(
        tenant
    )
    episode = await stores.episodes.get_episode(episode_id)
    meta = await stores.episodes.get_episode_meta(episode_id)
    if episode is None or meta is None:
        raise AssertionError("text write stored no episode")
    return episode, meta


class TestMemoryClient:
    """The Memory client writes, recalls and corrects facts end to end."""

    async def test_structured_write_is_searchable_and_readable(
        self, falkordb_store: FalkorDBGraphStore
    ):
        """A structured fact is indexed, found by search and listed on its entity."""
        tenant = new_tenant()
        try:
            async with open_memory(tenant) as memory:
                receipt = await memory.write_fact(
                    "Ana",
                    "works_at",
                    "Acme",
                    statement="Ana works at Acme.",
                    valid_from=datetime(2024, 1, 1, tzinfo=UTC),
                    wait_seconds=10,
                )
                await memory.drain()

                job = await memory.get_job(receipt.job_id)
                assert job.status is JobState.DONE
                assert job.error is None

                context = await memory.search("Where does Ana work?")
                assert context.state == "found"
                matches = [
                    fact
                    for fact in context.facts
                    if fact.subject.name == "Ana" and fact.object.name == "Acme"
                ]
                assert [fact.statement for fact in matches] == ["Ana works at Acme."]

                view = await memory.get_entity("Ana")
                assert view.entity.name == "Ana"
                assert "Ana works at Acme." in [fact.statement for fact in view.facts]
        finally:
            await falkordb_store.erase_tenant(tenant)

    async def test_same_idempotency_key_replays_the_receipt(
        self, falkordb_store: FalkorDBGraphStore
    ):
        """A repeated key returns the first job, and a different payload conflicts."""
        tenant = new_tenant()
        try:
            async with open_memory(tenant) as memory:
                first = await memory.write_fact(
                    "Ana",
                    "works_at",
                    "Acme",
                    idempotency_key="hire-ana",
                    wait_seconds=10,
                )
                second = await memory.write_fact(
                    "Ana",
                    "works_at",
                    "Acme",
                    idempotency_key="hire-ana",
                    wait_seconds=10,
                )

                assert first.idempotent_replay is False
                assert second.job_id == first.job_id
                assert second.idempotent_replay is True

                with pytest.raises(ConflictError):
                    await memory.write_fact(
                        "Ana",
                        "works_at",
                        "Beta",
                        idempotency_key="hire-ana",
                    )
        finally:
            await falkordb_store.erase_tenant(tenant)

    async def test_invalidate_closes_a_fact_but_keeps_its_history(
        self, falkordb_store: FalkorDBGraphStore
    ):
        """An invalidated fact leaves current search and stays in history search."""
        tenant = new_tenant()
        try:
            async with open_memory(tenant) as memory:
                await memory.write_fact(
                    "Ana",
                    "works_at",
                    "Acme",
                    statement="Ana works at Acme.",
                    valid_from=datetime(2024, 1, 1, tzinfo=UTC),
                    wait_seconds=10,
                )
                await memory.drain()
                found = await memory.search("Where does Ana work?")
                fact_id = next(
                    fact.fact_id
                    for fact in found.facts
                    if fact.statement == "Ana works at Acme."
                )

                invalidation = await memory.invalidate(fact_id, reason="Left Acme.")
                await memory.drain()
                invalidated = await memory.get_job(invalidation.job_id)
                assert invalidated.status is JobState.DONE

                current = await memory.search("Where does Ana work?")
                assert fact_id not in [fact.fact_id for fact in current.facts]

                historical = await memory.search(
                    "Where does Ana work?", include_history=True
                )
                assert fact_id in [fact.fact_id for fact in historical.facts]
        finally:
            await falkordb_store.erase_tenant(tenant)

    async def test_search_as_of_uses_valid_time(
        self, falkordb_store: FalkorDBGraphStore
    ):
        """A search inside the valid window finds the fact, a later one does not."""
        tenant = new_tenant()
        try:
            async with open_memory(tenant) as memory:
                await memory.write_fact(
                    "Ana",
                    "works_at",
                    "Acme",
                    statement="Ana works at Acme.",
                    valid_from=datetime(2024, 1, 1, tzinfo=UTC),
                    valid_to=datetime(2024, 6, 1, tzinfo=UTC),
                    wait_seconds=10,
                )

                during = await memory.search(
                    "Where does Ana work?",
                    as_of=datetime(2024, 3, 1, tzinfo=UTC),
                )
                after = await memory.search(
                    "Where does Ana work?",
                    as_of=datetime(2025, 1, 1, tzinfo=UTC),
                )

                assert [fact.statement for fact in during.facts] == [
                    "Ana works at Acme."
                ]
                assert [fact.statement for fact in after.facts] == []
        finally:
            await falkordb_store.erase_tenant(tenant)

    async def test_job_artifact_is_written(
        self, falkordb_store: FalkorDBGraphStore, tmp_path: Path
    ):
        """The receipt, job and search results serialise to a JSON artifact."""
        tenant = new_tenant()
        try:
            async with open_memory(tenant) as memory:
                receipt = await memory.write_fact(
                    "Ana",
                    "works_at",
                    "Acme",
                    statement="Ana works at Acme.",
                    valid_from=datetime(2024, 1, 1, tzinfo=UTC),
                    wait_seconds=10,
                )
                await memory.drain()
                job = await memory.get_job(receipt.job_id)
                context = await memory.search("Where does Ana work?")

                artifact = tmp_path / "artifact.json"
                artifact.write_text(
                    json.dumps(
                        {
                            "receipt": receipt.model_dump(mode="json"),
                            "job": job.model_dump(mode="json"),
                            "search": context.model_dump(mode="json"),
                        },
                        indent=2,
                    )
                )

                text = artifact.read_text()
                assert "Ana" in text
                assert "Acme" in text
                assert '"status": "done"' in text
        finally:
            await falkordb_store.erase_tenant(tenant)

    async def test_text_write_with_a_role_stores_a_message_episode(
        self, falkordb_store: FalkorDBGraphStore
    ):
        """A text write with a role stores a message episode with its metadata."""
        tenant = new_tenant()
        reference_time = datetime(2024, 5, 1, 12, 0, tzinfo=UTC)
        try:
            async with open_memory(tenant, models=_own_hydra_models()) as memory:
                receipt = await memory.write_text(
                    "I own Hydra.",
                    role=Role.ASSISTANT,
                    speaker="helper",
                    session_id="s1",
                    thread="t1",
                    reference_time=reference_time,
                    wait_seconds=10,
                )
                await memory.drain()

                job = await memory.get_job(receipt.job_id)
                assert job.status is JobState.DONE
                assert job.error is None
                assert job.result is not None

                context = await memory.search("What does the speaker own?")
                matches = [
                    fact for fact in context.facts if fact.object.name == "Hydra"
                ]
                assert [fact.statement for fact in matches] == [
                    "The speaker owns Hydra"
                ]

                episode, meta = await _stored_episode(
                    falkordb_store, tenant, job.result["episode_id"]
                )
                assert episode.kind is Kind.MESSAGE
                assert episode.actor.role is Role.ASSISTANT
                assert episode.actor.display_name == "helper"
                assert episode.occurred_at == reference_time
                assert meta.session_id == "s1"
                assert meta.thread == "t1"
        finally:
            await falkordb_store.erase_tenant(tenant)

    async def test_text_write_without_a_role_stores_a_document_episode(
        self, falkordb_store: FalkorDBGraphStore
    ):
        """A text write without a role stores a document episode from the user."""
        tenant = new_tenant()
        try:
            async with open_memory(tenant, models=_own_hydra_models()) as memory:
                receipt = await memory.write_text(
                    "I own Hydra.",
                    speaker="helper",
                    session_id="s1",
                    thread="t1",
                    reference_time=datetime(2024, 5, 1, 12, 0, tzinfo=UTC),
                    wait_seconds=10,
                )
                await memory.drain()

                job = await memory.get_job(receipt.job_id)
                assert job.status is JobState.DONE
                assert job.result is not None

                episode, _ = await _stored_episode(
                    falkordb_store, tenant, job.result["episode_id"]
                )
                assert episode.kind is Kind.DOCUMENT
                assert episode.actor.role is Role.USER
        finally:
            await falkordb_store.erase_tenant(tenant)

    async def test_text_write_without_extraction_models_is_refused_but_facts_still_work(
        self, falkordb_store: FalkorDBGraphStore
    ):
        """Text writes are refused without extraction models, structured facts work."""
        tenant = new_tenant()
        try:
            models = build_models(LLMConfig.model_construct())
            async with open_memory(tenant, models=models) as memory:
                with pytest.raises(LLMUnavailableError):
                    await memory.write_text("I own Hydra.")

                stores = await GraphBackend(
                    store=falkordb_store, name="falkordb"
                ).stores_for(tenant)
                assert await stores.episodes.list_episodes(limit=10) == []

                receipt = await memory.write_fact(
                    "Ana",
                    "works_at",
                    "Acme",
                    wait_seconds=10,
                )
                job = await memory.get_job(receipt.job_id)
                assert job.status is JobState.DONE
                assert job.error is None
        finally:
            await falkordb_store.erase_tenant(tenant)

    async def test_a_write_in_progress_is_found_by_its_episode_before_facts_land(
        self, falkordb_store: FalkorDBGraphStore
    ):
        """An unscoped search finds the stored episode before its facts exist."""
        tenant = new_tenant()
        release = asyncio.Event()
        try:
            async with open_memory(
                tenant, models=_held_hydra_models(release)
            ) as memory:
                receipt = await memory.write_text("I own Hydra.")
                assert receipt.status in (JobState.QUEUED, JobState.RUNNING)

                # The write job runs in the background, so the episode may take a
                # moment to appear in search. The held model keeps the facts pending.
                for _ in range(100):
                    before_facts = await memory.search("Hydra")
                    if before_facts.state == "found":
                        break
                    await asyncio.sleep(0.05)
                assert before_facts.state == "found"
                assert before_facts.facts == []

                # Nothing matches this query and the write job is still pending,
                # so the search reports indexing rather than none.
                unrelated = await memory.search("zebra quartz")
                assert unrelated.state == "indexing"
                assert unrelated.facts == []
                assert unrelated.entities == []

                release.set()
                await memory.drain()

                job = await memory.get_job(receipt.job_id)
                assert job.status is JobState.DONE
                assert job.error is None

                found = await memory.search("Hydra")
                assert found.state == "found"
        finally:
            await falkordb_store.erase_tenant(tenant)

    async def test_a_failed_model_call_becomes_a_failed_job(
        self, falkordb_store: FalkorDBGraphStore
    ):
        """A model call that errors fails its write job and stores no facts."""
        tenant = new_tenant()
        failures = [
            ModelHTTPError(status_code=500, model_name="test-model") for _ in range(10)
        ]
        model, _ = scripted_model(replies=failures)
        try:
            async with open_memory(
                tenant, models=scripted_role_models(model)
            ) as memory:
                receipt = await memory.write_text("I own Hydra.", wait_seconds=10)
                assert receipt.status is JobState.FAILED
                assert receipt.error is not None

                await memory.drain()

                job = await memory.get_job(receipt.job_id)
                assert job.status is JobState.FAILED
                assert job.error is not None
                assert job.error.code

                context = await memory.search("Hydra")
                assert context.facts == []
                assert context.entities == []

                unrelated = await memory.search("zebra quartz")
                assert unrelated.state == "none"
                assert unrelated.facts == []
                assert unrelated.entities == []
        finally:
            await falkordb_store.erase_tenant(tenant)

    async def test_hard_forget_needs_a_preview_token_and_removes_the_episode(
        self, falkordb_store: FalkorDBGraphStore
    ):
        """A hard forget removes the episode only after its dry-run token is used."""
        tenant = new_tenant()
        try:
            async with open_memory(tenant, models=_own_hydra_models()) as memory:
                receipt = await memory.write_text("I own Hydra.", wait_seconds=10)
                await memory.drain()
                job = await memory.get_job(receipt.job_id)
                assert job.status is JobState.DONE
                assert job.result is not None
                episode_id = job.result["episode_id"]

                preview = await memory.forget(
                    episode_id=episode_id, mode=ForgetMode.DELETE, dry_run=True
                )
                assert preview.dry_run is True
                assert preview.confirm_token
                assert preview.episode_ids == [episode_id]

                result = await memory.forget(
                    episode_id=episode_id,
                    mode=ForgetMode.DELETE,
                    confirm_token=preview.confirm_token,
                )
                assert result.dry_run is False
                assert result.not_cleaned == []

                stores = await GraphBackend(
                    store=falkordb_store, name="falkordb"
                ).stores_for(tenant)
                assert await stores.episodes.get_episode(episode_id) is None
        finally:
            await falkordb_store.erase_tenant(tenant)

    async def test_soft_forget_queues_an_invalidation(
        self, falkordb_store: FalkorDBGraphStore
    ):
        """A default forget queues an invalidation that leaves current search."""
        tenant = new_tenant()
        try:
            async with open_memory(tenant) as memory:
                await memory.write_fact(
                    "Ana",
                    "works_at",
                    "Acme",
                    statement="Ana works at Acme.",
                    valid_from=datetime(2024, 1, 1, tzinfo=UTC),
                    wait_seconds=10,
                )
                await memory.drain()
                found = await memory.search("Where does Ana work?")
                fact_id = next(
                    fact.fact_id
                    for fact in found.facts
                    if fact.statement == "Ana works at Acme."
                )

                result = await memory.forget(fact_id=fact_id, reason="Left Acme.")
                assert result.dry_run is False
                assert result.receipt is not None
                assert result.receipt.error is None

                await memory.drain()
                job = await memory.get_job(result.receipt.job_id)
                assert job.status is JobState.DONE

                current = await memory.search("Where does Ana work?")
                assert fact_id not in [fact.fact_id for fact in current.facts]
        finally:
            await falkordb_store.erase_tenant(tenant)

    async def test_tenants_are_isolated(self, falkordb_store: FalkorDBGraphStore):
        """A fact written in one tenant is invisible to another tenant."""
        first_tenant = new_tenant()
        second_tenant = new_tenant()
        try:
            async with (
                open_memory(first_tenant) as first,
                open_memory(second_tenant) as second,
            ):
                receipt = await first.write_fact(
                    "Ana",
                    "works_at",
                    "Acme",
                    statement="Ana works at Acme.",
                    valid_from=datetime(2024, 1, 1, tzinfo=UTC),
                    wait_seconds=10,
                )
                await first.drain()

                with pytest.raises(ObjectNotFoundError):
                    await second.get_entity("Ana")
                with pytest.raises(ObjectNotFoundError):
                    await second.get_job(receipt.job_id)

                context = await second.search("Where does Ana work?")
                assert context.facts == []
                assert context.entities == []
        finally:
            await falkordb_store.erase_tenant(second_tenant)
            await falkordb_store.erase_tenant(first_tenant)

    @pytest.mark.parametrize(
        "write",
        [
            pytest.param(lambda memory: memory.search(""), id="empty_query"),
            pytest.param(
                lambda memory: memory.write_fact("", "works_at", "Acme"),
                id="empty_subject",
            ),
        ],
    )
    async def test_invalid_requests_raise_invalid_argument(
        self, falkordb_store: FalkorDBGraphStore, write
    ):
        """A request with a missing required field raises InvalidArgumentError."""
        tenant = new_tenant()
        try:
            async with open_memory(tenant) as memory:
                with pytest.raises(InvalidArgumentError):
                    await write(memory)
        finally:
            await falkordb_store.erase_tenant(tenant)

    async def test_oversized_content_raises_payload_too_large(
        self, falkordb_store: FalkorDBGraphStore
    ):
        """Text over the content byte limit raises PayloadTooLargeError."""
        tenant = new_tenant()
        try:
            async with open_memory(tenant) as memory:
                with pytest.raises(PayloadTooLargeError):
                    await memory.write_text("a" * (MAX_CONTENT_BYTES + 1))
        finally:
            await falkordb_store.erase_tenant(tenant)

    async def test_injected_models_and_embedder_stay_usable_after_close(
        self, falkordb_store: FalkorDBGraphStore
    ):
        """Caller-owned models and embedder stay open after the client closes."""
        tenant = new_tenant()

        class CountingEmbedder(MockEmbedder):
            """Counts aclose calls to show that the client leaves it open."""

            def __init__(self) -> None:
                """Create the embedder with no closes recorded."""
                super().__init__()
                self.close_count = 0

            async def aclose(self) -> None:
                """Count one close call."""
                self.close_count += 1

        models = build_models(
            LLMConfig(
                model="m",
                base_url="https://llm.invalid/v1",
                api_key="k",
                _env_file=None,
            )
        )
        embedder = CountingEmbedder()
        try:
            async with Memory.open(
                tenant=tenant,
                service_settings=ServiceSettings(backend="falkordb", _env_file=None),
                state_settings=StateSettings(
                    backend="memory", allow_ephemeral=True, _env_file=None
                ),
                models=models,
                embedder=embedder,
            ) as memory:
                receipt = await memory.write_fact(
                    "Ana", "works_at", "Acme", wait_seconds=10
                )
                job = await memory.get_job(receipt.job_id)
                assert job.status is JobState.DONE
                context = await memory.search("Where does Ana work?")
                assert context.state == "found"

            assert not _judge_model(models).client._client.is_closed
            assert embedder.close_count == 0
            assert models.model_for(ModelRole.ANSWER) is not None
            vectors = await embedder.embed_batch(["x"])
            assert len(vectors) == 1
            assert len(vectors[0]) == embedder.dim
        finally:
            await models.aclose()
            await falkordb_store.erase_tenant(tenant)

    async def test_body_exception_still_closes_the_service(
        self, falkordb_store: FalkorDBGraphStore
    ):
        """An exception in the body propagates and a later client opens normally."""
        tenant = new_tenant()

        class BodyFailureError(Exception):
            """Raised inside the client block to test context exit."""

        try:
            with pytest.raises(BodyFailureError, match="body failed"):
                async with open_memory(tenant) as memory:
                    await memory.write_fact("Ana", "works_at", "Acme", wait_seconds=10)
                    raise BodyFailureError("body failed")

            async with open_memory(tenant) as memory:
                receipt = await memory.write_fact(
                    "Ana", "works_at", "Beta", wait_seconds=10
                )
                job = await memory.get_job(receipt.job_id)
                assert job.status is JobState.DONE
                assert job.error is None
        finally:
            await falkordb_store.erase_tenant(tenant)

    async def test_default_open_selects_the_endpoint_embedder_and_closes_it(
        self,
        falkordb_store: FalkorDBGraphStore,
        fake_embedding_client: FakeEmbeddingClient,
    ):
        """Default embedder selection serves writes and closes the endpoint client."""
        tenant = new_tenant()
        try:
            async with open_default_memory(tenant) as memory:
                receipt = await memory.write_fact(
                    "Ana", "works_at", "Acme", wait_seconds=10
                )
                await memory.drain()
                job = await memory.get_job(receipt.job_id)
                assert job.status is JobState.DONE

                context = await memory.search("Where does Ana work?")
                assert context.state == "found"
                assert [
                    (fact.subject.name, fact.object.name) for fact in context.facts
                ] == [("Ana", "Acme")]

            assert fake_embedding_client.close_count == 1
        finally:
            await falkordb_store.erase_tenant(tenant)

    async def test_default_open_closes_owned_resources_after_a_body_exception(
        self,
        falkordb_store: FalkorDBGraphStore,
        fake_embedding_client: FakeEmbeddingClient,
    ):
        """An exception in the body propagates and the owned embedder still closes."""
        tenant = new_tenant()

        class DefaultBodyFailureError(Exception):
            """Raised inside the client block to test context exit."""

        try:
            with pytest.raises(DefaultBodyFailureError, match="body failed"):
                async with open_default_memory(tenant) as memory:
                    await memory.write_fact("Ana", "works_at", "Acme", wait_seconds=10)
                    raise DefaultBodyFailureError("body failed")

            assert fake_embedding_client.close_count == 1
        finally:
            await falkordb_store.erase_tenant(tenant)

    async def test_default_open_closes_the_embedder_when_startup_fails(
        self,
        falkordb_store: FalkorDBGraphStore,
        fake_embedding_client: FakeEmbeddingClient,
    ):
        """A service that fails to start raises and still closes the embedder."""
        tenant = new_tenant()
        try:
            with pytest.raises(EphemeralStateError):
                async with open_default_memory(
                    tenant,
                    state_settings=StateSettings(
                        backend="memory", allow_ephemeral=False, _env_file=None
                    ),
                ):
                    pass

            assert fake_embedding_client.close_count == 1
        finally:
            await falkordb_store.erase_tenant(tenant)


class TestWriteFacts:
    """Keyless fact writes store the source text and only grounded facts."""

    async def test_valid_fact_is_stored_and_searchable(
        self, falkordb_store: FalkorDBGraphStore
    ):
        """A grounded fact is stored and searchable, and its source is kept."""
        tenant = new_tenant()
        source = "Ana works at Acme."
        try:
            async with open_memory(tenant) as memory:
                receipt = await memory.write_facts(
                    source,
                    [
                        HostFact(
                            subject="Ana",
                            subject_type="Person",
                            relation="WORKS_AT",
                            object="Acme",
                            object_type="Organization",
                            evidence_quote="Ana works at Acme",
                        )
                    ],
                    role=Role.USER,
                )
                await memory.drain()

                job = await memory.get_job(receipt.job_id)
                assert job.status is JobState.DONE
                assert job.error is None
                assert job.result is not None
                assert len(job.result["fact_ids"]) == 1
                assert job.result["dropped"] == []

                context = await memory.search("Where does Ana work?")
                matches = [
                    fact
                    for fact in context.facts
                    if fact.subject.name == "Ana" and fact.object.name == "Acme"
                ]
                assert len(matches) == 1

                episode, _ = await _stored_episode(
                    falkordb_store, tenant, job.result["episode_id"]
                )
                assert episode.content == source
                assert episode.kind is Kind.DOCUMENT
                assert episode.actor.role is Role.USER
        finally:
            await falkordb_store.erase_tenant(tenant)

    async def test_source_text_is_kept_verbatim_not_rebuilt_from_the_fact(
        self, falkordb_store: FalkorDBGraphStore
    ):
        """The stored Episode holds the whole source text, not the fact alone."""
        tenant = new_tenant()
        source = (
            "Ana joined the data team in May. Ana works at Acme. "
            "She leads the analytics group."
        )
        try:
            async with open_memory(tenant) as memory:
                receipt = await memory.write_facts(
                    source,
                    [
                        HostFact(
                            subject="Ana",
                            subject_type="Person",
                            relation="WORKS_AT",
                            object="Acme",
                            object_type="Organization",
                            evidence_quote="Ana works at Acme",
                        )
                    ],
                )
                await memory.drain()

                job = await memory.get_job(receipt.job_id)
                assert job.status is JobState.DONE
                assert job.result is not None

                episode, _ = await _stored_episode(
                    falkordb_store, tenant, job.result["episode_id"]
                )
                assert episode.content == source
        finally:
            await falkordb_store.erase_tenant(tenant)

    async def test_mixed_validity_reports_original_indices(
        self, falkordb_store: FalkorDBGraphStore
    ):
        """Valid facts are stored and each dropped fact keeps its input index."""
        tenant = new_tenant()
        try:
            async with open_memory(tenant) as memory:
                receipt = await memory.write_facts(
                    "Ana works at Acme.",
                    [
                        HostFact(
                            subject="Ana",
                            subject_type="Person",
                            relation="WORKS_AT",
                            object="Acme",
                            object_type="Organization",
                            evidence_quote="Ana works at Acme",
                        ),
                        HostFact(
                            subject="Ana",
                            subject_type="Person",
                            relation="WORKS_AT",
                            object="Acme",
                            object_type="Organization",
                            evidence_quote="Ana runs Acme",
                        ),
                        HostFact(
                            subject="Ana",
                            subject_type="Person",
                            relation="WORKS_AT",
                            object="",
                            object_type="Organization",
                            evidence_quote="Ana works at Acme",
                        ),
                    ],
                )
                await memory.drain()

                job = await memory.get_job(receipt.job_id)
                assert job.status is JobState.DONE
                assert job.result is not None
                assert len(job.result["fact_ids"]) == 1
                assert job.result["dropped"] == [
                    {
                        "index": 1,
                        "reason": "ungrounded",
                        "detail": "quote not found in episode",
                    },
                    {"index": 2, "reason": "missing_field", "detail": "object"},
                ]
        finally:
            await falkordb_store.erase_tenant(tenant)

    async def test_all_invalid_facts_keep_the_source_episode(
        self, falkordb_store: FalkorDBGraphStore
    ):
        """When every fact is dropped, the source Episode is still stored."""
        tenant = new_tenant()
        source = "Ana works at Acme."
        try:
            async with open_memory(tenant) as memory:
                receipt = await memory.write_facts(
                    source,
                    [
                        HostFact(
                            subject="Ana",
                            subject_type="Person",
                            relation="WORKS_AT",
                            object="Acme",
                            object_type="Organization",
                            evidence_quote="Ana runs Acme",
                        ),
                        HostFact(
                            subject="Ana",
                            subject_type="Person",
                            relation="WORKS_AT",
                            object="",
                            object_type="Organization",
                            evidence_quote="Ana works at Acme",
                        ),
                    ],
                )
                await memory.drain()

                job = await memory.get_job(receipt.job_id)
                assert job.status is JobState.DONE
                assert job.result is not None
                assert job.result["fact_ids"] == []
                assert [
                    (drop["index"], drop["reason"]) for drop in job.result["dropped"]
                ] == [(0, "ungrounded"), (1, "missing_field")]

                episode, _ = await _stored_episode(
                    falkordb_store, tenant, job.result["episode_id"]
                )
                assert episode.content == source
        finally:
            await falkordb_store.erase_tenant(tenant)

    async def test_empty_facts_store_only_the_source_episode(
        self, falkordb_store: FalkorDBGraphStore
    ):
        """An empty fact list stores the source Episode and no facts."""
        tenant = new_tenant()
        source = "Ana works at Acme."
        try:
            async with open_memory(tenant) as memory:
                receipt = await memory.write_facts(source, [])
                await memory.drain()

                job = await memory.get_job(receipt.job_id)
                assert job.status is JobState.DONE
                assert job.result is not None
                assert job.result["fact_ids"] == []
                assert job.result["dropped"] == []

                episode, _ = await _stored_episode(
                    falkordb_store, tenant, job.result["episode_id"]
                )
                assert episode.content == source
        finally:
            await falkordb_store.erase_tenant(tenant)

    async def test_same_idempotency_key_replays_the_job_and_adds_no_second_mention(
        self, falkordb_store: FalkorDBGraphStore
    ):
        """A repeated key replays the first job and links Ana to one mention."""
        tenant = new_tenant()
        source = "Ana works at Acme."
        facts = [
            HostFact(
                subject="Ana",
                subject_type="Person",
                relation="WORKS_AT",
                object="Acme",
                object_type="Organization",
                evidence_quote="Ana works at Acme",
            )
        ]
        try:
            async with open_memory(tenant) as memory:
                first = await memory.write_facts(
                    source, facts, idempotency_key="ana-works-at-acme"
                )
                second = await memory.write_facts(
                    source, facts, idempotency_key="ana-works-at-acme"
                )
                assert first.idempotent_replay is False
                assert second.job_id == first.job_id
                assert second.idempotent_replay is True
                await memory.drain()

                stores = await GraphBackend(
                    store=falkordb_store, name="falkordb"
                ).stores_for(tenant)
                assert len(await stores.episodes.list_episodes(limit=10)) == 1
                entities = [
                    entity
                    for entity in await stores.resolver_store.active_entities()
                    if entity.canonical_name == "Ana"
                ]
                assert len(entities) == 1
                mentions = await stores.resolver_store.mentions_of(
                    entities[0].entity_id
                )
                assert len(mentions) == 1
        finally:
            await falkordb_store.erase_tenant(tenant)

    async def test_wrong_direction_is_dropped(self, falkordb_store: FalkorDBGraphStore):
        """A WORKS_AT fact with its endpoints swapped is dropped as wrong_direction."""
        tenant = new_tenant()
        try:
            async with open_memory(tenant) as memory:
                receipt = await memory.write_facts(
                    "Ana works at Acme.",
                    [
                        HostFact(
                            subject="Acme",
                            subject_type="Organization",
                            relation="WORKS_AT",
                            object="Ana",
                            object_type="Person",
                            evidence_quote="Ana works at Acme",
                        )
                    ],
                )
                await memory.drain()

                job = await memory.get_job(receipt.job_id)
                assert job.status is JobState.DONE
                assert job.result is not None
                assert job.result["fact_ids"] == []
                assert [
                    (drop["index"], drop["reason"]) for drop in job.result["dropped"]
                ] == [(0, "wrong_direction")]
        finally:
            await falkordb_store.erase_tenant(tenant)

    async def test_self_relation_is_dropped(self, falkordb_store: FalkorDBGraphStore):
        """A relation from an entity to itself is dropped as self_relation."""
        tenant = new_tenant()
        try:
            async with open_memory(tenant) as memory:
                receipt = await memory.write_facts(
                    "Ana knows herself.",
                    [
                        HostFact(
                            subject="Ana",
                            subject_type="Person",
                            relation="KNOWS",
                            object="Ana",
                            object_type="Person",
                            evidence_quote="Ana knows herself",
                        )
                    ],
                )
                await memory.drain()

                job = await memory.get_job(receipt.job_id)
                assert job.status is JobState.DONE
                assert job.result is not None
                assert job.result["fact_ids"] == []
                assert [
                    (drop["index"], drop["reason"]) for drop in job.result["dropped"]
                ] == [(0, "self_relation")]
        finally:
            await falkordb_store.erase_tenant(tenant)

    async def test_inverted_time_window_is_dropped(
        self, falkordb_store: FalkorDBGraphStore
    ):
        """A fact whose valid_from is after valid_to is dropped as time_order."""
        tenant = new_tenant()
        try:
            async with open_memory(tenant) as memory:
                receipt = await memory.write_facts(
                    "Ana works at Acme.",
                    [
                        HostFact(
                            subject="Ana",
                            subject_type="Person",
                            relation="WORKS_AT",
                            object="Acme",
                            object_type="Organization",
                            evidence_quote="Ana works at Acme",
                            valid_from=datetime(2024, 6, 1, tzinfo=UTC),
                            valid_to=datetime(2024, 1, 1, tzinfo=UTC),
                        )
                    ],
                )
                await memory.drain()

                job = await memory.get_job(receipt.job_id)
                assert job.status is JobState.DONE
                assert job.result is not None
                assert job.result["fact_ids"] == []
                assert [
                    (drop["index"], drop["reason"]) for drop in job.result["dropped"]
                ] == [(0, "time_order")]
        finally:
            await falkordb_store.erase_tenant(tenant)

    async def test_duplicate_facts_are_merged(self, falkordb_store: FalkorDBGraphStore):
        """A repeated fact in one write is stored once and reported as merged."""
        tenant = new_tenant()
        fact = HostFact(
            subject="Ana",
            subject_type="Person",
            relation="WORKS_AT",
            object="Acme",
            object_type="Organization",
            evidence_quote="Ana works at Acme",
        )
        try:
            async with open_memory(tenant) as memory:
                receipt = await memory.write_facts("Ana works at Acme.", [fact, fact])
                await memory.drain()

                job = await memory.get_job(receipt.job_id)
                assert job.status is JobState.DONE
                assert job.result is not None
                assert len(job.result["fact_ids"]) == 1
                assert [
                    (drop["index"], drop["reason"]) for drop in job.result["dropped"]
                ] == [(1, "duplicate_merged")]
        finally:
            await falkordb_store.erase_tenant(tenant)

    async def test_quote_from_another_episode_is_ungrounded(
        self, falkordb_store: FalkorDBGraphStore
    ):
        """A quote that appears only in an earlier episode does not ground a fact."""
        tenant = new_tenant()
        try:
            async with open_memory(tenant) as memory:
                first = await memory.write_facts(
                    "Ana works at Acme.",
                    [
                        HostFact(
                            subject="Ana",
                            subject_type="Person",
                            relation="WORKS_AT",
                            object="Acme",
                            object_type="Organization",
                            evidence_quote="Ana works at Acme",
                        )
                    ],
                )
                await memory.drain()
                first_job = await memory.get_job(first.job_id)
                assert first_job.result is not None
                assert len(first_job.result["fact_ids"]) == 1

                second = await memory.write_facts(
                    "The weather is mild.",
                    [
                        HostFact(
                            subject="Ana",
                            subject_type="Person",
                            relation="WORKS_AT",
                            object="Acme",
                            object_type="Organization",
                            evidence_quote="Ana works at Acme",
                        )
                    ],
                )
                await memory.drain()

                job = await memory.get_job(second.job_id)
                assert job.status is JobState.DONE
                assert job.result is not None
                assert job.result["fact_ids"] == []
                assert [
                    (drop["index"], drop["reason"]) for drop in job.result["dropped"]
                ] == [(0, "ungrounded")]
        finally:
            await falkordb_store.erase_tenant(tenant)

    async def test_unknown_relation_is_dropped_and_not_declared(
        self, falkordb_store: FalkorDBGraphStore
    ):
        """A relation outside the ontology is dropped and never stored."""
        tenant = new_tenant()
        try:
            async with open_memory(tenant) as memory:
                receipt = await memory.write_facts(
                    "Ana mentors Ben.",
                    [
                        HostFact(
                            subject="Ana",
                            subject_type="Person",
                            relation="MENTORS",
                            object="Ben",
                            object_type="Person",
                            evidence_quote="Ana mentors Ben",
                        )
                    ],
                )
                await memory.drain()

                job = await memory.get_job(receipt.job_id)
                assert job.status is JobState.DONE
                assert job.result is not None
                assert job.result["fact_ids"] == []
                assert [
                    (drop["index"], drop["reason"]) for drop in job.result["dropped"]
                ] == [(0, "unknown_relation")]

                context = await memory.search("mentor")
                assert context.facts == []
        finally:
            await falkordb_store.erase_tenant(tenant)

    async def test_unknown_entity_type_keeps_the_fact_with_an_unknown_endpoint(
        self, falkordb_store: FalkorDBGraphStore
    ):
        """A fact with an undeclared entity type is stored, not dropped."""
        tenant = new_tenant()
        try:
            async with open_memory(tenant) as memory:
                receipt = await memory.write_facts(
                    "Voyager is related to Acme.",
                    [
                        HostFact(
                            subject="Voyager",
                            subject_type="Spaceship",
                            relation="RELATED_TO",
                            object="Acme",
                            object_type="Organization",
                            evidence_quote="Voyager is related to Acme",
                        )
                    ],
                )
                await memory.drain()

                job = await memory.get_job(receipt.job_id)
                assert job.status is JobState.DONE
                assert job.result is not None
                assert len(job.result["fact_ids"]) == 1
                assert job.result["dropped"] == []
        finally:
            await falkordb_store.erase_tenant(tenant)

    async def test_keyless_write_never_calls_a_model(
        self, falkordb_store: FalkorDBGraphStore
    ):
        """A keyless fact write stores the fact without calling any model."""
        tenant = new_tenant()

        def refuse(call: ScriptedCall) -> object:
            """Fail the test if the write reaches a model."""
            raise AssertionError("keyless write called a model")

        model, recorded = scripted_model(answer=refuse)
        try:
            async with open_memory(
                tenant, models=scripted_role_models(model)
            ) as memory:
                receipt = await memory.write_facts(
                    "Ana works at Acme.",
                    [
                        HostFact(
                            subject="Ana",
                            subject_type="Person",
                            relation="WORKS_AT",
                            object="Acme",
                            object_type="Organization",
                            evidence_quote="Ana works at Acme",
                        )
                    ],
                )
                await memory.drain()

                job = await memory.get_job(receipt.job_id)
                assert job.status is JobState.DONE
                assert job.error is None
                assert recorded.calls == []
        finally:
            await falkordb_store.erase_tenant(tenant)
