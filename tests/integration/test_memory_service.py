"""Public memory service behavior over isolated in-memory adapters."""

import asyncio
import json
import re
from datetime import UTC, datetime

import pytest
from pydantic_ai.exceptions import ModelHTTPError

import ontogr_mem.service.writes as memory_writes
from ontogr_mem.data_models import Role
from ontogr_mem.llm import ModelRole, RoleModels
from ontogr_mem.ontology.diff import diff
from ontogr_mem.ontology.manager import OntologyManager
from ontogr_mem.ontology.models import Mode
from ontogr_mem.ontology.settings import OntologySettings
from ontogr_mem.service.contracts import (
    AddMemory,
    GetEntity,
    InvalidateMemory,
    ListEpisodes,
    MemoryKind,
    SearchMemory,
    UpdateMemory,
)
from ontogr_mem.service.errors import (
    LLMUnavailableError,
    ObjectNotFoundError,
)
from ontogr_mem.service.principal import Grant
from ontogr_mem.vectorstore import VectorWriter, write_episode_vectors
from tests.support.embedder import MockEmbedder
from tests.support.episodes import make_episode
from tests.support.graph_backend import GraphBackend
from tests.support.memory_service import memory_service_harness
from tests.support.scripted_model import (
    ScriptedCall,
    scripted_model,
    scripted_role_models,
)


def _prompt_block(prompt: str, tag: str) -> str:
    """Return one JSON prompt block's body."""
    match = re.search(rf"<{tag}>\n(.*?)\n</{tag}>", prompt, re.DOTALL)
    if match is None:
        raise AssertionError(f"missing {tag} block")
    return match.group(1)


def _memory_models() -> RoleModels:
    """Return a model pool that answers entity and fact prompts for memory writes."""

    def answer(call: ScriptedCall) -> object:
        prompt = call.prompt
        if "Task: list the entities" in prompt:
            text = _prompt_block(prompt, "episode_text")
            entity_types = json.loads(_prompt_block(prompt, "entity_types"))
            type_ids = {item["label"].casefold(): item["id"] for item in entity_types}
            names = [name for name in ("Ana", "Hydra") if name in text]
            return {
                "entities": [
                    {
                        "name": name,
                        "type_id": type_ids["thing" if name == "Hydra" else "person"],
                        "confidence": 1.0,
                    }
                    for name in names
                ]
            }

        text = _prompt_block(prompt, "episode_text")
        relation_types = json.loads(_prompt_block(prompt, "relation_types"))

        def relation_id(label: str) -> int | None:
            return next(
                (
                    item["id"]
                    for item in relation_types
                    if item["label"].casefold() == label.casefold()
                ),
                None,
            )

        def fact(
            statement: str,
            quote: str,
            *,
            label: str | None,
            refs: list[str],
            kind: str = "state",
        ) -> dict[str, object]:
            return {
                "text": statement,
                "entity_refs": refs,
                "relation_type_id": relation_id(label) if label else None,
                "relation_label": label,
                "fact_kind": kind,
                "polarity": "affirmed",
                "modality": "actual",
                "valid_from": None,
                "valid_to": None,
                "precision": None,
                "time_expression": None,
                "evidence_quote": quote,
                "confidence": 1.0,
            }

        if "violet comet" in text:
            facts = [
                fact(
                    "A violet comet note exists",
                    "A violet comet note exists.",
                    label=None,
                    refs=[],
                )
            ]
        elif "I direct Hydra" in text and "I acquired Hydra" in text:
            facts = [
                fact(
                    "The speaker owns Hydra",
                    "I own Hydra.",
                    label="OWNS",
                    refs=["actor", "e0"],
                ),
                fact(
                    "The speaker directs Hydra",
                    "I direct Hydra.",
                    label="DIRECTS",
                    refs=["actor", "e0"],
                ),
                fact(
                    "The speaker acquired Hydra",
                    "I acquired Hydra yesterday.",
                    label="OWNS",
                    refs=["actor", "e0"],
                    kind="event",
                ),
            ]
        elif "I direct Hydra" in text:
            facts = [
                fact(
                    "The speaker owns Hydra",
                    "I own Hydra.",
                    label="OWNS",
                    refs=["actor", "e0"],
                ),
                fact(
                    "The speaker directs Hydra",
                    "I direct Hydra.",
                    label="DIRECTS",
                    refs=["actor", "e0"],
                ),
            ]
        elif "I acquired Hydra" in text:
            facts = [
                fact(
                    "The speaker owns Hydra",
                    "I own Hydra.",
                    label="OWNS",
                    refs=["actor", "e0"],
                ),
                fact(
                    "The speaker acquired Hydra",
                    "I acquired Hydra yesterday.",
                    label="ACQUIRED",
                    refs=["actor", "e0"],
                    kind="event",
                ),
                fact(
                    "A private note exists", "A note was private.", label=None, refs=[]
                ),
            ]
        else:
            facts = [
                fact(
                    "The speaker owns Hydra",
                    "I own Hydra.",
                    label="OWNS",
                    refs=["actor", "e0"],
                )
            ]
        return {"facts": facts}

    model, _ = scripted_model(answer=answer)
    return scripted_role_models(model)


class TestMemoryService:
    """Exercise verbs through the service boundary."""

    async def test_status_lists_grants_and_reads_hide_denied_tenants(
        self, graph_backend: GraphBackend
    ):
        """Granted Tenants are listed and denied reads use the not-found error."""
        harness = await memory_service_harness(
            graph_backend,
        )

        status = await harness.service.get_status(harness.principal)

        assert "t1" in status.tenants
        operations = await harness.operations.list_operations(tenant=None)
        assert [(row.kind.value, row.outcome.value) for row in operations] == [
            ("retrieve", "ok")
        ]
        with pytest.raises(ObjectNotFoundError) as denied:
            await harness.service.list_episodes(
                harness.principal.model_copy(update={"default_tenant": "t2"}),
                ListEpisodes(),
            )
        assert str(denied.value) == "tenant or object not found"
        granted = harness.principal.model_copy(
            update={"grants": {"t2": harness.principal.grants["t1"]}}
        )
        empty = await harness.service.list_episodes(granted, ListEpisodes(tenant="t2"))
        assert empty.episodes == []

    async def test_status_includes_default_tenant_for_wildcard_grant(
        self, graph_backend: GraphBackend
    ):
        """A wildcard grant exposes the principal's default tenant in status."""
        harness = await memory_service_harness(
            graph_backend,
        )
        principal = harness.principal.model_copy(
            update={"grants": {"*": frozenset({Grant.READ})}}
        )

        status = await harness.service.get_status(principal)

        assert status.tenants == [principal.default_tenant]

    async def test_structured_writes_are_readable_through_entity_and_episode_verbs(
        self, graph_backend: GraphBackend
    ):
        """A fact write stores its relation statement and its session metadata."""
        harness = await memory_service_harness(
            graph_backend,
        )
        effective_at = datetime(2024, 3, 1, tzinfo=UTC)

        receipt = await harness.service.add_memory(
            harness.principal,
            AddMemory(
                tenant="t1",
                kind=MemoryKind.FACT,
                subject="Ana",
                relation="works_at",
                object="Acme",
                valid_from=effective_at,
                session_id="session-one",
                wait_seconds=5,
                idempotency_key="fact-one",
            ),
        )
        job = await harness.service.get_job(harness.principal, receipt.job_id)

        assert job.status.value == "done"
        assert job.result is not None and len(job.result["fact_ids"]) == 1
        entity = await harness.service.get_entity(
            harness.principal, GetEntity(tenant="t1", name="ANA")
        )
        assert entity.entity.name == "Ana"
        assert entity.facts[0].statement == "Ana works at Acme"
        episodes = await harness.service.list_episodes(
            harness.principal, ListEpisodes(tenant="t1", session_id="session-one")
        )
        assert [episode.session_id for episode in episodes.episodes] == ["session-one"]

    async def test_search_returns_fact_and_rendered_context(
        self, graph_backend: GraphBackend
    ):
        """A stored fact is available in the structured and prompt response."""
        harness = await memory_service_harness(
            graph_backend,
        )
        await harness.service.add_memory(
            harness.principal,
            AddMemory(
                tenant="t1",
                kind=MemoryKind.FACT,
                subject="Ana",
                relation="works_at",
                object="Acme",
                wait_seconds=5,
            ),
        )

        result = await harness.service.search_memory(
            harness.principal,
            SearchMemory(tenant="t1", query="Ana works at Acme", format="both"),
        )

        assert result.state == "found"
        assert result.facts and result.facts[0].statement == "Ana works at Acme"
        assert result.rendered

    async def test_session_search_ignores_episode_hits_from_other_sessions(
        self, graph_backend: GraphBackend
    ):
        """An episode-only hit outside the requested session is not found."""
        stores = await graph_backend.stores_for("t1", MockEmbedder())
        episode = make_episode(
            "session-search",
            content="A violet comet note exists.",
            metadata={"session_id": "stored-session"},
        )
        await stores.episodes.write_episodes([episode])
        await write_episode_vectors(VectorWriter(stores.vectors), [episode])
        harness = await memory_service_harness(graph_backend)

        unscoped = await harness.service.search_memory(
            harness.principal,
            SearchMemory(tenant="t1", query="violet comet note"),
        )
        scoped = await harness.service.search_memory(
            harness.principal,
            SearchMemory(
                tenant="t1",
                query="violet comet note",
                session_id="requested-session",
            ),
        )

        assert unscoped.state == "found"
        assert unscoped.facts == []
        assert scoped.facts == []
        assert scoped.entities == []
        assert scoped.state == "none"
        assert scoped.rendered is None

    async def test_text_write_without_a_model_fails_before_queueing(
        self, graph_backend: GraphBackend
    ):
        """A free-text request does not create a job without any language model."""
        harness = await memory_service_harness(
            graph_backend,
        )

        with pytest.raises(LLMUnavailableError):
            await harness.service.add_memory(
                harness.principal,
                AddMemory(tenant="t1", kind=MemoryKind.TEXT, content="Ana works here."),
            )

        assert await harness.state.tenant_ids() == []

    @pytest.mark.parametrize(
        "configured_role",
        [ModelRole.ENTITY_EXTRACTION, ModelRole.FACT_EXTRACTION],
    )
    async def test_text_write_with_one_extraction_role_fails_the_job(
        self, graph_backend: GraphBackend, configured_role: ModelRole
    ):
        """A text write needs both extraction roles, not just one of them."""
        model, _ = scripted_model()
        harness = await memory_service_harness(
            graph_backend,
            models=scripted_role_models(**{configured_role.value: model}),
        )

        receipt = await harness.service.add_memory(
            harness.principal,
            AddMemory(
                tenant="t1",
                kind=MemoryKind.TEXT,
                content="Ana works here.",
                wait_seconds=5,
            ),
        )
        job = await harness.service.get_job(harness.principal, receipt.job_id)

        assert job.status.value == "failed"
        assert job.error is not None
        assert job.error.code.value == "llm_unavailable"

    async def test_text_write_with_both_extraction_roles_completes(
        self, graph_backend: GraphBackend
    ):
        """Configuring both extraction roles lets a text write finish."""
        harness = await memory_service_harness(
            graph_backend,
            models=_memory_models(),
        )

        receipt = await harness.service.add_memory(
            harness.principal,
            AddMemory(
                tenant="t1",
                kind=MemoryKind.TEXT,
                content="A violet comet note exists.",
                wait_seconds=5,
            ),
        )
        job = await harness.service.get_job(harness.principal, receipt.job_id)

        assert job.status.value == "done"

    async def test_text_write_fails_after_output_validation_retries_without_job_retry(
        self, graph_backend: GraphBackend
    ):
        """Malformed extraction output ends the job after the built-in re-asks only."""
        model, scripted = scripted_model(answer=lambda call: "not json")
        harness = await memory_service_harness(
            graph_backend,
            models=scripted_role_models(model),
        )

        receipt = await harness.service.add_memory(
            harness.principal,
            AddMemory(
                tenant="t1",
                kind=MemoryKind.TEXT,
                content="Ana works here.",
                wait_seconds=5,
            ),
        )
        job = await harness.service.get_job(harness.principal, receipt.job_id)

        assert job.status.value == "failed"
        assert job.error is not None
        assert job.error.code.value == "llm_unavailable"
        assert len(scripted.calls) == 3
        await asyncio.sleep(0.2)
        assert len(scripted.calls) == 3

    async def test_text_write_with_provider_failure_calls_the_model_once(
        self, graph_backend: GraphBackend
    ):
        """A provider HTTP failure fails the job after a single model call."""
        model, scripted = scripted_model(
            [ModelHTTPError(503, "m", {"error": "unavailable"})]
        )
        harness = await memory_service_harness(
            graph_backend,
            models=scripted_role_models(model),
        )

        receipt = await harness.service.add_memory(
            harness.principal,
            AddMemory(
                tenant="t1",
                kind=MemoryKind.TEXT,
                content="Ana works here.",
                wait_seconds=5,
            ),
        )
        job = await harness.service.get_job(harness.principal, receipt.job_id)

        assert job.status.value == "failed"
        assert job.error is not None
        assert job.error.code.value == "llm_unavailable"
        assert len(scripted.calls) == 1
        await asyncio.sleep(0.2)
        assert len(scripted.calls) == 1

    async def test_update_closes_the_current_fact_and_adds_its_replacement(
        self, graph_backend: GraphBackend
    ):
        """An update closes one fact and makes its replacement current."""
        harness = await memory_service_harness(
            graph_backend,
        )
        original = await harness.service.add_memory(
            harness.principal,
            AddMemory(
                tenant="t1",
                kind=MemoryKind.FACT,
                subject="Ana",
                relation="works_at",
                object="Acme",
                valid_from=datetime(2024, 3, 1, tzinfo=UTC),
                wait_seconds=5,
            ),
        )
        original_job = await harness.service.get_job(harness.principal, original.job_id)
        assert original_job.result is not None
        fact_id = original_job.result["fact_ids"][0]

        receipt = await harness.service.update_memory(
            harness.principal,
            UpdateMemory(
                tenant="t1",
                fact_id=fact_id,
                object="Globex",
                effective_at=datetime(2025, 6, 1, tzinfo=UTC),
                wait_seconds=5,
            ),
        )

        job = await harness.service.get_job(harness.principal, receipt.job_id)
        entity = await harness.service.get_entity(
            harness.principal, GetEntity(tenant="t1", name="Ana")
        )
        assert job.status.value == "done"
        if job.result is None:
            raise AssertionError("completed update has no result")
        assert job.result["closed_fact_id"] == fact_id
        assert entity.facts[0].object.name == "Globex"

    async def test_invalidate_closes_a_fact_and_keeps_its_history(
        self, graph_backend: GraphBackend
    ):
        """An invalidated fact remains readable before its applied end time."""
        harness = await memory_service_harness(
            graph_backend,
        )
        forget_principal = harness.principal.model_copy(
            update={"grants": {"t1": harness.principal.grants["t1"] | {Grant.FORGET}}}
        )
        receipt = await harness.service.add_memory(
            harness.principal,
            AddMemory(
                tenant="t1",
                kind=MemoryKind.FACT,
                subject="Ana",
                relation="works_at",
                object="Initech",
                valid_from=datetime(2024, 3, 1, tzinfo=UTC),
                wait_seconds=5,
            ),
        )
        job = await harness.service.get_job(harness.principal, receipt.job_id)
        assert job.result is not None
        fact_id = job.result["fact_ids"][0]

        invalidated = await harness.service.invalidate(
            forget_principal,
            InvalidateMemory(
                tenant="t1",
                fact_id=fact_id,
                at=datetime(2025, 9, 1, tzinfo=UTC),
                wait_seconds=5,
            ),
        )
        historical = await harness.service.search_memory(
            harness.principal,
            SearchMemory(
                tenant="t1",
                query="Ana works at Initech",
                valid_at=datetime(2025, 8, 1, tzinfo=UTC),
            ),
        )
        current = await harness.service.search_memory(
            harness.principal,
            SearchMemory(tenant="t1", query="Ana works at Initech"),
        )
        history = await harness.service.search_memory(
            harness.principal,
            SearchMemory(
                tenant="t1",
                query="Ana works at Initech",
                include_history=True,
            ),
        )

        assert invalidated.status.value == "done"
        assert historical.facts and historical.facts[0].fact_id == fact_id
        assert current.facts == []
        assert any(
            fact.valid_to == datetime(2025, 9, 1, tzinfo=UTC) for fact in history.facts
        )

    async def test_update_refuses_a_closed_fact(self, graph_backend: GraphBackend):
        """An already closed fact cannot be replaced a second time."""
        harness = await memory_service_harness(
            graph_backend,
        )
        receipt = await harness.service.add_memory(
            harness.principal,
            AddMemory(
                tenant="t1",
                kind=MemoryKind.FACT,
                subject="Ana",
                relation="works_at",
                object="Acme",
                valid_from=datetime(2024, 3, 1, tzinfo=UTC),
                valid_to=datetime(2025, 1, 1, tzinfo=UTC),
                wait_seconds=5,
            ),
        )
        job = await harness.service.get_job(harness.principal, receipt.job_id)
        assert job.result is not None
        fact_id = job.result["fact_ids"][0]

        receipt = await harness.service.update_memory(
            harness.principal,
            UpdateMemory(
                tenant="t1",
                fact_id=fact_id,
                object="Globex",
                effective_at=datetime(2025, 6, 1, tzinfo=UTC),
                wait_seconds=5,
            ),
        )
        assert receipt.status.value == "failed"
        assert receipt.error is not None
        assert receipt.error.code.value == "conflict"
        assert receipt.error.message == "fact is closed, not active"

    async def test_episode_pages_do_not_repeat_episodes(
        self, graph_backend: GraphBackend
    ):
        """A next cursor starts after the last episode on the preceding page."""
        harness = await memory_service_harness(
            graph_backend,
        )
        for object_name in ("Acme", "Globex", "Initech"):
            await harness.service.add_memory(
                harness.principal,
                AddMemory(
                    tenant="t1",
                    kind=MemoryKind.FACT,
                    subject="Ana",
                    relation="works_at",
                    object=object_name,
                    wait_seconds=5,
                ),
            )

        first = await harness.service.list_episodes(
            harness.principal, ListEpisodes(tenant="t1", limit=2)
        )
        second = await harness.service.list_episodes(
            harness.principal,
            ListEpisodes(tenant="t1", limit=2, cursor=first.next_cursor),
        )

        first_ids = {episode.episode_id for episode in first.episodes}
        second_ids = {episode.episode_id for episode in second.episodes}
        assert len(first.episodes) == 2
        assert len(second.episodes) == 1
        assert first_ids.isdisjoint(second_ids)
        assert second.next_cursor is None

    async def test_update_rejects_an_effective_time_before_fact_start(
        self, graph_backend: GraphBackend
    ):
        """The update contract rejects an end that precedes the fact start."""
        harness = await memory_service_harness(
            graph_backend,
        )
        receipt = await harness.service.add_memory(
            harness.principal,
            AddMemory(
                tenant="t1",
                kind=MemoryKind.FACT,
                subject="Ana",
                relation="works_at",
                object="Acme",
                valid_from=datetime(2024, 3, 1, tzinfo=UTC),
                wait_seconds=5,
            ),
        )
        job = await harness.service.get_job(harness.principal, receipt.job_id)
        assert job.result is not None
        fact_id = job.result["fact_ids"][0]

        receipt = await harness.service.update_memory(
            harness.principal,
            UpdateMemory(
                tenant="t1",
                fact_id=fact_id,
                object="Globex",
                effective_at=datetime(2024, 2, 1, tzinfo=UTC),
                wait_seconds=5,
            ),
        )

        job = await harness.service.get_job(harness.principal, receipt.job_id)
        assert job.status.value == "failed"
        assert job.error is not None
        assert job.error.code.value == "invalid_argument"
        episodes = await harness.service.list_episodes(
            harness.principal, ListEpisodes(tenant="t1", limit=10)
        )
        assert len(episodes.episodes) == 1

    async def test_text_write_reports_skips_after_retry_and_keeps_episode_text(
        self, graph_backend: GraphBackend, monkeypatch: pytest.MonkeyPatch
    ):
        """A retry keeps item outcomes, facts, episode metadata and actor data."""
        harness = await memory_service_harness(graph_backend, models=_memory_models())
        await harness.service.add_memory(
            harness.principal,
            AddMemory(
                tenant="t1",
                kind=MemoryKind.FACT,
                subject="Ana",
                relation="owns",
                object="Hydra",
                speaker="agent<&>",
                role=Role.USER,
                wait_seconds=5,
            ),
        )
        stores = await graph_backend.stores_for("t1")
        manager = OntologyManager(
            stores.ontology_store, None, OntologySettings(), harness.clock
        )
        catalog = await manager.catalog()
        await manager.apply(
            diff(catalog, catalog.model_copy(update={"relation_mode": Mode.STRICT}))
        )
        original_write = memory_writes.write_episode_vectors
        vector_attempts = 0

        async def fail_first_vector_write(writer, episodes):
            nonlocal vector_attempts
            vector_attempts += 1
            if vector_attempts == 1:
                raise RuntimeError("temporary vector failure")
            await original_write(writer, episodes)

        monkeypatch.setattr(
            memory_writes, "write_episode_vectors", fail_first_vector_write
        )
        request = AddMemory(
            tenant="t1",
            kind=MemoryKind.MESSAGE,
            content="I own Hydra. I direct Hydra. I acquired Hydra yesterday.",
            speaker="agent<&>",
            role=Role.ASSISTANT,
            session_id="mixed-outcomes",
            idempotency_key="mixed-outcomes",
            wait_seconds=5,
        )

        receipt = await harness.service.add_memory(harness.principal, request)
        job = await harness.service.get_job(harness.principal, receipt.job_id)

        assert receipt.status.value == "done", job.error
        assert receipt.result == job.result
        assert job.result is not None
        assert job.result["ontology_refused"] == 1
        assert job.result["skipped"] == {"not_state": 1}
        assert job.attempts == 2
        assert vector_attempts == 2
        assert len(job.result["fact_ids"]) == 1
        entity = await harness.service.get_entity(
            harness.principal, GetEntity(tenant="t1", name="Hydra")
        )
        assert len(entity.facts) == 2
        assert any(fact.fact_id in job.result["fact_ids"] for fact in entity.facts)

        observations = await stores.temporal_store.observations_of_episode(
            job.result["episode_id"]
        )
        assert len(observations) == 1
        assert [
            (item.actor_role, item.source_system, item.connector)
            for item in observations
        ] == [("assistant", "agent-memory", "agent-interface")]
        episodes = await harness.service.list_episodes(
            harness.principal,
            ListEpisodes(tenant="t1", session_id="mixed-outcomes"),
        )
        assert [
            (item.actor_id, item.actor_name, item.actor_role)
            for item in episodes.episodes
        ] == [("agent<&>", "agent<&>", "assistant")]

        replay = await harness.service.add_memory(harness.principal, request)
        assert replay.idempotent_replay
        assert replay.result == job.result

        skipped_receipt = await harness.service.add_memory(
            harness.principal,
            AddMemory(
                tenant="t1",
                kind=MemoryKind.MESSAGE,
                content="A violet comet note exists.",
                speaker="agent<&>",
                role=Role.ASSISTANT,
                session_id="all-skipped",
                wait_seconds=5,
            ),
        )
        skipped_job = await harness.service.get_job(
            harness.principal, skipped_receipt.job_id
        )
        assert skipped_job.result is not None
        assert skipped_job.result["ontology_refused"] == 0
        assert skipped_job.result["skipped"] == {"no_relation": 1}
        assert skipped_job.result["fact_ids"] == []
        listed = await harness.service.list_episodes(
            harness.principal,
            ListEpisodes(tenant="t1", session_id="all-skipped"),
        )
        assert len(listed.episodes) == 1
        search = await harness.service.search_memory(
            harness.principal,
            SearchMemory(tenant="t1", query="violet comet note", format="both"),
        )
        assert search.state == "found"
        assert search.rendered is not None
        assert (
            'actor_id="agent&lt;&amp;&gt;" actor_name="agent&lt;&amp;&gt;" '
            'actor_role="assistant"'
        ) in search.rendered

    async def test_strict_refusal_keeps_known_and_rejects_structured_fact(
        self, graph_backend: GraphBackend
    ):
        """STRICT drops unknown text links but does not reject the episode."""
        harness = await memory_service_harness(graph_backend, models=_memory_models())
        first_receipt = await harness.service.add_memory(
            harness.principal,
            AddMemory(
                tenant="t1",
                kind=MemoryKind.FACT,
                subject="Ana",
                relation="owns",
                object="Hydra",
                speaker="speaker-1",
                role=Role.USER,
                wait_seconds=5,
            ),
        )
        stores = await graph_backend.stores_for("t1")
        manager = OntologyManager(
            stores.ontology_store, None, OntologySettings(), harness.clock
        )
        catalog = await manager.catalog()
        await manager.apply(
            diff(catalog, catalog.model_copy(update={"relation_mode": Mode.STRICT}))
        )

        receipt = await harness.service.add_memory(
            harness.principal,
            AddMemory(
                tenant="t1",
                kind=MemoryKind.MESSAGE,
                content="I own Hydra. I direct Hydra.",
                speaker="speaker-1",
                role=Role.ASSISTANT,
                session_id="strict-relations",
                wait_seconds=5,
            ),
        )
        assert receipt.status.value == "done"
        assert receipt.result is not None
        assert receipt.result["ontology_refused"] == 1
        assert receipt.result["skipped"] == {}
        assert len(receipt.result["fact_ids"]) == 1
        assert first_receipt.result is not None
        catalog = await manager.catalog()
        assert catalog.relation_by_label("DIRECTS") is None
        episodes = await harness.service.list_episodes(
            harness.principal,
            ListEpisodes(tenant="t1", limit=10),
        )
        actor_values = {
            episode.episode_id: (
                episode.actor_id,
                episode.actor_name,
                episode.actor_role,
            )
            for episode in episodes.episodes
        }
        assert actor_values == {
            first_receipt.result["episode_id"]: ("speaker-1", "speaker-1", "user"),
            receipt.result["episode_id"]: ("speaker-1", "speaker-1", "assistant"),
        }

        rejected = await harness.service.add_memory(
            harness.principal,
            AddMemory(
                tenant="t1",
                kind=MemoryKind.FACT,
                subject="Ana",
                relation="directs",
                object="Hydra",
                wait_seconds=5,
            ),
        )
        assert rejected.status.value == "failed"
        assert rejected.result is None
