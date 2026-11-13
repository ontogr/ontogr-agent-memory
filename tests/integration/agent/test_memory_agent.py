"""Integration tests for the default chat agent on the compose FalkorDB service."""

import json
import re
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from datetime import timedelta
from pathlib import Path

import pytest
from pydantic_ai.messages import UserPromptPart
from pydantic_ai.models import Model

from ontogr_mem.agent import MemoryCapability, build_default_agent
from ontogr_mem.client import Memory
from ontogr_mem.data_models import Episode, Kind, Role
from ontogr_mem.graphstore.falkordb import FalkorDBGraphStore
from ontogr_mem.llm import RoleModels
from ontogr_mem.relational_store.config import StateSettings
from ontogr_mem.service.bootstrap import ServiceSettings
from ontogr_mem.tenant_stores import TenantStores
from tests.support.embedder import MockEmbedder
from tests.support.falkordb import falkordb_test_settings
from tests.support.graph_backend import GraphBackend
from tests.support.scripted_model import (
    ScriptedCall,
    scripted_model,
    scripted_role_models,
)
from tests.support.tenants import new_tenant


_USER_SENTENCE = "I own Hydra."
_ASSISTANT_REPLY = "Noted."


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
async def _open_session(
    tenant: str, state_path: Path, models: RoleModels
) -> AsyncGenerator[Memory]:
    """Open a Memory client on FalkorDB with a durable SQLite state file.

    Args:
        tenant: Tenant the client writes to and reads from.
        state_path: SQLite file for the service state. Sessions opened on the
            same path read what the earlier sessions stored.
        models: Model pool for the client.
    """
    async with Memory.open(
        tenant=tenant,
        service_settings=ServiceSettings(backend="falkordb", _env_file=None),
        state_settings=StateSettings(
            backend="sqlite", sqlite_path=str(state_path), _env_file=None
        ),
        models=models,
        embedder=MockEmbedder(),
    ) as memory:
        yield memory


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


def _chat_extraction(call: ScriptedCall) -> object:
    """Return the scripted extraction for the episode that the prompt names.

    The user sentence yields the OWNS fact. The assistant reply is stored as an
    Episode without facts, so its extraction returns empty lists.
    """
    # Earlier episodes in the prompt can repeat the user sentence, so only the
    # episode block decides which answer to give.
    if _prompt_block(call.prompt, "episode_text") == _USER_SENTENCE:
        return _hydra_answer(call)
    if "Task: list the entities" in call.prompt:
        return {"entities": []}
    return {"facts": []}


def _chat_models(answer: Model) -> RoleModels:
    """Return a model pool with scripted extraction and the given answer model.

    Args:
        answer: Model that the default agent calls to produce its reply.
    """
    extraction, _ = scripted_model(answer=_chat_extraction)
    return scripted_role_models(extraction, answer=answer)


def _recall_answer(call: ScriptedCall) -> str:
    """Answer from the memory text that the capability injects into the request."""
    memory_text = "\n".join(
        part.content
        for part in call.messages[-1].parts
        if isinstance(part, UserPromptPart) and isinstance(part.content, str)
    )
    if "The speaker owns Hydra" in memory_text:
        return "You own Hydra."
    return "I do not know."


async def _assert_chat_episode(
    stores: TenantStores, episode: Episode, *, speaker: str, content: str
) -> None:
    """Assert that an episode is a chat message from one speaker in session chat-1."""
    assert episode.kind is Kind.MESSAGE
    assert episode.content == content
    assert episode.actor.display_name == speaker
    assert episode.occurred_at.utcoffset() == timedelta(0)
    meta = await stores.episodes.get_episode_meta(episode.id)
    if meta is None:
        raise AssertionError("episode has no metadata")
    assert meta.session_id == "chat-1"
    assert meta.thread == "chat-1"


class TestMemoryAgent:
    """The default chat agent stores each turn and recalls it in a later session."""

    async def test_chat_turn_is_stored_and_recalled_after_reopening(
        self, falkordb_store: FalkorDBGraphStore, tmp_path: Path
    ):
        """A chat turn is stored as two episodes and recalled after reopening."""
        tenant = new_tenant()
        state_path = tmp_path / "state.sqlite3"
        try:
            reply_model, _ = scripted_model(replies=[_ASSISTANT_REPLY], mode="prompted")
            async with _open_session(
                tenant, state_path, _chat_models(reply_model)
            ) as memory:
                agent = build_default_agent(memory)
                reply = await agent.run(_USER_SENTENCE, conversation_id="chat-1")
                await memory.drain()
                assert reply.output == _ASSISTANT_REPLY

                context = await memory.search("What does the speaker own?")
                assert context.state == "found"
                matches = [
                    fact for fact in context.facts if fact.object.name == "Hydra"
                ]
                assert [fact.statement for fact in matches] == [
                    "The speaker owns Hydra"
                ]
                fact = matches[0]

                # The client has no episode read, so the episode port of a
                # separate store object over the same tenant is used.
                stores = await GraphBackend(
                    store=falkordb_store, name="falkordb"
                ).stores_for(tenant)
                episodes = await stores.episodes.list_episodes(
                    limit=10, session_id="chat-1"
                )
                assert len(episodes) == 2
                episodes_by_role = {episode.actor.role: episode for episode in episodes}
                assert set(episodes_by_role) == {Role.USER, Role.ASSISTANT}
                user_episode = episodes_by_role[Role.USER]
                assistant_episode = episodes_by_role[Role.ASSISTANT]

                await _assert_chat_episode(
                    stores, user_episode, speaker="user", content=_USER_SENTENCE
                )
                await _assert_chat_episode(
                    stores,
                    assistant_episode,
                    speaker="assistant",
                    content=_ASSISTANT_REPLY,
                )

                # The fact cites only the user Episode, so the assistant reply
                # contributes no evidence to it.
                assert fact.episode_ids == [user_episode.id]

            recall_model, _ = scripted_model(answer=_recall_answer, mode="prompted")
            async with _open_session(
                tenant, state_path, _chat_models(recall_model)
            ) as memory:
                # A new conversation id with empty history shows that the answer
                # comes from stored memory and not from the chat transcript.
                agent = build_default_agent(
                    memory,
                    capability=MemoryCapability(memory, write=False, tools=()),
                )
                recall = await agent.run(
                    "What do I own?", conversation_id="chat-2", message_history=[]
                )
                assert recall.output == "You own Hydra."

            failing_model, _ = scripted_model(
                replies=[RuntimeError("answer model is unavailable")],
                mode="prompted",
            )
            async with _open_session(
                tenant, state_path, _chat_models(failing_model)
            ) as memory:
                agent = build_default_agent(memory)
                # A failed reply must leave the store with only the earlier pair.
                with pytest.raises(RuntimeError):
                    await agent.run("This turn fails.", conversation_id="chat-1")
                await memory.drain()

            episodes = await stores.episodes.list_episodes(limit=10)
            assert {episode.id for episode in episodes} == {
                user_episode.id,
                assistant_episode.id,
            }

            record = {
                "tenant": tenant,
                "session_id": "chat-1",
                "thread": "chat-1",
                "episodes": [
                    {
                        "episode_id": user_episode.id,
                        "role": "user",
                        "speaker": "user",
                        "content": _USER_SENTENCE,
                    },
                    {
                        "episode_id": assistant_episode.id,
                        "role": "assistant",
                        "speaker": "assistant",
                        "content": _ASSISTANT_REPLY,
                    },
                ],
                "recall_answer": recall.output,
            }
            artifact = tmp_path / "chat_memory.json"
            artifact.write_text(json.dumps(record, indent=2))
            assert json.loads(artifact.read_text()) == record
        finally:
            await falkordb_store.erase_tenant(tenant)
