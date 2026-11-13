"""Model-visible memory tools: their order, schemas, writes and argument errors."""

import re
from datetime import UTC, datetime

import pytest
from pydantic_ai import Agent
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.tools import ToolDefinition

from ontogr_mem.agent import MemoryCapability
from ontogr_mem.service.contracts import WriteReceipt
from ontogr_mem.service.errors import InvalidArgumentError, PayloadTooLargeError
from tests.support.fake_memory import FakeMemory, Invalidation, fact_context


NOTE = "Ana moved to Berlin."
REASON = "Ana no longer works at Acme."
ANSWER_RULE = "Answer only from the memory context."
HARD_DELETE_CONTROLS = {
    "mode",
    "confirm",
    "confirm_token",
    "confirmation_token",
    "episode_id",
    "hard_delete",
    "force",
}


@pytest.fixture(autouse=True)
def quiet_pydantic_ai(monkeypatch: pytest.MonkeyPatch) -> None:
    """Hide the pydantic-ai startup banner from test output."""
    monkeypatch.setenv("PYDANTIC_AI_NO_BANNER", "1")


class _RejectingMemory(FakeMemory):
    """A fake memory whose invalidate rejects every fact with one error."""

    def __init__(self, error: InvalidArgumentError | PayloadTooLargeError) -> None:
        """Keep the error that invalidate raises."""
        super().__init__()
        self._error = error

    async def invalidate(self, fact_id: str, *, reason: str = "") -> WriteReceipt:
        """Raise the configured error instead of closing the fact."""
        raise self._error


def _tool_returns(messages: list[ModelMessage]) -> list[ToolReturnPart]:
    """Return every tool result the model has received, in request order."""
    return [
        part
        for message in messages
        if isinstance(message, ModelRequest)
        for part in message.parts
        if isinstance(part, ToolReturnPart)
    ]


def _fact_id_from(search_text: str) -> str:
    """Return the first bracketed fact id in a search result."""
    match = re.search(r"\[(fact-\d+)\]", search_text)
    if match is None:
        raise AssertionError("search result has no bracketed fact id")
    return match.group(1)


async def _offered_tools(capability: MemoryCapability) -> list[ToolDefinition]:
    """Run one plain answer and return the tool definitions the model received."""
    seen: list[AgentInfo] = []

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        seen.append(info)
        return ModelResponse(parts=[TextPart("ok")])

    agent = Agent(
        FunctionModel(respond),
        output_type=str,
        name="assistant",
        capabilities=[capability],
    )
    await agent.run("hello")
    return seen[0].function_tools


class TestMemoryCapabilityTools:
    """Behavior of the search, remember and forget tools that the model calls."""

    @pytest.mark.parametrize(
        ("tools", "expected"),
        [
            (("search", "remember", "forget"), ["search", "remember", "forget"]),
            (("search",), ["search"]),
            (("remember", "forget"), ["remember", "forget"]),
            (("forget", "search"), ["search", "forget"]),
            ((), []),
        ],
        ids=["all", "search-only", "writes-only", "reversed-input", "none"],
    )
    async def test_offers_enabled_tools_in_fixed_order(
        self, tools: tuple[str, ...], expected: list[str]
    ) -> None:
        """Offers only enabled tools, in the fixed order search, remember, forget."""
        capability = MemoryCapability(FakeMemory(), tools=tools)

        offered = await _offered_tools(capability)

        assert [tool.name for tool in offered] == expected

    def test_rejects_unknown_tool_name_when_built(self) -> None:
        """Raises ValueError that names the unknown tool."""
        with pytest.raises(ValueError, match="bogus"):
            MemoryCapability(FakeMemory(), tools=("search", "bogus"))

    async def test_every_model_visible_argument_has_a_description(self) -> None:
        """Gives each argument of each offered tool a description."""
        offered = await _offered_tools(MemoryCapability(FakeMemory()))

        for tool in offered:
            for argument, schema in tool.parameters_json_schema["properties"].items():
                assert "description" in schema, f"{tool.name}.{argument}"

    async def test_forget_takes_only_fact_id_and_reason(self) -> None:
        """Exposes exactly the fact_id and reason arguments on forget."""
        offered = await _offered_tools(MemoryCapability(FakeMemory()))
        by_name = {tool.name: tool for tool in offered}

        forget_arguments = set(by_name["forget"].parameters_json_schema["properties"])

        assert forget_arguments == {"fact_id", "reason"}

    async def test_no_tool_exposes_a_hard_delete_control(self) -> None:
        """Offers no argument that could select a hard delete on any tool."""
        offered = await _offered_tools(MemoryCapability(FakeMemory()))

        all_arguments = {
            argument
            for tool in offered
            for argument in tool.parameters_json_schema["properties"]
        }

        assert all_arguments.isdisjoint(HARD_DELETE_CONTROLS)

    async def test_forgets_a_fact_found_by_search_and_queues_a_note(self) -> None:
        """Forgets the fact whose id the search result showed, then queues a note."""
        memory = FakeMemory(context=fact_context(("fact-123", "Ana works at Acme")))

        def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            results = {part.tool_name: part for part in _tool_returns(messages)}
            if "search" not in results:
                return ModelResponse(parts=[ToolCallPart("search", {"query": "Ana"})])
            if "forget" not in results:
                # Take the id from the search result itself, not from other context.
                fact_id = _fact_id_from(str(results["search"].content))
                return ModelResponse(
                    parts=[
                        ToolCallPart("forget", {"fact_id": fact_id, "reason": REASON})
                    ]
                )
            if "remember" not in results:
                return ModelResponse(
                    parts=[ToolCallPart("remember", {"content": NOTE})]
                )
            return ModelResponse(parts=[TextPart("done")])

        agent = Agent(
            FunctionModel(respond),
            output_type=str,
            name="assistant",
            capabilities=[MemoryCapability(memory)],
        )

        before = datetime.now(UTC)
        run = await agent.run(
            "Where does Ana work now?", conversation_id="conversation-1"
        )
        after = datetime.now(UTC)

        assert run.output == "done"
        assert memory.invalidations == [Invalidation("fact-123", REASON)]
        [note] = [text for text in memory.texts if text.content == NOTE]
        assert (note.role, note.speaker, note.session_id, note.thread) == (
            None,
            "assistant",
            "conversation-1",
            "conversation-1",
        )
        assert note.reference_time is not None
        assert before <= note.reference_time <= after

    async def test_keeps_queued_note_when_the_run_fails_after_it(self) -> None:
        """Keeps a note that a tool call queued even when the run then fails."""
        memory = FakeMemory()
        capability = MemoryCapability(memory, tools=("remember", "forget"), write=False)

        def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            if not _tool_returns(messages):
                return ModelResponse(
                    parts=[ToolCallPart("remember", {"content": NOTE})]
                )
            raise RuntimeError("model unavailable")

        agent = Agent(
            FunctionModel(respond),
            output_type=str,
            name="assistant",
            capabilities=[capability],
        )

        with pytest.raises(RuntimeError, match="model unavailable"):
            await agent.run("Remember that Ana moved.")

        assert [text.content for text in memory.texts] == [NOTE]

    async def test_write_false_stores_only_the_note_when_the_run_succeeds(self) -> None:
        """Stores the explicit note but no automatic Episodes after a good run."""
        memory = FakeMemory()
        capability = MemoryCapability(memory, tools=("remember", "forget"), write=False)

        def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            if not _tool_returns(messages):
                return ModelResponse(
                    parts=[ToolCallPart("remember", {"content": NOTE})]
                )
            return ModelResponse(parts=[TextPart("Noted.")])

        agent = Agent(
            FunctionModel(respond),
            output_type=str,
            name="assistant",
            capabilities=[capability],
        )

        result = await agent.run("Remember that Ana moved.")

        assert result.output == "Noted."
        assert [text.content for text in memory.texts] == [NOTE]

    async def test_answer_only_run_retrieves_once_and_offers_no_tools(self) -> None:
        """Retrieves once, offers no tools and accepts no writes for an answer."""
        memory = FakeMemory(context=fact_context(("fact-1", "Ana works at Acme")))
        capability = MemoryCapability(memory, inject=True, tools=(), write=False)
        seen: list[AgentInfo] = []

        def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            seen.append(info)
            return ModelResponse(parts=[TextPart("Acme")])

        agent = Agent(
            FunctionModel(respond),
            output_type=str,
            name="assistant",
            instructions=ANSWER_RULE,
            capabilities=[capability],
        )

        run = await agent.run("Where does Ana work?")

        assert run.output == "Acme"
        assert len(memory.queries) == 1
        assert seen[0].function_tools == []
        assert memory.texts == []
        assert ANSWER_RULE in (seen[0].instructions or "")

    @pytest.mark.parametrize(
        "error",
        [
            InvalidArgumentError("fact id is too long"),
            PayloadTooLargeError("reason is too long"),
        ],
        ids=["invalid-argument", "payload-too-large"],
    )
    async def test_rejected_argument_reaches_the_model_as_retry_feedback(
        self, error: InvalidArgumentError | PayloadTooLargeError
    ) -> None:
        """Sends the service message back to the model so the run can finish."""
        memory = _RejectingMemory(error)
        feedback: list[str] = []

        def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            forget_results = [
                part for part in _tool_returns(messages) if part.tool_name == "forget"
            ]
            if not forget_results:
                return ModelResponse(
                    parts=[ToolCallPart("forget", {"fact_id": "fact-1"})]
                )
            feedback.extend(str(part.content) for part in forget_results)
            return ModelResponse(parts=[TextPart("Ana's fact could not be forgotten.")])

        agent = Agent(
            FunctionModel(respond),
            output_type=str,
            name="assistant",
            capabilities=[MemoryCapability(memory)],
        )

        run = await agent.run("Forget the Acme fact.")

        assert run.output == "Ana's fact could not be forgotten."
        assert len(feedback) == 1
        assert str(error) in feedback[0]
        assert memory.invalidations == []
