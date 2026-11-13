"""Unit tests for the chat loop with a scripted model and a fake memory."""

import asyncio
import io
import signal
import threading
from collections import deque
from collections.abc import AsyncIterator, Callable, Iterator
from contextlib import suppress
from typing import Any, cast

import pytest
from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage, ModelRequest, UserPromptPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

from ontogr_mem.agent import MemoryCapability
from ontogr_mem.cli.chat import run_chat
from ontogr_mem.client.memory import Memory
from tests.support.fake_memory import FakeMemory


StreamFunction = Callable[[list[ModelMessage], AgentInfo], AsyncIterator[str]]

FIRST_PROMPT = "my code word is amber"
SECOND_PROMPT = "what was my code word"


class ModelStreamError(RuntimeError):
    """A failure raised by a model stream after it has started answering."""


def _input_from(lines: list[str]) -> Callable[[str], str]:
    """Return a reader that yields each line in order, then raises EOFError."""
    pending = deque(lines)

    def read_line(prompt: str) -> str:
        if not pending:
            raise EOFError
        return pending.popleft()

    return read_line


def _streams(*deltas: str) -> StreamFunction:
    """Return a stream function that yields each delta in order."""

    async def stream(
        messages: list[ModelMessage], info: AgentInfo
    ) -> AsyncIterator[str]:
        for delta in deltas:
            yield delta

    return stream


async def _echo_earlier_prompt(
    messages: list[ModelMessage], info: AgentInfo
) -> AsyncIterator[str]:
    """Echo FIRST_PROMPT when an earlier request in the history contains it."""
    # The last request is the prompt being answered, so only the requests before
    # it can contain the first prompt.
    earlier_texts = [
        part.content
        for message in messages[:-1]
        if isinstance(message, ModelRequest)
        for part in message.parts
        if isinstance(part, UserPromptPart) and isinstance(part.content, str)
    ]
    if any(FIRST_PROMPT in text for text in earlier_texts):
        yield FIRST_PROMPT
    else:
        yield "noted"


def _agent(stream_function: StreamFunction, memory: Memory) -> Agent[Any, str]:
    """Return an agent that answers through stream_function and uses memory."""
    return Agent(
        FunctionModel(stream_function=stream_function),
        output_type=str,
        name="assistant",
        capabilities=[MemoryCapability(memory=memory, tools=())],
    )


@pytest.fixture(autouse=True)
def no_pydantic_ai_banner(monkeypatch: pytest.MonkeyPatch) -> None:
    """Suppress the startup banner that pydantic_ai prints."""
    monkeypatch.setenv("PYDANTIC_AI_NO_BANNER", "1")


@pytest.fixture
def reader_gate() -> Iterator[threading.Event]:
    """Hold a blocked reader thread until the test ends."""
    gate = threading.Event()
    yield gate
    gate.set()


class TestChat:
    """The chat loop streams answers, keeps history and drains memory on exit."""

    async def test_prints_each_streamed_delta_once_then_a_newline(self) -> None:
        """Each streamed delta is printed once and the answer ends with a newline."""
        fake = FakeMemory()
        memory = cast(Memory, fake)
        output = io.StringIO()

        async with asyncio.timeout(5):
            await run_chat(
                memory,
                _agent(_streams("Good ", "morning"), memory),
                input_fn=_input_from(["hi"]),
                output=output,
            )

        text = output.getvalue()
        assert "assistant> Good morning\n" in text
        assert text.count("Good morning") == 1

    async def test_second_turn_sees_the_first_prompt_in_one_conversation(self) -> None:
        """A second turn receives the first prompt, and both turns share a session."""
        fake = FakeMemory()
        memory = cast(Memory, fake)
        output = io.StringIO()

        async with asyncio.timeout(5):
            await run_chat(
                memory,
                _agent(_echo_earlier_prompt, memory),
                input_fn=_input_from([FIRST_PROMPT, SECOND_PROMPT]),
                output=output,
            )

        text = output.getvalue()
        assert "assistant> noted\n" in text
        assert f"assistant> {FIRST_PROMPT}\n" in text
        assert "saving memory... press Ctrl-C to stop waiting" in text
        assert fake.drained
        stored_contents = [stored.content for stored in fake.texts]
        assert stored_contents == [FIRST_PROMPT, "noted", SECOND_PROMPT, FIRST_PROMPT]
        session_ids = {stored.session_id for stored in fake.texts}
        assert len(session_ids) == 1
        assert None not in session_ids

    @pytest.mark.parametrize(
        "lines",
        [
            pytest.param(["/exit", "hello"], id="exit-before-next-line"),
            pytest.param(["", "   ", "\t"], id="blank-lines-then-end-of-input"),
            pytest.param(["  /exit  ", "hello"], id="padded-exit"),
            pytest.param([], id="end-of-input"),
        ],
    )
    async def test_exit_blank_lines_and_end_of_input_run_no_model_and_drain(
        self, lines: list[str]
    ) -> None:
        """Exit, blank lines and end of input never call the model but still drain."""
        fake = FakeMemory()
        memory = cast(Memory, fake)
        model_calls: list[list[ModelMessage]] = []

        async def stream(
            messages: list[ModelMessage], info: AgentInfo
        ) -> AsyncIterator[str]:
            model_calls.append(messages)
            yield "unused"

        async with asyncio.timeout(5):
            await run_chat(
                memory,
                _agent(stream, memory),
                input_fn=_input_from(lines),
                output=io.StringIO(),
            )

        assert model_calls == []
        assert fake.texts == []
        assert fake.drained

    async def test_event_loop_keeps_running_while_the_reader_waits(
        self, reader_gate: threading.Event
    ) -> None:
        """The event loop keeps running while the reader waits for a line."""
        fake = FakeMemory()
        memory = cast(Memory, fake)
        ticks = 0
        ticks_when_read: list[int] = []

        async def count_ticks() -> None:
            nonlocal ticks
            while True:
                ticks += 1
                await asyncio.sleep(0)

        def read_after_gate(prompt: str) -> str:
            # The bounded wait keeps a loop-blocking reader from hanging the run.
            reader_gate.wait(timeout=5)
            ticks_when_read.append(ticks)
            return "/exit"

        ticker = asyncio.create_task(count_ticks())
        chat = asyncio.create_task(
            run_chat(
                memory,
                _agent(_streams("unused"), memory),
                input_fn=read_after_gate,
                output=io.StringIO(),
            )
        )
        try:
            async with asyncio.timeout(10):
                while ticks < 100:
                    await asyncio.sleep(0)
                reader_gate.set()
                await chat
        finally:
            ticker.cancel()
            chat.cancel()
            with suppress(asyncio.CancelledError):
                await ticker

        assert ticks_when_read[0] >= 100

    async def test_model_error_is_raised_after_saving_without_storing_the_partial(
        self,
    ) -> None:
        """A failed stream re-raises after the drain and stores no partial answer."""
        fake = FakeMemory()
        memory = cast(Memory, fake)
        output = io.StringIO()

        async def failing_stream(
            messages: list[ModelMessage], info: AgentInfo
        ) -> AsyncIterator[str]:
            yield "Partial answer"
            raise ModelStreamError("model stream broke")

        with pytest.raises(ModelStreamError, match="model stream broke"):
            async with asyncio.timeout(5):
                await run_chat(
                    memory,
                    _agent(failing_stream, memory),
                    input_fn=_input_from(["hi"]),
                    output=output,
                )

        assert "saving memory... press Ctrl-C to stop waiting" in output.getvalue()
        assert fake.drained
        assert all("Partial answer" not in stored.content for stored in fake.texts)

    async def test_cancelling_while_waiting_for_input_drains_before_propagating(
        self, reader_gate: threading.Event
    ) -> None:
        """Cancelling the loop drains memory before the cancellation propagates."""
        fake = FakeMemory()
        memory = cast(Memory, fake)
        reader_started = threading.Event()

        def read_after_gate(prompt: str) -> str:
            reader_started.set()
            reader_gate.wait(timeout=5)
            return "/exit"

        chat = asyncio.create_task(
            run_chat(
                memory,
                _agent(_streams("unused"), memory),
                input_fn=read_after_gate,
                output=io.StringIO(),
            )
        )
        async with asyncio.timeout(5):
            assert await asyncio.to_thread(reader_started.wait, 5)
            chat.cancel()
            reader_gate.set()
            with pytest.raises(asyncio.CancelledError):
                await chat

        assert chat.cancelled()
        assert fake.drained

    async def test_drain_timeout_is_reported_and_the_loop_returns(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A drain that outlasts its timeout is reported and run_chat returns."""
        # Shrinks the production drain bound so that the test finishes quickly.
        monkeypatch.setattr("ontogr_mem.cli.chat._DRAIN_TIMEOUT_SECONDS", 0.05)
        fake = FakeMemory()
        fake.drain_gate = asyncio.Event()
        memory = cast(Memory, fake)
        output = io.StringIO()

        async with asyncio.timeout(5):
            await run_chat(
                memory,
                _agent(_streams("unused"), memory),
                input_fn=_input_from([]),
                output=output,
            )

        assert "Saving timed out; unfinished jobs resume next start." in (
            output.getvalue()
        )

    async def test_second_ctrl_c_interrupts_the_drain_and_restores_the_handler(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A SIGINT during the drain cancels the wait and restores the old handler."""
        previous = object()
        signal_calls: list[tuple[int, Any]] = []

        def record_signal(signum: int, handler: Any) -> object:
            signal_calls.append((signum, handler))
            return previous

        # Recording the calls keeps the real SIGINT disposition untouched and lets
        # the test invoke the installed handler directly.
        monkeypatch.setattr("ontogr_mem.cli.chat.signal.signal", record_signal)
        fake = FakeMemory()
        fake.drain_gate = asyncio.Event()
        memory = cast(Memory, fake)
        output = io.StringIO()

        chat = asyncio.create_task(
            run_chat(
                memory,
                _agent(_streams("unused"), memory),
                input_fn=_input_from([]),
                output=output,
            )
        )
        async with asyncio.timeout(5):
            await fake.drain_started.wait()
            handler = next(
                handler for signum, handler in signal_calls if signum == signal.SIGINT
            )
            handler(signal.SIGINT, None)
            await chat

        assert "Saving interrupted; unfinished jobs resume next start." in (
            output.getvalue()
        )
        assert signal_calls[-1] == (signal.SIGINT, previous)
