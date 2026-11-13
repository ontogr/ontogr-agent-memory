"""Stream terminal answers and drain queued memory writes before exit."""

from __future__ import annotations

import asyncio
import signal
import sys
import threading
from collections.abc import Callable
from typing import Any, TextIO
from uuid import uuid4

from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage

from ontogr_mem.client import Memory


_DRAIN_TIMEOUT_SECONDS = 30.0


async def _read_line(input_fn: Callable[[str], str]) -> str:
    """Read without blocking the event loop or its shutdown executor.

    The reader is a daemon thread, so an interrupted chat can exit while a
    line is still being read. The default executor would wait for it.
    """
    loop = asyncio.get_running_loop()
    future: asyncio.Future[str] = loop.create_future()

    def deliver(value: str | Exception | None) -> None:
        if future.done():
            return
        if value is None:
            future.cancel()
        elif isinstance(value, Exception):
            future.set_exception(value)
        else:
            future.set_result(value)

    def read() -> None:
        try:
            value: str | Exception | None = input_fn("")
        except KeyboardInterrupt:
            value = None
        except Exception as error:
            value = error
        try:
            loop.call_soon_threadsafe(deliver, value)
        except RuntimeError:
            return

    threading.Thread(target=read, daemon=True).start()
    return await future


async def _drain(memory: Memory, output: TextIO) -> None:
    """Wait at most 30 seconds; a Ctrl-C during saving cancels the wait."""
    output.write("\nsaving memory... press Ctrl-C to stop waiting\n")
    output.flush()
    task = asyncio.create_task(memory.drain())
    loop = asyncio.get_running_loop()
    previous_handler = None
    if threading.current_thread() is threading.main_thread():
        # The handler runs outside the loop, so it must wake the loop to cancel.
        previous_handler = signal.signal(
            signal.SIGINT,
            lambda _signum, _frame: loop.call_soon_threadsafe(task.cancel),
        )
    try:
        await asyncio.wait_for(task, timeout=_DRAIN_TIMEOUT_SECONDS)
    except TimeoutError:
        output.write("Saving timed out; unfinished jobs resume next start.\n")
    except asyncio.CancelledError:
        output.write("Saving interrupted; unfinished jobs resume next start.\n")
    finally:
        if previous_handler is not None:
            signal.signal(signal.SIGINT, previous_handler)
        output.flush()


async def run_chat(
    memory: Memory,
    agent: Agent[Any, str],
    *,
    input_fn: Callable[[str], str] = input,
    output: TextIO = sys.stdout,
) -> None:
    """Run text Turns with process-local history, then drain queued writes.

    Args:
        memory: Open client that remains open until this function returns.
        agent: Text-answer agent with a streaming model and memory capability.
        input_fn: Blocking line reader, injectable for non-TTY tests.
        output: Destination for answers and shutdown status.
    """
    messages: list[ModelMessage] = []
    conversation_id = str(uuid4())
    try:
        while True:
            output.write("you> ")
            output.flush()
            try:
                prompt = await _read_line(input_fn)
            except EOFError:
                break
            if prompt.strip() == "/exit":
                break
            if not prompt.strip():
                continue
            output.write("assistant> ")
            output.flush()
            async with agent.run_stream(
                prompt,
                message_history=messages,
                conversation_id=conversation_id,
            ) as result:
                async for text in result.stream_text(delta=True):
                    output.write(text)
                    output.flush()
                await result.get_output()
            messages = result.all_messages()
            output.write("\n")
    finally:
        await _drain(memory, output)
