"""A scripted model client for tests."""

import json
from collections.abc import Callable
from typing import Any

from ontogr_mem.llm import LLMRequest, LLMResponse


class ScriptedLLM:
    """An `LLMClient` that answers from a queue or from a function.

    Replies are strings, objects encoded as JSON, or exceptions to raise. Every
    request is kept in `requests`.
    """

    def __init__(
        self,
        replies: list[Any] | None = None,
        *,
        answer: Callable[[LLMRequest], Any] | None = None,
    ) -> None:
        """Create the client.

        Args:
            replies: Replies returned in order, one per call.
            answer: Function that builds the reply from the request. Used when
                the queue is empty.
        """
        self._replies = list(replies or [])
        self._answer = answer
        self.requests: list[LLMRequest] = []

    async def complete(self, request: LLMRequest) -> LLMResponse:
        """Return the next scripted reply."""
        self.requests.append(request)
        if self._replies:
            reply = self._replies.pop(0)
        elif self._answer is not None:
            reply = self._answer(request)
        else:
            msg = "ScriptedLLM has no reply left"
            raise AssertionError(msg)
        if isinstance(reply, Exception):
            raise reply
        if isinstance(reply, LLMResponse):
            return reply
        text = reply if isinstance(reply, str) else json.dumps(reply)
        return LLMResponse(text=text, input_tokens=len(request.prompt), output_tokens=1)
