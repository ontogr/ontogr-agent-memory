"""Deterministic models that exercise agent output validation without a network."""

import json
from collections import deque
from collections.abc import Callable, Sequence
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolCallPart,
    UserPromptPart,
)
from pydantic_ai.models import Model
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.profiles import ModelProfile

from ontogr_mem.llm import LLMConfig, ModelRole, RoleModels, build_models


@dataclass(frozen=True, slots=True)
class ScriptedCall:
    """Instructions, last user text and detached history seen by a model."""

    system: str
    prompt: str
    messages: list[ModelMessage]
    info: AgentInfo


@dataclass(slots=True)
class Scripted:
    """Recorded model calls in request order."""

    calls: list[ScriptedCall] = field(default_factory=list)


def scripted_model(
    replies: Sequence[Any] = (),
    *,
    answer: Callable[[ScriptedCall], Any] | None = None,
    mode: Literal["tool", "native", "prompted"] = "tool",
) -> tuple[FunctionModel, Scripted]:
    """Return a queued model and recorder, with an optional fallback answer.

    Args:
        replies: Objects, raw JSON strings or exceptions, consumed in order.
        answer: Called after the reply queue is empty.
        mode: Structured output protocol used by the model profile.

    Returns:
        The model and its independent recorder.

    Raises:
        AssertionError: If a call exhausts the queue without a fallback.
    """
    pending = deque(replies)
    scripted = Scripted()

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
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
        scripted.calls.append(call)
        if pending:
            reply = pending.popleft()
        elif answer is not None:
            reply = answer(call)
        else:
            raise AssertionError("unexpected model call: no scripted reply")
        if isinstance(reply, Exception):
            raise reply
        encoded = reply if isinstance(reply, str) else json.dumps(reply)
        if mode == "tool":
            if not info.output_tools:
                raise AssertionError("tool output requested without an output tool")
            return ModelResponse(
                parts=[ToolCallPart(info.output_tools[0].name, encoded)]
            )
        return ModelResponse(parts=[TextPart(encoded)])

    profile = ModelProfile(
        default_structured_output_mode=mode,
        supports_json_schema_output=True,
        supports_json_object_output=True,
    )
    return FunctionModel(respond, profile=profile), scripted


def scripted_role_models(model: Model | None = None, **per_role: Model) -> RoleModels:
    """Build limited models for all kinds of work or the named overrides.

    Args:
        model: Optional model used for every kind of work.
        **per_role: Models keyed by ModelRole values, overriding the shared model.

    Returns:
        A pool with no owned HTTP clients and no ambient model configuration.
    """
    overrides: dict[ModelRole, Model | str] = {}
    if model is not None:
        overrides.update(dict.fromkeys(ModelRole, model))
    overrides.update({ModelRole(name): value for name, value in per_role.items()})
    return build_models(LLMConfig(_env_file=None), overrides=overrides)
