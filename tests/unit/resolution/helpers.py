"""Shared fakes and builders for resolution tests."""

import json
import re
from collections.abc import Callable
from datetime import UTC, datetime

from ontogr_mem.data_models.entities import Entity
from ontogr_mem.llm import LLMCallError, LLMRequest, LLMResponse


FIXED_TIME = datetime(2026, 10, 21, tzinfo=UTC)


def entity(
    entity_id: str,
    name: str,
    type_label: str = "Company",
    summary: str = "",
) -> Entity:
    """Build an entity with a fixed clock."""
    return Entity(
        entity_id=entity_id,
        type_label=type_label,
        canonical_name=name,
        strict_key=name.casefold(),
        loose_key=name.casefold(),
        summary=summary,
        created_at=FIXED_TIME,
        updated_at=FIXED_TIME,
    )


_PAIR = re.compile(
    r'<pair n=(\d+)>\nA: name=("(?:[^"\\]|\\.)*").*?\nB: name=("(?:[^"\\]|\\.)*")',
    re.DOTALL,
)


class OracleLLM:
    """Answers verification prompts from a ``(name_a, name_b) -> bool`` rule."""

    def __init__(
        self,
        decide: Callable[[str, str], bool | None],
        *,
        skip: set[int] | None = None,
        garbage: bool = False,
    ) -> None:
        """``decide`` returning None leaves the pair unanswered."""
        self.decide, self.skip, self.garbage = decide, skip or set(), garbage
        self.requests: list[LLMRequest] = []

    async def complete(self, request: LLMRequest) -> LLMResponse:
        """Reply in the format the verifier asks for."""
        self.requests.append(request)
        lines = []
        for match in _PAIR.finditer(request.prompt):
            n = int(match.group(1))
            a, b = json.loads(match.group(2)), json.loads(match.group(3))
            if n in self.skip and len(self.requests) == 1:
                continue
            ans = self.decide(a, b)
            if ans is None:
                continue
            if self.garbage:
                lines.append(f"{n}. {a} vs {b} | rule: x | perhaps")
            else:
                lines.append(f"{n}. {a} vs {b} | rule: test | {'YES' if ans else 'NO'}")
        return LLMResponse(text="\n".join(lines), input_tokens=10, output_tokens=5)


class FailingLLM:
    """Always raises the language-model call error."""

    async def complete(self, request: LLMRequest) -> LLMResponse:
        """Raise."""
        raise LLMCallError("down")
