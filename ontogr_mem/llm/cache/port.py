"""Completion-cache values, key generation, and async cache port."""

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol

from ontogr_mem.llm.base import ModelSize


@dataclass(frozen=True, slots=True)
class CacheEntry:
    """A completion and the prompt, usage, and tags needed to manage it.

    Attributes:
        prompt: User prompt associated with the completion.
        text: Completion text returned by the model.
        input_tokens: Provider-reported input token count.
        output_tokens: Provider-reported output token count.
        tags: Context tags used for targeted cache eviction.
    """

    prompt: str
    text: str
    input_tokens: int
    output_tokens: int
    tags: frozenset[str] = frozenset()


def cache_key(
    *,
    model_name: str,
    model_size: ModelSize,
    tenant: str,
    system: str,
    prompt: str,
    schema: dict[str, Any] | None,
    max_output_tokens: int,
) -> str:
    """Hash the model tier and request fields that determine a completion."""
    value = [
        model_name,
        model_size.value,
        tenant,
        system,
        prompt,
        schema,
        max_output_tokens,
    ]
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode()).hexdigest()


def holds(entry: CacheEntry, needle: str) -> bool:
    """Return whether the prompt or completion contains ``needle``."""
    return needle in entry.prompt or needle in entry.text


class ResponseCache(Protocol):
    """Async cache used for model completions and erasure lookups."""

    async def get(self, key: str) -> CacheEntry | None:
        """Return a cached completion, or None on a miss."""
        ...

    async def put(self, key: str, entry: CacheEntry) -> None:
        """Store or replace a completion."""
        ...

    async def delete(self, key: str) -> None:
        """Remove one completion if it exists."""
        ...

    async def evict_tag(self, tag: str) -> int:
        """Remove entries carrying ``tag`` and return the number removed."""
        ...

    async def count_tag(self, tag: str) -> int:
        """Count entries carrying ``tag`` without removing them."""
        ...

    async def scan(self, needle: str) -> int:
        """Count entries containing ``needle`` in prompt or completion text."""
        ...

    async def evict_text(self, needle: str) -> int:
        """Remove entries containing ``needle`` and return the number removed."""
        ...
