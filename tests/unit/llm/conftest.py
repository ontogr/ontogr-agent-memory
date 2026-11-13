"""Isolate model settings tests from the real environment."""

import os
from collections.abc import Iterator

import pytest


@pytest.fixture(autouse=True)
def clean_llm_env(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Remove every MEMORY_LLM_*, OPENAI_* and ANTHROPIC_* variable."""
    for name in list(os.environ):
        if name.startswith(("MEMORY_LLM_", "OPENAI_", "ANTHROPIC_")):
            monkeypatch.delenv(name)
    yield
