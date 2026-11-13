"""Isolate model settings tests from the real environment."""

import os
from collections.abc import Iterator
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def clean_llm_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[None]:
    """Isolate model settings tests from the environment and working directory.

    Remove every MEMORY_LLM_*, OPENAI_* and ANTHROPIC_* variable and run the
    test from an empty temporary directory, so a local .env file is not read.
    """
    for name in list(os.environ):
        if name.startswith(("MEMORY_LLM_", "OPENAI_", "ANTHROPIC_")):
            monkeypatch.delenv(name)
    monkeypatch.chdir(tmp_path)
    yield
