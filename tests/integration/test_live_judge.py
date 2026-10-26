"""The judge against a real model on Ollama Cloud. Skipped without a key.

The run prints the full report so the unmeasured judge thresholds can be
compared with real numbers. Run it with ``OLLAMA_API_KEY`` set for one command.
"""

import asyncio

import pytest

from tests.eval.runner import run_eval
from tests.support.ollama import OllamaLLMClient, available


@pytest.mark.skipif(not available(), reason="OLLAMA_API_KEY is not set")
def test_the_labelled_set_runs_end_to_end(capsys) -> None:  # type: ignore[no-untyped-def]
    """Every labelled pair is judged, and few are left unanswered."""
    report = asyncio.run(run_eval(OllamaLLMClient()))

    with capsys.disabled():
        print("\n" + report.render())
    assert report.n == 39
    assert report.calls > 0
    assert report.unanswered <= report.n // 4
