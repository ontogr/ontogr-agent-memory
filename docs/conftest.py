"""Fixtures for the Python blocks in the documentation pages.

Each block runs in a forked child process. The parent picks the data root before
any fork, so every child writes to the same directory for its page. The last
block of a page stops the server that the page started. A temporary root is
removed when pytest exits in the parent process.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest


def _data_root() -> tuple[Path, tempfile.TemporaryDirectory[str] | None]:
    """Return the shared data root and the temporary directory that owns it."""
    configured = os.environ.get("DOCS_TEST_DATA_ROOT")
    if configured:
        return Path(configured), None
    temporary = tempfile.TemporaryDirectory(prefix="ontogr-docs-")
    return Path(temporary.name), temporary


DATA_ROOT, _TEMP_DATA_ROOT = _data_root()
# Keep the temporary directory referenced so cleanup runs when the parent exits.
_last_item_by_page: dict[str, pytest.Item] = {}


@pytest.hookimpl(trylast=True)
def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    """Record the last markdown item of each page before the blocks are forked."""
    if config.getoption("collectonly"):
        return
    for item in items:
        path = getattr(item, "path", None)
        if path is not None and path.suffix in {".md", ".mdx"}:
            _last_item_by_page[str(path)] = item


@pytest.fixture(autouse=True)
def docs_page_store(
    request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Give a page its own data directory and stop its server after the page."""
    path = getattr(request.node, "path", None)
    if path is None or path.suffix not in {".md", ".mdx"}:
        yield None
        return
    page = str(path)
    data_directory = DATA_ROOT / f"{path.parent.name}-{path.stem}"
    monkeypatch.setenv("MEMORY_DATA_DIR", str(data_directory))
    yield
    if _last_item_by_page.get(page) is not request.node:
        return
    environment = os.environ.copy()
    environment["MEMORY_DATA_DIR"] = str(data_directory)
    subprocess.run(
        [sys.executable, "-m", "ontogr_mem", "server", "stop"],
        env=environment,
        capture_output=True,
        check=False,
    )
