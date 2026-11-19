"""Focused lifecycle coverage for bootstrap helpers."""

from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from ontogr_mem.errors import StoreInUseError
from ontogr_mem.service import bootstrap


@pytest.mark.asyncio
async def test_clear_session_opens_service_and_releases_sqlite_lock(
    monkeypatch, tmp_path
):
    """A clear session holds the SQLite server lock only while open."""
    database = tmp_path / "state.db"
    monkeypatch.setattr(
        bootstrap,
        "StateSettings",
        lambda: SimpleNamespace(
            url=SimpleNamespace(get_secret_value=lambda: f"sqlite:///{database}")
        ),
    )
    lock_events = []

    class Lock:
        def __init__(self, path):
            lock_events.append(("init", path))

        def acquire(self):
            lock_events.append(("acquire", None))

        def release(self):
            lock_events.append(("release", None))

    service = object()

    @asynccontextmanager
    async def open_service(**kwargs):
        assert kwargs["start_jobs"] is False
        yield SimpleNamespace(service=service)

    monkeypatch.setattr(bootstrap, "ServerLock", Lock)
    monkeypatch.setattr(bootstrap, "open_service", open_service)
    async with bootstrap.clear_session("tenant-a") as actual:
        assert actual is service
        assert [event[0] for event in lock_events] == ["init", "acquire"]
    assert [event[0] for event in lock_events] == ["init", "acquire", "release"]
    assert lock_events[0][1] == Path(f"{database}.lock")


@pytest.mark.asyncio
async def test_clear_session_refuses_when_sqlite_lock_is_held(monkeypatch, tmp_path):
    """A held local server lock prevents opening clear state."""
    database = tmp_path / "state.db"
    monkeypatch.setattr(
        bootstrap,
        "StateSettings",
        lambda: SimpleNamespace(
            url=SimpleNamespace(get_secret_value=lambda: f"sqlite:///{database}")
        ),
    )

    class Lock:
        def __init__(self, _path):
            pass

        def acquire(self):
            raise OSError("locked")

    open_service = AsyncMock()
    monkeypatch.setattr(bootstrap, "ServerLock", Lock)
    monkeypatch.setattr(bootstrap, "open_service", open_service)
    with pytest.raises(StoreInUseError, match="Stop the server first"):
        async with bootstrap.clear_session("tenant-a"):
            pytest.fail("must not yield when server lock is held")
    open_service.assert_not_awaited()
