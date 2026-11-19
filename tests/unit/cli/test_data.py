"""Tenant listing and clear command safeguards."""

from contextlib import asynccontextmanager
from datetime import UTC, datetime

import pytest

from ontogr_mem.cli import data
from ontogr_mem.service.errors import ConflictError
from ontogr_mem.service.forget.results import ClearTenantPreview


class TestTenantCommands:
    """The commands only touch the selected tenants and require safe consent."""

    async def test_lists_tenants_in_sorted_order(self, monkeypatch):
        class Store:
            async def list_tenants(self):
                return ["zeta", "alpha"]

        @asynccontextmanager
        async def open_graph_store():
            yield Store()

        monkeypatch.setattr(data, "open_graph_store", open_graph_store)

        assert await data.list_tenants_text() == "alpha\nzeta"

    async def test_missing_tenant_does_not_open_clear_session(self, monkeypatch, capsys):
        class Store:
            async def list_tenants(self):
                return ["beta"]

        @asynccontextmanager
        async def open_graph_store():
            yield Store()

        async def unexpected_session(_tenant):
            raise AssertionError("a missing tenant must not start a clear session")

        monkeypatch.setattr(data, "open_graph_store", open_graph_store)
        monkeypatch.setattr(data, "clear_session", unexpected_session)

        result = await data.clear_tenant_command("alpha")

        assert result == 0
        assert "No data for tenant alpha" in capsys.readouterr().out

    @pytest.mark.parametrize(
        ("stdin_tty", "stdout_tty"), [(False, True), (True, False)]
    )
    async def test_non_interactive_clear_refuses_before_deletion(
        self, monkeypatch, capsys, stdin_tty, stdout_tty
    ):
        service = _Service()
        _install_clear_dependencies(monkeypatch, service)
        monkeypatch.setattr(data.sys, "stdin", _Terminal(stdin_tty))
        monkeypatch.setattr(
            data.sys, "stdout", _OutputTerminal(stdout_tty, data.sys.stdout)
        )

        result = await data.clear_tenant_command("alpha")

        assert result == 1
        assert "interactive terminal" in capsys.readouterr().out
        assert service.clear_calls == 0

    async def test_wrong_confirmation_refuses_without_clearing(
        self, monkeypatch, capsys
    ):
        service = _Service()
        _install_clear_dependencies(monkeypatch, service)
        monkeypatch.setattr(data.sys, "stdin", _Terminal(True))
        monkeypatch.setattr(data.sys, "stdout", _OutputTerminal(True, data.sys.stdout))
        monkeypatch.setattr("builtins.input", lambda _prompt: "alpha ")

        result = await data.clear_tenant_command("alpha")

        assert result == 1
        assert "did not match" in capsys.readouterr().out
        assert service.clear_calls == 0

    async def test_exact_confirmation_clears_selected_tenant(
        self, monkeypatch, capsys
    ):
        service = _Service()
        _install_clear_dependencies(monkeypatch, service)
        monkeypatch.setattr(data.sys, "stdin", _Terminal(True))
        monkeypatch.setattr(data.sys, "stdout", _OutputTerminal(True, data.sys.stdout))
        monkeypatch.setattr("builtins.input", lambda _prompt: "alpha")

        result = await data.clear_tenant_command("alpha")

        output = capsys.readouterr().out
        assert result == 0
        assert service.clear_calls == 1
        assert "Kept: audit rows, operation rows and erasure rows" in output

    async def test_job_conflict_is_reported_without_success(
        self, monkeypatch, capsys
    ):
        service = _Service(conflict=True)
        _install_clear_dependencies(monkeypatch, service)
        monkeypatch.setattr(data.sys, "stdin", _Terminal(True))
        monkeypatch.setattr(data.sys, "stdout", _OutputTerminal(True, data.sys.stdout))
        monkeypatch.setattr("builtins.input", lambda _prompt: "alpha")

        result = await data.clear_tenant_command("alpha")

        output = capsys.readouterr().out
        assert result == 1
        assert "leased" in output
        assert "Removed graph" not in output
        assert service.clear_calls == 1


class _Terminal:
    def __init__(self, is_tty: bool) -> None:
        self._is_tty = is_tty

    def isatty(self) -> bool:
        return self._is_tty


class _OutputTerminal:
    def __init__(self, is_tty: bool, output) -> None:
        self._is_tty = is_tty
        self._output = output

    def isatty(self) -> bool:
        return self._is_tty

    def write(self, value: str) -> int:
        return self._output.write(value)

    def flush(self) -> None:
        self._output.flush()


class _Store:
    async def list_tenants(self) -> list[str]:
        return ["alpha", "beta"]


class _Service:
    def __init__(self, *, conflict: bool = False) -> None:
        self.clear_calls = 0
        self.conflict = conflict

    async def preview_clear_tenant(self, _principal, _tenant):
        return ClearTenantPreview(
            mode="clear",
            episode_ids=("episode-1",),
            fact_ids=("fact-1",),
            entity_ids=("entity-1",),
            job_count=1,
            confirm_token="token",
            expires_at=datetime.now(UTC),
        )

    async def clear_tenant(self, _principal, _request):
        self.clear_calls += 1
        if self.conflict:
            raise ConflictError("Jobs are leased until later.")


def _install_clear_dependencies(monkeypatch, service: _Service) -> None:
    @asynccontextmanager
    async def open_graph_store():
        yield _Store()

    @asynccontextmanager
    async def clear_session(_tenant):
        yield service

    monkeypatch.setattr(data, "open_graph_store", open_graph_store)
    monkeypatch.setattr(data, "clear_session", clear_session)
