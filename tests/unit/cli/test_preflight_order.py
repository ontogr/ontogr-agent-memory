"""Preflight runs before an embedder, a state store or a graph opens."""

import subprocess
import sys
import textwrap

import pytest


class TestPreflightBeforeOpen:
    """A missing extra is reported before a command creates any store file."""

    @pytest.mark.parametrize(
        ("command", "blocked", "state_url", "expected", "expects_uvx"),
        [
            pytest.param(
                "chat",
                ("falkordb", "redis", "fastembed"),
                "sqlite",
                ("falkordb", "fastembed"),
                False,
                id="chat-missing-graph-and-embedder",
            ),
            pytest.param(
                "mcp",
                ("fastmcp",),
                "sqlite",
                ("mcp",),
                True,
                id="mcp-missing-fastmcp",
            ),
            pytest.param(
                "chat",
                ("asyncpg",),
                "postgres",
                ("postgres",),
                False,
                id="chat-missing-postgres",
            ),
        ],
    )
    def test_reports_missing_extras_before_opening_stores(
        self, tmp_path, command, blocked, state_url, expected, expects_uvx
    ):
        """One error names every missing extra before any store file appears."""
        state_path = tmp_path / "state.sqlite3"
        cache_dir = tmp_path / "models"
        cache_dir.mkdir()
        url = (
            "postgresql://memory:memory@127.0.0.1:5432/memory"
            if state_url == "postgres"
            else f"sqlite:///{state_path}"
        )
        code = f"""
            import os
            import sys

            os.environ["MEMORY_LLM_MODEL"] = "test-model"
            os.environ["MEMORY_LLM_BASE_URL"] = "http://127.0.0.1:1/v1"
            os.environ["MEMORY_GRAPH_URL"] = "redis://127.0.0.1:6379"
            os.environ["MEMORY_STATE_URL"] = {url!r}
            os.environ["FASTEMBED_CACHE_PATH"] = r"{cache_dir}"

            import importlib.abc

            blocked = {blocked!r}

            class Blocker(importlib.abc.MetaPathFinder):
                def find_spec(self, name, path=None, target=None):
                    if name.split(".")[0] in blocked:
                        raise ImportError(name)
                    return None

            sys.meta_path.insert(0, Blocker())
            from ontogr_mem.cli import main
            sys.exit(main([{command!r}]))
        """
        result = subprocess.run(
            [sys.executable, "-I", "-c", textwrap.dedent(code)],
            cwd=tmp_path,
            capture_output=True,
            text=True,
            check=False,
        )

        assert result.returncode == 1, result.stdout + result.stderr
        for name in expected:
            assert name in result.stderr
        if expects_uvx:
            assert 'uvx --from "ontogr-mem[' in result.stderr
        if state_url == "sqlite":
            assert not state_path.exists()
        assert list(cache_dir.iterdir()) == []
