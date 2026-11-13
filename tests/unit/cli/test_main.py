"""The ontogr-mem command line: argument parsing and exit statuses."""

import os

import pytest
from pydantic import BaseModel, ValidationError

from ontogr_mem.cli import main
from ontogr_mem.errors import AgentMemoryError, MissingExtraError
from ontogr_mem.llm.failures import ModelNotConfiguredError
from ontogr_mem.vectorstore.builtin import EmbedderUnavailableError


class ChatLimits(BaseModel):
    """Settings-like model whose integer field rejects text input."""

    max_turns: int


def _validation_error() -> ValidationError:
    """Return a real pydantic error whose input is the sentinel secret text."""
    with pytest.raises(ValidationError) as excinfo:
        ChatLimits(max_turns="sentinel-secret-value")

    return excinfo.value


def _fake_asyncio_run(outcome):
    """Stand in for asyncio.run so that no chat session starts.

    The stand-in closes the coroutine it receives, which keeps Python from
    warning that it was never awaited. It then raises outcome, or returns
    None when outcome is None.
    """

    def fake_run(coroutine):
        coroutine.close()
        if outcome is not None:
            raise outcome

    return fake_run


class TestMain:
    """Behavior of the ontogr-mem command line entry point."""

    @pytest.fixture(autouse=True)
    def restore_banner_setting(self, monkeypatch):
        """Remove the banner variable that main sets after each test."""
        # main writes the variable straight to os.environ. Setting and then
        # deleting it through monkeypatch records the original state, so
        # teardown restores it for the other tests in this process.
        monkeypatch.setenv("PYDANTIC_AI_NO_BANNER", "0")
        monkeypatch.delenv("PYDANTIC_AI_NO_BANNER")

    @pytest.mark.parametrize(
        ("argv", "expected_text"),
        [
            pytest.param(["--help"], "chat", id="top-level"),
            pytest.param(["chat", "--help"], "--tenant", id="chat"),
            pytest.param(["--help"], "mcp", id="top-level-names-mcp"),
        ],
    )
    def test_help_prints_usage_and_exits_with_status_zero(
        self, capsys, argv, expected_text
    ):
        """Help prints usage to stdout and exits with status zero."""
        with pytest.raises(SystemExit) as excinfo:
            main(argv)

        assert excinfo.value.code == 0
        assert expected_text in capsys.readouterr().out

    @pytest.mark.parametrize("flag", ["--transport", "--host", "--port", "--tenant"])
    def test_mcp_help_names_each_flag(self, capsys, flag):
        """The mcp subcommand help lists its flags and exits with status zero."""
        with pytest.raises(SystemExit) as excinfo:
            main(["mcp", "--help"])

        assert excinfo.value.code == 0
        assert flag in capsys.readouterr().out

    def test_missing_subcommand_exits_with_status_two(self):
        """A command line without the chat subcommand is a usage error."""
        with pytest.raises(SystemExit) as excinfo:
            main([])

        assert excinfo.value.code == 2

    def test_malformed_tenant_exits_with_status_two(self, monkeypatch, capsys):
        """A tenant id outside the allowed characters is a usage error."""
        # The fake keeps a parsing regression from starting a real chat session.
        monkeypatch.setattr("ontogr_mem.cli.asyncio.run", _fake_asyncio_run(None))

        with pytest.raises(SystemExit) as excinfo:
            main(["chat", "--tenant", "bad tenant!"])

        assert excinfo.value.code == 2
        stderr = capsys.readouterr().err
        assert "usage:" in stderr
        assert "use 1 to 60 letters, digits, dashes or underscores" in stderr
        assert "validate_tenant_id" not in stderr

    def test_normal_exit_returns_zero(self, monkeypatch):
        """A chat session that ends without error returns status zero."""
        monkeypatch.setattr("ontogr_mem.cli.asyncio.run", _fake_asyncio_run(None))

        assert main(["chat"]) == 0

    def test_keyboard_interrupt_returns_130(self, monkeypatch):
        """An interrupted chat session returns the conventional status 130."""
        monkeypatch.setattr(
            "ontogr_mem.cli.asyncio.run", _fake_asyncio_run(KeyboardInterrupt())
        )

        assert main(["chat"]) == 130

    @pytest.mark.parametrize(
        ("error", "expected_texts"),
        [
            pytest.param(
                MissingExtraError("falkordb"),
                ('pip install "ontogr-mem[falkordb]"',),
                id="missing-extra",
            ),
            pytest.param(
                EmbedderUnavailableError(
                    "local embedding model could not load",
                    hint="install the local embedding extra",
                ),
                (
                    "local embedding model could not load",
                    "install the local embedding extra",
                ),
                id="embedder-unavailable",
            ),
            pytest.param(
                ModelNotConfiguredError("no model is configured for chat"),
                ("MEMORY_LLM_MODEL", "MEMORY_LLM_BASE_URL", "MEMORY_LLM_API_KEY"),
                id="model-not-configured",
            ),
            pytest.param(
                AgentMemoryError("store rejected the write"),
                ("Memory failed; check database, embedding and model settings.",),
                id="memory-error",
            ),
        ],
    )
    def test_reports_each_failure_on_stderr_and_returns_one(
        self, monkeypatch, capsys, error, expected_texts
    ):
        """Each failure prints its guidance to stderr and returns status one."""
        monkeypatch.setattr("ontogr_mem.cli.asyncio.run", _fake_asyncio_run(error))

        status = main(["chat"])

        stderr = capsys.readouterr().err
        assert status == 1
        for expected_text in expected_texts:
            assert expected_text in stderr

    @pytest.mark.parametrize(
        ("error", "expected_text"),
        [
            pytest.param(
                _validation_error(),
                "Invalid memory settings; check the named MEMORY_* variables.",
                id="validation-error",
            ),
            pytest.param(
                RuntimeError("sentinel-secret-value"),
                "Command failed; check the model endpoint and memory services.",
                id="unexpected-error",
            ),
        ],
    )
    def test_error_text_with_input_is_replaced_by_fixed_guidance(
        self, monkeypatch, capsys, error, expected_text
    ):
        """Error text can echo secret input, so only fixed guidance is printed."""
        monkeypatch.setattr("ontogr_mem.cli.asyncio.run", _fake_asyncio_run(error))

        status = main(["chat"])

        stderr = capsys.readouterr().err
        assert status == 1
        assert expected_text in stderr
        assert "sentinel-secret-value" not in stderr

    def test_disables_the_agent_banner_before_the_chat_starts(self, monkeypatch):
        """The banner variable is already "1" when the chat coroutine runs."""
        banner_values = []

        def fake_run(coroutine):
            banner_values.append(os.environ.get("PYDANTIC_AI_NO_BANNER"))
            coroutine.close()

        monkeypatch.setattr("ontogr_mem.cli.asyncio.run", fake_run)

        main(["chat"])

        assert banner_values == ["1"]
