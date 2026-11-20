"""Check reminder output and per-session intervals for the Claude Code hook."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


PLUGIN_ROOT = Path(__file__).parents[3] / "integrations" / "claude-code"
HOOK = PLUGIN_ROOT / "hooks" / "nudge.py"


class TestNudgeHook:
    """Run the hook as a subprocess with isolated session storage."""

    @pytest.fixture
    def environment(self, tmp_path: Path) -> dict[str, str]:
        """Set the hook's persistent data directory to a temporary path."""
        return {**os.environ, "CLAUDE_PLUGIN_DATA": str(tmp_path)}

    def run_hook(
        self, environment: dict[str, str], event: str, payload: dict[str, object]
    ) -> str:
        """Run one hook event and return its standard output."""
        result = subprocess.run(
            [sys.executable, str(HOOK), event],
            input=json.dumps(payload),
            text=True,
            capture_output=True,
            check=True,
            env=environment,
        )
        return result.stdout

    def test_session_start_prints_nudge(self, environment: dict[str, str]) -> None:
        """Print the session-start reminder without session state."""
        output = json.loads(self.run_hook(environment, "session-start", {}))
        assert output["hookSpecificOutput"]["additionalContext"] == (
            (PLUGIN_ROOT / "hooks" / "nudge_session_start.txt").read_text().strip()
        )

    def test_user_prompt_prints_on_tenth_prompt(
        self, environment: dict[str, str]
    ) -> None:
        """Print the user-prompt reminder on the configured tenth prompt."""
        session_id = "123e4567-e89b-12d3-a456-426614174000"
        for _ in range(9):
            assert (
                self.run_hook(environment, "user-prompt", {"session_id": session_id})
                == ""
            )
        output = json.loads(
            self.run_hook(environment, "user-prompt", {"session_id": session_id})
        )
        assert output["hookSpecificOutput"]["additionalContext"] == (
            (PLUGIN_ROOT / "hooks" / "nudge_user_prompt.txt").read_text().strip()
        )

    def test_stop_prints_on_fifth_response_and_ignores_hook_continuation(
        self, environment: dict[str, str]
    ) -> None:
        """Count stop events and ignore hook continuation events."""
        session_id = "123e4567-e89b-12d3-a456-426614174000"
        for _ in range(4):
            assert self.run_hook(environment, "stop", {"session_id": session_id}) == ""
        result = self.run_hook(
            environment, "stop", {"session_id": session_id, "stop_hook_active": False}
        )
        output = json.loads(result)
        assert output["hookSpecificOutput"]["additionalContext"] == (
            (PLUGIN_ROOT / "hooks" / "nudge_stop.txt").read_text().strip()
        )
        assert (
            self.run_hook(
                environment,
                "stop",
                {"session_id": session_id, "stop_hook_active": True},
            )
            == ""
        )
