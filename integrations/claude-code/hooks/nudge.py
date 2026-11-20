"""Print periodic reminders for Claude Code sessions."""

import json
import os
import re
import sys
import time
from pathlib import Path


DEFAULTS = {"prompt": 10, "stop": 5}
EVENT_NAMES = {
    "session-start": "SessionStart",
    "user-prompt": "UserPromptSubmit",
    "stop": "Stop",
}


def main() -> None:  # noqa: PLR0911
    """Print an event reminder when its configured interval is reached."""
    event = sys.argv[1] if len(sys.argv) == 2 else ""
    if event not in EVENT_NAMES:
        return
    root = Path(__file__).parent
    if event == "session-start":
        emit(event, (root / "nudge_session_start.txt").read_text())
        return
    try:
        payload = json.load(sys.stdin)
    except (ValueError, OSError):
        payload = {}
    session_id = payload.get("session_id") if isinstance(payload, dict) else None
    if not isinstance(session_id, str) or not re.fullmatch(
        r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}",
        session_id,
    ):
        return
    if event == "stop" and payload.get("stop_hook_active") is True:
        return
    data_directory = os.environ.get("CLAUDE_PLUGIN_DATA")
    if not data_directory:
        return
    try:
        directory = Path(data_directory)
        directory.mkdir(parents=True, exist_ok=True)
        now = time.time()
        for state_file in directory.glob("nudge-*.json"):
            if now - state_file.stat().st_mtime > 7 * 24 * 60 * 60:
                state_file.unlink()
        state_path = directory / f"nudge-{session_id}.json"
        state = json.loads(state_path.read_text()) if state_path.exists() else {}
        counter = "prompt" if event == "user-prompt" else "stop"
        count = state.get(counter, 0) + 1
        state[counter] = count
        state_path.write_text(json.dumps(state))
    except (OSError, ValueError, TypeError):
        return
    environment_option = f"CLAUDE_PLUGIN_OPTION_{counter.upper()}_NUDGE_EVERY"
    try:
        threshold = int(os.environ.get(environment_option, DEFAULTS[counter]))
    except ValueError:
        threshold = DEFAULTS[counter]
    if not 1 <= threshold <= 50 or count % threshold:
        return
    filename = (
        f"nudge_{counter}_prompt.txt" if counter == "prompt" else "nudge_stop.txt"
    )
    emit(event, (root / filename).read_text())


def emit(event: str, context: str) -> None:
    """Write a hook response with additional context."""
    print(
        json.dumps(
            {
                "hookSpecificOutput": {
                    "hookEventName": EVENT_NAMES[event],
                    "additionalContext": context.strip(),
                }
            }
        )
    )


if __name__ == "__main__":
    main()
