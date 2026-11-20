"""Check that release versions and distribution files agree."""

from __future__ import annotations

import json
import os
import re
import sys
import tomllib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXTRAS = "local,neo4j,postgres"
PIN_PATTERN = re.compile(r"ontogr-mem\[([^]]+)\]==([0-9A-Za-z.+!-]+)")
COMMAND_PATTERN = re.compile(
    rf"^.*ontogr-mem\[{re.escape(EXTRAS)}\]==[0-9A-Za-z.+!-]+.*$"
)


def _fail(message: str) -> None:
    print(f"distribution check: {message}", file=sys.stderr)
    raise SystemExit(1)


def main() -> None:  # noqa: PLR0912, PLR0915
    """Validate versions, extras, copied files, and host configuration syntax."""
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text())
    package_version = pyproject["project"]["version"]
    plugin_dir = ROOT / "integrations/claude-code"
    plugin = json.loads((plugin_dir / ".claude-plugin/plugin.json").read_text())
    version = plugin["version"]
    if version != package_version:
        _fail("plugin and package versions differ")

    compose = (ROOT / "compose.yaml").read_text()
    match = re.search(r"ONTOGR_MEM_VERSION:-([^}]+)", compose)
    if match is None or match.group(1) != version:
        _fail("Compose version default differs from package version")

    tag = os.environ.get("TAG")
    if tag and tag.removeprefix("v") != version:
        _fail("TAG differs from package version")

    integrations = ROOT / "integrations"
    for path in integrations.rglob("*"):
        if not path.is_file():
            continue
        content = path.read_text(errors="replace")
        for extras, pinned_version in PIN_PATTERN.findall(content):
            if extras != EXTRAS:
                _fail(f"unexpected extras in {path.relative_to(ROOT)}")
            if pinned_version != version:
                _fail(f"unexpected version in {path.relative_to(ROOT)}")

    dockerfile = (ROOT / "Dockerfile").read_text()
    docker_extras = re.findall(r"--extra\s+([\w-]+)", dockerfile)
    if docker_extras != ["local", "neo4j", "postgres"]:
        _fail("Dockerfile extras differ from the distribution extras")

    first_run = (ROOT / "integrations/hosts/first-run.txt").read_text().strip()
    if not COMMAND_PATTERN.fullmatch(first_run):
        _fail("first-run command has an unexpected format")
    if f"=={version}" not in first_run:
        _fail("first-run version differs from package version")
    for name in ("claude-code-cli.txt",):
        command = (ROOT / "integrations/hosts" / name).read_text().strip()
        if not COMMAND_PATTERN.fullmatch(command) or f"=={version}" not in command:
            _fail(f"{name} has an unexpected command")

    pairs = (
        (
            "ontogr_mem/mcp_server/skills/memory-write/SKILL.md",
            "integrations/claude-code/skills/memory-write/SKILL.md",
        ),
        (
            "ontogr_mem/mcp_server/skills/memory-recall/SKILL.md",
            "integrations/claude-code/skills/memory-recall/SKILL.md",
        ),
        (
            "ontogr_mem/mcp_server/wording/nudge_session_start.txt",
            "integrations/claude-code/hooks/nudge_session_start.txt",
        ),
        (
            "ontogr_mem/mcp_server/wording/nudge_user_prompt.txt",
            "integrations/claude-code/hooks/nudge_user_prompt.txt",
        ),
        (
            "ontogr_mem/mcp_server/wording/nudge_stop.txt",
            "integrations/claude-code/hooks/nudge_stop.txt",
        ),
    )
    for source, copy in pairs:
        if (ROOT / source).read_bytes() != (ROOT / copy).read_bytes():
            _fail(f"copy differs: {copy}")

    json.loads((plugin_dir / ".mcp.json").read_text())
    json.loads((plugin_dir / "hooks/hooks.json").read_text())
    json.loads((ROOT / "integrations/hosts/cursor.json").read_text())
    with (ROOT / "integrations/hosts/codex.toml").open("rb") as file:
        tomllib.load(file)
    opencode = (ROOT / "integrations/hosts/opencode.jsonc").read_text()
    opencode = re.sub(r"(?m)^\s*//.*$", "", opencode)
    json.loads(opencode)

    marketplace = {
        "name": "ontogr",
        "owner": {"name": "OntoGr"},
        "description": "OntoGr memory plugins for Claude Code.",
        "plugins": [
            {
                "name": plugin["name"],
                "source": {
                    "source": "git-subdir",
                    "url": "https://github.com/ontogr/ontogr-agent-memory.git",
                    "path": "integrations/claude-code",
                    "ref": f"v{version}",
                },
                "description": (
                    "Long-term memory for Claude Code through the OntoGr MCP server."
                ),
            }
        ],
    }
    entry = marketplace["plugins"][0]
    if entry["name"] != plugin["name"] or entry["source"]["ref"] != f"v{version}":
        _fail("candidate marketplace does not match the plugin")
    if (
        not (ROOT / entry["source"]["path"]).is_dir()
        or not marketplace["owner"]["name"]
    ):
        _fail("candidate marketplace metadata is incomplete")


if __name__ == "__main__":
    main()
