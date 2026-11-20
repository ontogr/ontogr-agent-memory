"""Update docs-api hook pins from uv.lock."""

from __future__ import annotations

import argparse
import re
import tomllib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / ".pre-commit-config.yaml"
LOCK = ROOT / "uv.lock"
PACKAGES = ("griffe2md",)
HOOK_RE = re.compile(
    r"(?P<header>\n\s+- id: docs-api\n)(?P<body>.*?)(?=\n\s+- id:|\n\nci:)",
    re.DOTALL,
)
DEPENDENCIES_RE = re.compile(r"^(\s+additional_dependencies: ).*$", re.MULTILINE)


def render() -> str:
    """Return the config with docs-api versions from uv.lock."""
    with LOCK.open("rb") as handle:
        packages = {
            item["name"].lower(): item["version"]
            for item in tomllib.load(handle)["package"]
        }
    dependencies = [f'"{name}=={packages[name]}"' for name in PACKAGES]
    replacement = r"\1" + f"[{', '.join(dependencies)}]"
    config = CONFIG.read_text()
    match = HOOK_RE.search(config)
    if match is None:
        raise RuntimeError("docs-api hook not found")
    body, replacement_count = DEPENDENCIES_RE.subn(replacement, match.group(0), count=1)
    if replacement_count != 1:
        raise RuntimeError("docs-api additional_dependencies not found")
    return config[: match.start()] + body + config[match.end() :]


def main() -> int:
    """Update or check the generated docs-api pins."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    generated = render()
    current = CONFIG.read_text()
    if current == generated:
        return 0
    if args.check:
        print(
            ".pre-commit-config.yaml docs-api pins are stale; run make sync-docs-pins"
        )
        return 1
    CONFIG.write_text(generated)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
