"""Read and write files owned by the memory server."""

import json
import os
from pathlib import Path
from typing import Any


def write_server_info(path: Path, info: dict[str, Any]) -> None:
    """Write the server's public process metadata as JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(json.dumps(info, sort_keys=True), encoding="utf-8")
    os.replace(temporary_path, path)


def read_server_info(path: Path) -> dict[str, Any] | None:
    """Read server metadata, returning None when it is absent or invalid."""
    try:
        info = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return info if isinstance(info, dict) else None


def write_server_token(path: Path, token: str) -> None:
    """Write a bearer token with owner-only permissions on POSIX."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as token_file:
        token_file.write(token)
    if os.name != "nt":
        path.chmod(0o600)


def read_server_token(path: Path) -> str | None:
    """Read a server token, returning None when the file is absent."""
    try:
        return path.read_text(encoding="utf-8").strip() or None
    except OSError:
        return None
