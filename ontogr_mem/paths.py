"""Resolve per-user data and cache directories."""

import os
import sys
from pathlib import Path


DATA_DIRECTORY_ENV = "MEMORY_DATA_DIR"


def data_directory() -> Path:
    """Return the data directory without creating it."""
    override = os.environ.get(DATA_DIRECTORY_ENV)
    if override:
        return Path(override).expanduser()
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / "ontogr-mem"
    if sys.platform == "win32":
        return Path(os.environ.get("LOCALAPPDATA", Path.home())) / "ontogr-mem"
    xdg = os.environ.get("XDG_DATA_HOME")
    root = (
        Path(xdg) if xdg and Path(xdg).is_absolute() else Path.home() / ".local/share"
    )
    return root / "ontogr-mem"


def cache_directory() -> Path:
    """Return the cache directory without creating it."""
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Caches" / "ontogr-mem"
    if sys.platform == "win32":
        return (
            Path(os.environ.get("LOCALAPPDATA", Path.home())) / "ontogr-mem" / "cache"
        )
    xdg = os.environ.get("XDG_CACHE_HOME")
    root = Path(xdg) if xdg and Path(xdg).is_absolute() else Path.home() / ".cache"
    return root / "ontogr-mem"


def model_cache_directory() -> Path:
    """Return the FastEmbed cache directory without creating it."""
    override = os.environ.get("FASTEMBED_CACHE_PATH")
    return Path(override).expanduser() if override else cache_directory() / "fastembed"


def ensure_directory(path: Path) -> Path:
    """Create a directory and restrict its POSIX permissions when new."""
    existed = path.exists()
    path.mkdir(parents=True, exist_ok=True)
    if not existed and os.name == "posix":
        path.chmod(0o700)
    return path
