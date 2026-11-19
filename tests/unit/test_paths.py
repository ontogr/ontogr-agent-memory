"""Behavior tests for per-user path resolution."""

import os
from pathlib import Path

import pytest

from ontogr_mem.paths import (
    cache_directory,
    data_directory,
    ensure_directory,
    model_cache_directory,
)


class TestDataDirectory:
    """The data directory honors the override and platform defaults."""

    def test_uses_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A non-empty override wins over the platform default."""
        monkeypatch.setenv("MEMORY_DATA_DIR", "/custom/data")
        assert data_directory() == Path("/custom/data")

    def test_uses_linux_xdg_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """An absolute XDG data home selects the Linux data directory."""
        monkeypatch.setattr("ontogr_mem.paths.sys.platform", "linux")
        monkeypatch.delenv("MEMORY_DATA_DIR", raising=False)
        monkeypatch.setenv("HOME", "/home/test")
        monkeypatch.setenv("XDG_DATA_HOME", "/data")
        assert data_directory() == Path("/data/ontogr-mem")

    def test_ignores_empty_windows_localappdata(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An empty Windows data variable falls back to the user's home."""
        monkeypatch.setattr("ontogr_mem.paths.sys.platform", "win32")
        monkeypatch.delenv("MEMORY_DATA_DIR", raising=False)
        monkeypatch.setenv("HOME", "/home/test")
        monkeypatch.setenv("LOCALAPPDATA", "")
        assert data_directory() == Path("/home/test/ontogr-mem")


class TestCacheDirectory:
    """The cache directory uses the XDG cache home on Linux."""

    def test_uses_xdg_cache(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """An absolute XDG cache home selects the Linux cache directory."""
        monkeypatch.setattr("ontogr_mem.paths.sys.platform", "linux")
        monkeypatch.setenv("XDG_CACHE_HOME", "/cache")
        assert cache_directory() == Path("/cache/ontogr-mem")

    def test_ignores_empty_windows_localappdata(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An empty Windows cache variable falls back to the user's home."""
        monkeypatch.setattr("ontogr_mem.paths.sys.platform", "win32")
        monkeypatch.setenv("HOME", "/home/test")
        monkeypatch.setenv("LOCALAPPDATA", "")
        assert cache_directory() == Path("/home/test/ontogr-mem/cache")


class TestModelCacheDirectory:
    """The model cache honors FastEmbed's environment override."""

    def test_uses_fastembed_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """FastEmbed's cache variable wins over the computed cache path."""
        monkeypatch.setenv("FASTEMBED_CACHE_PATH", "/models")
        assert model_cache_directory() == Path("/models")

    def test_does_not_create_directories(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Resolving the model cache path leaves the directory absent."""
        monkeypatch.setattr("ontogr_mem.paths.sys.platform", "linux")
        monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
        monkeypatch.delenv("FASTEMBED_CACHE_PATH", raising=False)
        expected = tmp_path / "cache" / "ontogr-mem" / "fastembed"
        assert model_cache_directory() == expected
        assert not (tmp_path / "cache").exists()


class TestEnsureDirectory:
    """The directory helper creates missing directories safely."""

    def test_creates_directory_with_restricted_mode(self, tmp_path: Path) -> None:
        """A new POSIX directory gets mode 0o700."""
        path = ensure_directory(tmp_path / "new")
        assert path.is_dir()
        if os.name == "posix":
            assert path.stat().st_mode & 0o777 == 0o700

    def test_keeps_existing_mode(self, tmp_path: Path) -> None:
        """An existing directory keeps its current mode."""
        path = tmp_path / "existing"
        path.mkdir(mode=0o755)
        before = path.stat().st_mode & 0o777
        ensure_directory(path)
        assert path.stat().st_mode & 0o777 == before
