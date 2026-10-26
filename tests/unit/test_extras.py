"""Tests for lazy import of optional dependencies."""

import json
import sys

import pytest

from ontogr_mem._extras import import_extra
from ontogr_mem.errors import MissingExtraError


class TestImportExtra:
    """Behavior of import_extra."""

    def test_returns_an_installed_module(self):
        """An importable module is returned as is."""
        assert import_extra("json", "ignored") is json

    def test_raises_install_hint_when_module_is_missing(self):
        """A missing module becomes MissingExtraError chained from ImportError."""
        with pytest.raises(MissingExtraError) as excinfo:
            import_extra("no_such_module_xyz", "falkordb")

        assert excinfo.value.extra == "falkordb"
        assert isinstance(excinfo.value.__cause__, ImportError)

    def test_does_not_hide_a_broken_installed_module(self, tmp_path, monkeypatch):
        """A module whose own dependency is missing keeps its original error."""
        (tmp_path / "broken_extra_pkg.py").write_text("import missing_dep_xyz\n")
        monkeypatch.syspath_prepend(str(tmp_path))
        monkeypatch.delitem(sys.modules, "broken_extra_pkg", raising=False)

        with pytest.raises(ModuleNotFoundError) as excinfo:
            import_extra("broken_extra_pkg", "falkordb")

        assert excinfo.value.name == "missing_dep_xyz"
