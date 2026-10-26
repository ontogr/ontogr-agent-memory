"""Checks that the installed package reports the project version."""

import tomllib
from pathlib import Path

import ontogr_mem


def test_version_matches_pyproject():
    """The package reports the version declared in pyproject.toml."""
    pyproject = Path(__file__).parents[2] / "pyproject.toml"
    expected = tomllib.loads(pyproject.read_text())["project"]["version"]

    assert ontogr_mem.__version__ == expected
