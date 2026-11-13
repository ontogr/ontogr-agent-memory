"""Checks the packaged skill files in the wheel that make build writes."""

import zipfile
from pathlib import Path

import pytest

from ontogr_mem.mcp_server.skills import SKILLS_ROOT


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
DIST_ROOT = REPOSITORY_ROOT / "dist"
SKILL_NAMES = ["memory-write", "memory-recall"]


@pytest.fixture
def built_wheel() -> Path:
    """Return the newest wheel in dist/, or skip the test when there is none."""
    wheels = list(DIST_ROOT.glob("ontogr_mem-*.whl"))
    if not wheels:
        pytest.skip("no wheel in dist; run make build")
    return max(wheels, key=lambda wheel: wheel.stat().st_mtime)


class TestWheelSkills:
    """The built wheel ships each packaged skill file byte for byte."""

    @pytest.mark.parametrize("name", SKILL_NAMES)
    def test_ships_skill_file_with_source_bytes(self, built_wheel, name):
        """Each packaged SKILL.md in the wheel matches the source tree."""
        member = f"ontogr_mem/mcp_server/skills/{name}/SKILL.md"
        source_bytes = (SKILLS_ROOT / name / "SKILL.md").read_bytes()

        with zipfile.ZipFile(built_wheel) as wheel:
            assert member in wheel.namelist()
            shipped_bytes = wheel.read(member)

        assert shipped_bytes == source_bytes

    def test_omits_skills_package_init(self, built_wheel):
        """The skills folder ships as data, so the wheel has no __init__.py."""
        with zipfile.ZipFile(built_wheel) as wheel:
            assert "ontogr_mem/mcp_server/skills/__init__.py" not in wheel.namelist()
