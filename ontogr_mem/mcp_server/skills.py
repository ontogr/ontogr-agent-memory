"""Packaged skill files that the get_skill tool reads."""

from pathlib import Path
from typing import Literal

from ontogr_mem.service.errors import InvalidArgumentError


SKILLS_ROOT = Path(__file__).with_name("skills").resolve()


async def get_skill(name: Literal["memory-write", "memory-recall"]) -> str:
    """Read memory-write before saving facts, or memory-recall before recall.

    Returns the complete packaged skill text. It does not search or write memory.
    """
    if name not in {"memory-write", "memory-recall"}:
        raise InvalidArgumentError("unknown skill name")
    path = (SKILLS_ROOT / name / "SKILL.md").resolve()
    if not path.is_relative_to(SKILLS_ROOT):
        raise InvalidArgumentError("skill path is outside the packaged directory")
    return path.read_text(encoding="utf-8")
