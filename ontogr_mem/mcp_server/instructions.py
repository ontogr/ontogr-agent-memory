"""Instruction text the MCP server gives to clients."""

from pathlib import Path


INSTRUCTIONS: str = (
    Path(__file__)
    .with_name("wording")
    .joinpath("instructions.txt")
    .read_text(encoding="utf-8")
    .strip()
)
