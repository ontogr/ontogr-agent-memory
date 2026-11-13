"""Instruction text the MCP server gives to clients."""

INSTRUCTIONS = (
    "Call get_skill('memory-write') before saving memory, or "
    "get_skill('memory-recall') before recalling it. "
    "Use the tools listed by this server; unavailable tools are not enabled."
)
