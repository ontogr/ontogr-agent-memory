"""Instruction text the MCP server gives to clients."""

INSTRUCTIONS = (
    "If get_skill is listed, call get_skill('memory-write') before saving memory, or "
    "get_skill('memory-recall') before recalling it. A host that supports skills "
    "can also load them itself. Use the tools listed by this server; "
    "unavailable tools are not enabled."
)
