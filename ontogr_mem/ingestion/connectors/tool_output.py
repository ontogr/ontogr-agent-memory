"""Tool-output connector."""

from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from ontogr_mem.data_models import (
    Actor,
    Content,
    EpisodeDraft,
    Kind,
    Role,
    Source,
)


@dataclass(frozen=True, slots=True)
class ToolOutputInput:
    """The result of one tool call."""

    call_id: str
    tool_name: str
    output: Content
    run_at: datetime
    args: dict[str, Any] = field(default_factory=dict)


class ToolOutputConnector:
    """Maps a tool call result to a tool_output episode.

    The tool name is the actor and the call id is the source ref. Arguments go
    in metadata, not in the content.
    """

    def collect(self, item: ToolOutputInput) -> Iterable[EpisodeDraft]:
        """Yield exactly one draft."""
        yield EpisodeDraft(
            kind=Kind.TOOL_OUTPUT,
            occurred_at=item.run_at,
            source=Source(system="agent_tools", ref=item.call_id, connector="tool"),
            actor=Actor(id=item.tool_name, role=Role.TOOL),
            content=item.output,
            metadata={"args": item.args},
        )
