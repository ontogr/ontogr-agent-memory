"""Chat connector: one message, one episode."""

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime

from ontogr_mem.data_models import Actor, EpisodeDraft, Kind, Role, Source


@dataclass(frozen=True, slots=True)
class ChatMessage:
    """One chat message."""

    message_id: str
    session_id: str
    author_id: str
    role: Role
    text: str
    sent_at: datetime
    author_name: str = ""


class ChatConnector:
    """Maps a chat message to a message episode.

    The session goes in `source.description` and metadata, never into the
    content. The speaker is the actor, never a "Speaker: text" prefix.
    """

    def collect(self, item: ChatMessage) -> Iterable[EpisodeDraft]:
        """Yield exactly one draft."""
        yield EpisodeDraft(
            kind=Kind.MESSAGE,
            occurred_at=item.sent_at,
            source=Source(
                system="chat",
                ref=item.message_id,
                connector="chat",
                description=f"session {item.session_id}",
            ),
            actor=Actor(
                id=item.author_id, role=item.role, display_name=item.author_name
            ),
            content=item.text,
            metadata={"session_id": item.session_id},
        )
