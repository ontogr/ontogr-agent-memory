"""Event connector."""

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from ontogr_mem.data_models import Actor, EpisodeDraft, Kind, Role, Source


@dataclass(frozen=True, slots=True)
class EventInput:
    """A provider event."""

    provider: str
    event_id: str
    event_time: datetime
    payload: dict[str, Any]


class EventConnector:
    """Maps a provider event to an event episode.

    `occurred_at` is the event time, never the ingest time. The raw payload is
    the content, including any `type` field, so the event type is part of the
    content hash.
    """

    def collect(self, item: EventInput) -> Iterable[EpisodeDraft]:
        """Yield exactly one draft."""
        yield EpisodeDraft(
            kind=Kind.EVENT,
            occurred_at=item.event_time,
            source=Source(system=item.provider, ref=item.event_id, connector="event"),
            actor=Actor(id=item.provider, role=Role.SERVICE),
            content=item.payload,
        )
