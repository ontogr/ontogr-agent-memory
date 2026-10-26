"""Episode types: the draft a connector yields and the stored episode."""

import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import Any, Protocol


Content = str | dict[str, Any]
"""Episode content: text, or a structured payload kept as is.

Structured payloads stay dicts end to end because flattening them to prose would
lose the field structure that later extraction relies on.
"""


class Kind(StrEnum):
    """What an episode represents.

    The kind selects the connector that produced the episode and, later, the
    extraction prompt. An unknown kind must raise.
    """

    MESSAGE = "message"
    DOCUMENT = "document"
    TOOL_OUTPUT = "tool_output"
    EVENT = "event"
    RECORD = "record"


class Role(StrEnum):
    """Who produced an episode."""

    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    TOOL = "tool"
    SERVICE = "service"


class TemporalStatus(StrEnum):
    """Where an episode stands in temporal reconciliation."""

    PENDING = "pending"
    COMMITTED = "committed"
    RETRACTED = "retracted"


@dataclass(frozen=True, slots=True)
class Source:
    """Where an episode came from.

    Attributes:
        system: Originating system, such as "chat" or "stripe".
        ref: Id of the item in that system. With `system` it is the identity.
        connector: Name of the connector that mapped the item.
        description: Free text such as the chat session. It is never hashed.
    """

    system: str
    ref: str
    connector: str
    description: str = ""

    @property
    def id(self) -> str:
        """Identity of the source inside a tenant."""
        return f"{self.system}|{self.ref}"


@dataclass(frozen=True, slots=True)
class Actor:
    """The speaker or producer of an episode, kept apart from the content."""

    id: str
    role: Role
    display_name: str = ""


@dataclass(frozen=True, slots=True)
class EpisodeDraft:
    """What a connector yields.

    Connectors never set `ingested_at`, `content_hash` or the id. The pipeline
    sets them when it seals the draft.

    Attributes:
        kind: What the episode represents.
        occurred_at: When the event happened. Must be timezone-aware.
        source: Where the episode came from.
        actor: Who produced it.
        content: The text or structured payload.
        metadata: Free-form extras that are not hashed. Envelope field names
            are rejected.
    """

    kind: Kind
    occurred_at: datetime
    source: Source
    actor: Actor
    content: Content
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class Episode:
    """A sealed episode.

    Attributes:
        id: Deterministic id built from the source, time and content hash.
        kind: What the episode represents.
        occurred_at: When the event happened.
        ingested_at: When ingestion first stored it. Not part of the id.
        source: Where the episode came from.
        actor: Who produced it.
        content: The normalized text or a copy of the structured payload.
        content_hash: Hash of the normalized content. A store returns None
            until ingestion has recorded it, which marks the episode as fully
            ingested.
        metadata: Free-form extras.
        temporal_status: Set only by temporal reconciliation.
        temporal_committed_hash: Set only by temporal reconciliation.
    """

    id: str
    kind: Kind
    occurred_at: datetime
    ingested_at: datetime
    source: Source
    actor: Actor
    content: Content
    content_hash: str | None
    metadata: dict[str, Any] = field(default_factory=dict)
    temporal_status: TemporalStatus | None = None
    temporal_committed_hash: str | None = None

    @property
    def text(self) -> str:
        """The episode content as one string.

        Structured content becomes the canonical JSON that the graph stores,
        with sorted keys, compact separators and no ASCII escaping.
        """
        if isinstance(self.content, str):
            return self.content
        return json.dumps(
            self.content, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        )


@dataclass(frozen=True, slots=True)
class EpisodeMeta:
    """Session and fact references stored with an episode in its tenant graph.

    Attributes:
        episode_id: Id of the episode that owns the metadata.
        session_id: Session that contains the episode, when known.
        thread: Thread that contains the episode, when known.
        subject_ref: External subject reference, when known.
        fact_ids: Ordered ids of the facts associated with the episode.
    """

    episode_id: str
    session_id: str | None = None
    thread: str | None = None
    subject_ref: str | None = None
    fact_ids: tuple[str, ...] = ()


class PreviousEpisodes(Protocol):
    """Reads the episodes that came before a given episode in the same tenant."""

    async def before(self, episode: Episode, limit: int) -> Sequence[Episode]:
        """Return up to `limit` earlier, fully ingested episodes, oldest first."""
        ...
