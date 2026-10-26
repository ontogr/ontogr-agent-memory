"""Normalization, content hashing, deterministic ids and sealing."""

import copy
import hashlib
import json
import unicodedata
from datetime import datetime

from ontogr_mem.data_models import (
    Content,
    Episode,
    EpisodeDraft,
    stable_id,
    to_micros,
)
from ontogr_mem.ingestion.validate import check_draft


def normalize(content: Content) -> str:
    """Return the normalized form of the content that is hashed.

    Text is NFC, LF newlines and trimmed. Structured payloads become canonical
    JSON (sorted keys, no whitespace).
    """
    if isinstance(content, str):
        text = unicodedata.normalize("NFC", content)
        return text.replace("\r\n", "\n").replace("\r", "\n").strip()
    canonical = json.dumps(
        content, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )
    return unicodedata.normalize("NFC", canonical)


def content_hash(content: Content) -> str:
    """Return the SHA-256 of the normalized content.

    The payload type is part of the hash, so the text "{}" and the empty object
    do not collide.
    """
    tag = "text" if isinstance(content, str) else "json"
    return hashlib.sha256(f"{tag}:{normalize(content)}".encode()).hexdigest()


def episode_id(
    source_system: str, source_ref: str, occurred_at: datetime, digest: str
) -> str:
    """Return the deterministic id of an episode.

    The id covers the source, the occurrence time to the microsecond and the
    content hash. `ingested_at` is not part of it. Delivering the same event
    again maps to the same episode, while the same payload at another time stays
    distinct.
    """
    return stable_id(source_system, source_ref, str(to_micros(occurred_at)), digest)


def seal(draft: EpisodeDraft, *, ingested_at: datetime) -> Episode:
    """Turn a draft into an `Episode`.

    Text content is stored normalized. Structured content and metadata are
    copied, so later changes by the caller cannot alter the episode.

    Raises:
        DraftValidationError: If the draft is invalid.
    """
    check_draft(draft)
    digest = content_hash(draft.content)
    content: Content = (
        normalize(draft.content)
        if isinstance(draft.content, str)
        else copy.deepcopy(draft.content)
    )
    return Episode(
        id=episode_id(draft.source.system, draft.source.ref, draft.occurred_at, digest),
        kind=draft.kind,
        occurred_at=draft.occurred_at,
        ingested_at=ingested_at,
        source=draft.source,
        actor=draft.actor,
        content=content,
        content_hash=digest,
        metadata=copy.deepcopy(draft.metadata),
    )
