"""Write side of the signals the stages read: feedback, access and session entries.

No other module records these, so callers such as the retrieval layer or the agent
loop use these functions. Ids are ordered by the store's counter, so the order of
writes is the order of ids.
"""

from datetime import datetime

from ontogr_mem.consolidation.models import (
    AccessEvent,
    FeedbackEvent,
    SessionEntry,
)
from ontogr_mem.consolidation.store import ConsStore


async def record_feedback(
    store: ConsStore,
    target_id: str,
    rating: float,
    now: datetime,
    *,
    implicit: bool = False,
) -> FeedbackEvent:
    """Store one rating of ``target_id``, where 0 is bad and 1 is good."""
    event = FeedbackEvent(
        event_id=f"fb-{await store.tick():08d}",
        target_id=target_id,
        rating=rating,
        implicit=implicit,
        created_at=now,
    )
    await store.put(event)
    return event


async def record_access(store: ConsStore, target_id: str, now: datetime) -> AccessEvent:
    """Store that ``target_id`` was read or mentioned."""
    event = AccessEvent(
        event_id=f"ac-{await store.tick():08d}", target_id=target_id, at=now
    )
    await store.put(event)
    return event


async def record_session_entry(
    store: ConsStore,
    session_id: str,
    text: str,
    now: datetime,
    *,
    helpfulness: int = 0,
    confidence: float = 1.0,
) -> SessionEntry:
    """Store one entry of an agent session for reflection."""
    seq = await store.tick()
    entry = SessionEntry(
        entry_id=f"se-{seq:08d}",
        session_id=session_id,
        seq=seq,
        text=text,
        helpfulness=helpfulness,
        confidence=confidence,
        created_at=now,
    )
    await store.put(entry)
    return entry
