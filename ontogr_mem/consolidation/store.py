"""Port of the consolidation document store, and how each document kind is keyed.

Every document is a typed model stored under its kind. A kind declares the scalar
fields it can be looked up by, and a lookup on any other field is refused. The
store is bound to one tenant, so no method takes a scope.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Protocol, TypeVar

from pydantic import BaseModel

from ontogr_mem.consolidation.errors import StoreError
from ontogr_mem.consolidation.models import (
    AccessEvent,
    ArchiveRecord,
    Bucket,
    Community,
    FeedbackEvent,
    FeedbackWeight,
    Proposal,
    Score,
    SessionEntry,
    Watermark,
)


M = TypeVar("M", bound=BaseModel)


@dataclass(frozen=True, slots=True)
class Kind:
    """How one model class is stored.

    Attributes:
        label: Node label of the kind in the graph backends.
        id_field: Field that identifies a document of the kind.
        fields: Scalar fields a lookup may name.
    """

    label: str
    id_field: str
    fields: tuple[str, ...] = ()


KINDS: dict[type[BaseModel], Kind] = {
    Watermark: Kind("ConsWatermark", "key"),
    FeedbackEvent: Kind("ConsFeedbackEvent", "event_id", ("applied", "target_id")),
    FeedbackWeight: Kind("ConsFeedbackWeight", "target_id"),
    AccessEvent: Kind("ConsAccessEvent", "event_id", ("target_id",)),
    SessionEntry: Kind("ConsSessionEntry", "entry_id", ("session_id",)),
    Proposal: Kind("ConsProposal", "proposal_id", ("status", "session_id")),
    Score: Kind("ConsScore", "target_id", ("kind",)),
    ArchiveRecord: Kind("ConsArchiveRecord", "target_id", ("target_kind", "state")),
    Community: Kind("ConsCommunity", "community_id"),
    Bucket: Kind("ConsBucket", "bucket_id", ("level",)),
}


def kind_of(model: type[BaseModel]) -> Kind:
    """Return the storage kind of ``model``.

    Raises:
        StoreError: If the store does not keep documents of that class.
    """
    try:
        return KINDS[model]
    except KeyError:
        raise StoreError(f"unknown model {model.__name__!r}") from None


def check_field(model: type[BaseModel], field: str) -> None:
    """Refuse a lookup on a field the model did not declare.

    Raises:
        StoreError: If ``field`` is not a lookup field of ``model``.
    """
    if field not in kind_of(model).fields:
        raise StoreError(f"{model.__name__!r} has no indexed field {field!r}")


def field_text(value: object) -> str:
    """Return how a lookup value is compared: booleans and enums as plain text."""
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(getattr(value, "value", value))


class ConsStore(Protocol):
    """Document store of one tenant's consolidation jobs."""

    async def tick(self) -> int:
        """Return the next value of the tenant's counter: 1, 2, 3, and so on."""
        ...

    async def get(self, model: type[M], doc_id: str) -> M | None:
        """Return one document, or None when it is not stored."""
        ...

    async def put(self, obj: BaseModel) -> None:
        """Write one document, replacing any stored under the same id."""
        ...

    async def delete(self, model: type[BaseModel], doc_id: str) -> None:
        """Remove one document. Removing a missing document is not an error."""
        ...

    async def find(self, model: type[M], field: str, value: object) -> list[M]:
        """Return the documents whose lookup ``field`` equals ``value``, by id.

        Raises:
            StoreError: If ``field`` is not a lookup field of ``model``.
        """
        ...

    async def list(self, model: type[M]) -> list[M]:
        """Return every document of ``model``, ordered by id."""
        ...

    async def acquire(self, holder: str, now: datetime, ttl_s: float) -> bool:
        """Take the lease if it is free, expired, or already held by ``holder``.

        Returns:
            True when ``holder`` holds the lease afterwards. False when another
            holder keeps it, or when the store could not decide after retrying.
        """
        ...

    async def request_rerun(self) -> None:
        """Ask the current holder to run once more after it finishes."""
        ...

    async def release(self, holder: str) -> bool:
        """Give the lease back if ``holder`` holds it.

        Returns:
            True when ``holder`` held the lease and a rerun was requested while
            it was held. False otherwise. The lease is released whenever
            ``holder`` held it.
        """
        ...
