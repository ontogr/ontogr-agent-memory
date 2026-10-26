"""In-memory twin of the consolidation store, for one tenant."""

from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from typing import TypeVar

from pydantic import BaseModel

from ontogr_mem.consolidation.store import check_field, field_text, kind_of


M = TypeVar("M", bound=BaseModel)
_EPOCH = datetime(1970, 1, 1, tzinfo=UTC)


@dataclass(frozen=True, slots=True)
class _Lease:
    """Holder, expiry and rerun flag of the tenant's one lease."""

    holder: str = ""
    expires: datetime = _EPOCH
    rerun: bool = False


class MemoryConsStore:
    """Consolidation store of one tenant, kept in memory.

    Documents are stored as JSON, so every read returns a fresh object. Each method
    runs without awaiting inside its body, so concurrent callers never interleave
    within one call.
    """

    def __init__(self) -> None:
        """Create an empty store."""
        self._docs: dict[type[BaseModel], dict[str, str]] = {}
        self._lease = _Lease()
        self._ticks = 0

    async def tick(self) -> int:
        """Return the next value of the tenant's counter: 1, 2, 3, and so on."""
        self._ticks += 1
        return self._ticks

    async def get(self, model: type[M], doc_id: str) -> M | None:
        """Return one document, or None when it is not stored."""
        kind_of(model)
        raw = self._docs.get(model, {}).get(doc_id)
        return model.model_validate_json(raw) if raw is not None else None

    async def put(self, obj: BaseModel) -> None:
        """Write one document, replacing any stored under the same id."""
        kind = kind_of(type(obj))
        bucket = self._docs.setdefault(type(obj), {})
        bucket[str(getattr(obj, kind.id_field))] = obj.model_dump_json()

    async def delete(self, model: type[BaseModel], doc_id: str) -> None:
        """Remove one document. Removing a missing document is not an error."""
        kind_of(model)
        self._docs.get(model, {}).pop(doc_id, None)

    async def find(self, model: type[M], field: str, value: object) -> list[M]:
        """Return the documents whose ``field`` equals ``value``, ordered by id."""
        check_field(model, field)
        want = field_text(value)
        return [
            doc
            for doc in await self.list(model)
            if field_text(getattr(doc, field)) == want
        ]

    async def acquire(self, holder: str, now: datetime, ttl_s: float) -> bool:
        """Take the lease if it is free, expired, or already held by ``holder``."""
        lease = self._lease
        if lease.holder not in ("", holder) and lease.expires > now:
            return False
        self._lease = replace(
            lease, holder=holder, expires=now + timedelta(seconds=ttl_s)
        )
        return True

    async def request_rerun(self) -> None:
        """Ask the current holder to run once more after it finishes."""
        self._lease = replace(self._lease, rerun=True)

    async def release(self, holder: str) -> bool:
        """Give the lease back if ``holder`` holds it; True if a rerun was asked."""
        if self._lease.holder != holder:
            return False
        rerun = self._lease.rerun
        self._lease = _Lease()
        return rerun

    async def list(self, model: type[M]) -> list[M]:
        """Return every document of ``model``, ordered by id."""
        kind_of(model)
        docs = self._docs.get(model, {})
        return [model.model_validate_json(docs[i]) for i in sorted(docs)]
