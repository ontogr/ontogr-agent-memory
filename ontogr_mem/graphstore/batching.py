"""Chunked writes with an item-by-item fallback, shared by every backend."""

import logging
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from ontogr_mem.graphstore.errors import StoreQueryError
from ontogr_mem.graphstore.types import WriteError, WriteResult


log = logging.getLogger(__name__)

Item = dict[str, Any]


@dataclass(frozen=True)
class BatchReply:
    """What one statement reports back.

    Attributes:
        ok: Indexes (`i` of the batch rows) the statement wrote or matched.
        created: Nodes or edges the server created.
    """

    ok: Sequence[int]
    created: int


@dataclass
class BatchOutcome:
    """Totals over every chunk of one write."""

    ok: set[int] = field(default_factory=set)
    created: int = 0
    errors: list[WriteError] = field(default_factory=list)
    fell_back: bool = False


async def write_batches(
    outcome: BatchOutcome,
    items: Sequence[Item],
    size: int,
    send: Callable[[list[Item]], Awaitable[BatchReply]],
) -> None:
    """Send `items` in chunks of `size` and add the results to `outcome`.

    A chunk that the server rejects is sent again one item at a time, so one bad
    item does not lose the others. Items that still fail are reported by index.
    Failures that are not a rejected statement (an unreachable server, an open
    breaker) propagate.
    """

    async def attempt(chunk: list[Item]) -> None:
        reply = await send(chunk)
        outcome.ok.update(reply.ok)
        outcome.created += reply.created

    for start in range(0, len(items), size):
        chunk = list(items[start : start + size])
        try:
            await attempt(chunk)
        except StoreQueryError as batch_exc:
            if len(chunk) == 1:
                outcome.errors.append(WriteError(chunk[0]["i"], str(batch_exc)))
                continue
            outcome.fell_back = True
            log.warning("batch write failed (%s); writing item by item", batch_exc)
            for item in chunk:
                try:
                    await attempt([item])
                except StoreQueryError as exc:
                    outcome.errors.append(WriteError(item["i"], str(exc)))


def to_write_result(attempted: int, outcome: BatchOutcome) -> WriteResult:
    """Build the `WriteResult` for a finished write."""
    failed = {e.index for e in outcome.errors}
    missing = tuple(
        i for i in range(attempted) if i not in outcome.ok and i not in failed
    )
    return WriteResult(
        attempted,
        len(outcome.ok),
        outcome.created,
        outcome.fell_back,
        tuple(outcome.errors),
        missing,
    )
