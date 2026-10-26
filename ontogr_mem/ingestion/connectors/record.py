"""Record connector: tabular rows without an LLM."""

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from ontogr_mem.data_models import Actor, EpisodeDraft, Kind, Role, Source


@dataclass(frozen=True, slots=True)
class RecordInput:
    """One row from a system of record."""

    system: str
    table: str
    primary_key: str
    row: dict[str, Any]
    updated_at: datetime


class RecordConnector:
    """Maps a row to a record episode.

    The primary key is `source.ref`, the update time is `occurred_at` and the
    row stays a dict. A new update time gives a new episode, so the row keeps a
    history.
    """

    def collect(self, item: RecordInput) -> Iterable[EpisodeDraft]:
        """Yield exactly one draft."""
        yield EpisodeDraft(
            kind=Kind.RECORD,
            occurred_at=item.updated_at,
            source=Source(system=item.system, ref=item.primary_key, connector="record"),
            actor=Actor(id=item.system, role=Role.SERVICE),
            content=item.row,
            metadata={"table": item.table},
        )
