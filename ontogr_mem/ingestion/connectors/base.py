"""Connector contract."""

from collections.abc import Iterable
from typing import Protocol, TypeVar

from ontogr_mem.data_models import EpisodeDraft


T_contra = TypeVar("T_contra", contravariant=True)


class Connector(Protocol[T_contra]):
    """Maps one typed input to episode drafts.

    A connector sets kind, occurred_at, source, actor, content and metadata. It
    never sets `ingested_at`, `content_hash` or the id, and it does no I/O.
    """

    def collect(self, item: T_contra) -> Iterable[EpisodeDraft]:
        """Map `item` to drafts."""
        ...
