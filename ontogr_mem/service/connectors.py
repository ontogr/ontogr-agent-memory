"""Connectors for service-created episode drafts."""

from collections.abc import Iterable
from typing import Any

from ontogr_mem.data_models import EpisodeDraft, Kind
from ontogr_mem.ingestion.connectors import (
    Connector,
    default_registry,
)


class DraftConnector(Connector[EpisodeDraft]):
    """Pass a completed draft through ingestion unchanged."""

    def collect(self, item: EpisodeDraft) -> Iterable[EpisodeDraft]:
        """Yield the draft supplied by the service."""
        yield item


def draft_registry() -> dict[Kind, Connector[Any]]:
    """Return a registry that accepts service-built drafts for every kind."""
    return default_registry(overrides={kind: DraftConnector() for kind in Kind})
