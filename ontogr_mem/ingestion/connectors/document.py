"""Document connector: one document, one episode per non-empty part."""

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime

from ontogr_mem.data_models import (
    Actor,
    EpisodeDraft,
    Kind,
    Role,
    Source,
    stable_id,
)


def doc_id(uri: str) -> str:
    """Return the document id: a hash of the uri."""
    return stable_id(uri)


@dataclass(frozen=True, slots=True)
class DocumentInput:
    """An already extracted and split document.

    Loading and splitting happen before this connector; it only maps parts.
    """

    uri: str
    parts: Sequence[str]
    mtime: datetime
    title: str = ""


class DocumentConnector:
    """Maps a document to one episode per non-empty part.

    All parts share the document's source system and `occurred_at`.
    `chunk_index` is the part's position in the input, so empty parts leave gaps
    and the others keep their numbers.
    """

    def collect(self, item: DocumentInput) -> Iterable[EpisodeDraft]:
        """Yield one draft per non-empty part."""
        document_id = doc_id(item.uri)
        for index, part in enumerate(item.parts):
            if not part.strip():
                continue
            yield EpisodeDraft(
                kind=Kind.DOCUMENT,
                occurred_at=item.mtime,
                source=Source(
                    system="document",
                    ref=f"{item.uri}#chunk-{index}",
                    connector="document",
                    description=item.title,
                ),
                actor=Actor(id="document", role=Role.SYSTEM),
                content=part,
                metadata={"doc_id": document_id, "chunk_index": index},
            )
