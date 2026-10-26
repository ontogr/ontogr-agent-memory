"""Text templates and the text hash."""

import hashlib
import unicodedata
from dataclasses import dataclass
from typing import Final

from ontogr_mem.vectorstore.models import VectorKind


def normalise_text(text: str) -> str:
    """Return the text in NFC with every whitespace run turned into one space."""
    return " ".join(unicodedata.normalize("NFC", text).split())


@dataclass(frozen=True, slots=True)
class Template:
    """A versioned way to turn an object's text into the embedded string.

    Attributes:
        id: Template id, stored with the space.
        version: Bumped when the rendering changes.
        exact: Embed the text as given instead of normalising whitespace.
    """

    id: str
    version: int
    exact: bool = False

    def render(self, text: str) -> str:
        """Return the string that is hashed and embedded."""
        return text if self.exact else normalise_text(text)


TEMPLATES: Final[dict[VectorKind, Template]] = {
    VectorKind.NODE: Template("node.name", 1),
    VectorKind.EDGE: Template("edge.fact", 1),
    VectorKind.EPISODE: Template("episode.text", 1, exact=True),
}


def text_hash(template: Template, rendered: str) -> str:
    """Return the hex SHA-256 over the template id, version and rendered text."""
    payload = f"{template.id}\x00{template.version}\x00{rendered}"
    return hashlib.sha256(payload.encode()).hexdigest()


def edge_text(statement: str, subject: str, relation: str, obj: str) -> str:
    """Return the fact sentence, or `"subject relation object"` when it is empty."""
    sentence = normalise_text(statement)
    if sentence:
        return sentence
    return normalise_text(f"{subject} {relation} {obj}")
