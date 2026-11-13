"""Built-in endpoint and CPU embedding providers."""

from ontogr_mem.vectorstore.builtin.endpoint import EndpointEmbedder
from ontogr_mem.vectorstore.builtin.local import LocalEmbedder
from ontogr_mem.vectorstore.builtin.select import select_embedder
from ontogr_mem.vectorstore.builtin.settings import EmbeddingSettings
from ontogr_mem.vectorstore.errors import EmbedderUnavailableError


__all__ = [
    "EmbedderUnavailableError",
    "EmbeddingSettings",
    "EndpointEmbedder",
    "LocalEmbedder",
    "select_embedder",
]
