"""Choose a built-in embedder from independent embedding settings."""

from ontogr_mem.vectorstore.builtin.endpoint import EndpointEmbedder
from ontogr_mem.vectorstore.builtin.local import LocalEmbedder
from ontogr_mem.vectorstore.builtin.settings import EmbeddingSettings
from ontogr_mem.vectorstore.embedder import Embedder


async def select_embedder(settings: EmbeddingSettings | None = None) -> Embedder:
    """Open an endpoint when configured, otherwise open the local CPU model.

    Args:
        settings: Embedding configuration, or load it from the environment.

    Returns:
        An opened built-in embedder owned by the caller.

    Raises:
        EmbedderUnavailableError: If no local provider can load.
        EmbeddingError: If a selected endpoint fails to open.
    """
    selected = settings if settings is not None else EmbeddingSettings()
    if selected.model and selected.base_url:
        return await EndpointEmbedder.open(selected)
    return await LocalEmbedder.open()
