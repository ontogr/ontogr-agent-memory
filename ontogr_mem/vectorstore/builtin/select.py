"""Choose a built-in embedder from independent embedding settings."""

from ontogr_mem.vectorstore.builtin.endpoint import EndpointEmbedder
from ontogr_mem.vectorstore.builtin.local import LocalEmbedder
from ontogr_mem.vectorstore.builtin.settings import EmbeddingSettings
from ontogr_mem.vectorstore.errors import EmbedderUnavailableError


async def select_embedder(
    settings: EmbeddingSettings | None = None,
) -> LocalEmbedder | EndpointEmbedder:
    """Open the configured embedding provider without falling back on failure.

    Args:
        settings: Embedding configuration, or load it from the environment.

    Returns:
        An opened built-in embedder owned by the caller.

    Raises:
        EmbedderUnavailableError: If endpoint settings are incomplete or the
            local provider cannot load.
        EmbeddingError: If the selected provider fails to open.
    """
    selected = settings if settings is not None else EmbeddingSettings()

    if selected.provider == "fastembed":
        return await LocalEmbedder.open(selected)

    has_model = bool(selected.model)
    has_base_url = bool(selected.base_url)
    endpoint_configured = has_model and has_base_url
    if (selected.provider == "endpoint" and not endpoint_configured) or (
        selected.provider == "auto" and has_model != has_base_url
    ):
        raise EmbedderUnavailableError(
            "embedding endpoint configuration is incomplete",
            hint=(
                "Set both MEMORY_EMBEDDING_MODEL and "
                "MEMORY_EMBEDDING_BASE_URL, or choose "
                "MEMORY_EMBEDDING_PROVIDER=fastembed."
            ),
        )

    if endpoint_configured:
        return await EndpointEmbedder.open(selected)
    return await LocalEmbedder.open(selected)
