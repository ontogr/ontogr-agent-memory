"""Choose a built-in embedder from independent embedding settings."""

from ontogr_mem.vectorstore.builtin.endpoint import EndpointEmbedder
from ontogr_mem.vectorstore.builtin.local import LocalEmbedder
from ontogr_mem.vectorstore.builtin.settings import EmbeddingSettings
from ontogr_mem.vectorstore.errors import EmbedderUnavailableError


def choose_provider(settings: EmbeddingSettings) -> str:
    """Return the configured provider without opening an embedder."""
    has_model = bool(settings.model)
    has_base_url = bool(settings.base_url)
    if (settings.provider == "endpoint" and not (has_model and has_base_url)) or (
        settings.provider == "auto" and has_model != has_base_url
    ):
        raise EmbedderUnavailableError(
            "embedding endpoint configuration is incomplete",
            hint=(
                "Set both MEMORY_EMBEDDING_MODEL and "
                "MEMORY_EMBEDDING_BASE_URL, or choose "
                "MEMORY_EMBEDDING_PROVIDER=fastembed."
            ),
        )
    if settings.provider == "fastembed" or not (has_model and has_base_url):
        return "fastembed"
    return "endpoint"


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

    provider = choose_provider(selected)
    if provider == "fastembed":
        return await LocalEmbedder.open(selected)
    return await EndpointEmbedder.open(selected)
