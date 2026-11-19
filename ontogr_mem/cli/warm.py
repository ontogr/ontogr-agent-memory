"""Warm the local embedding model cache."""

from ontogr_mem.paths import model_cache_directory
from ontogr_mem.vectorstore.builtin.local import LocalEmbedder
from ontogr_mem.vectorstore.builtin.settings import EmbeddingSettings


async def warm_model(settings: EmbeddingSettings | None = None) -> int:
    """Download or validate the configured local embedding model."""
    selected = settings if settings is not None else EmbeddingSettings()
    if selected.provider == "endpoint" or selected.base_url:
        print("Embedding endpoint is configured; there is no local model to download.")
        return 0
    name = selected.fastembed_kwargs.get(
        "model_name", "ibm-granite/granite-embedding-small-english-r2"
    )
    print(f"Embedding model: {name}")
    print(f"Cache: {model_cache_directory()}")
    embedder = await LocalEmbedder.open(selected)
    try:
        print(f"Model ready: {embedder.dim} dimensions.")
    finally:
        await embedder.aclose()
    return 0
