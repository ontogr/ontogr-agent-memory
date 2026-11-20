"""Integration tests for lazy vector store embedder loading."""

from ontogr_mem.vectorstore.errors import EmbedderUnavailableError
from ontogr_mem.vectorstore.memory.backend import InMemoryBackend
from ontogr_mem.vectorstore.models import VectorKind
from ontogr_mem.vectorstore.settings import VectorStoreSettings
from ontogr_mem.vectorstore.source import EmbedderSource
from ontogr_mem.vectorstore.vector_store import VectorStore
from tests.support.embedder import MockEmbedder


class TestLazyEmbedder:
    """Verify lazy embedder loading behavior."""

    async def test_backend_operations_do_not_load_embedder(self) -> None:
        """Allow backend-only operations without loading an embedder."""

        async def unavailable():
            raise EmbedderUnavailableError("not configured", hint="configure embedder")

        store = VectorStore(InMemoryBackend(), EmbedderSource(unavailable))

        assert await store.delete(VectorKind.NODE, ["missing"]) == 0
        assert await store.exists(VectorKind.NODE, ["missing"]) == set()
        assert await store.iter_ids(VectorKind.NODE, after=None, limit=100) == []

    async def test_upsert_raises_when_embedder_is_unavailable(self) -> None:
        """Raise the configured error when upsert needs an unavailable embedder."""

        async def unavailable():
            raise EmbedderUnavailableError("not configured", hint="configure embedder")

        store = VectorStore(InMemoryBackend(), EmbedderSource(unavailable))

        try:
            await store.upsert(VectorKind.NODE, [])
        except EmbedderUnavailableError:
            pass
        else:
            raise AssertionError("upsert should load the configured embedder")

    async def test_upsert_retries_embedder_load_after_failure(self) -> None:
        """Retry embedder loading after an earlier load failure."""
        attempts = 0

        async def load_embedder():
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                raise EmbedderUnavailableError(
                    "temporary failure", hint="configure embedder"
                )
            return MockEmbedder()

        store = VectorStore(InMemoryBackend(), EmbedderSource(load_embedder))

        try:
            await store.upsert(VectorKind.NODE, [])
        except EmbedderUnavailableError:
            pass
        else:
            raise AssertionError("the first load should fail")

        await store.upsert(VectorKind.NODE, [])
        assert attempts == 2

    async def test_lazy_gateway_uses_vector_store_settings(self) -> None:
        """Apply configured gateway limits after lazy embedder loading."""
        settings = VectorStoreSettings(
            batch_size=7, max_attempts=5, attempt_timeout_s=12.0
        )

        async def load_embedder():
            return MockEmbedder()

        store = VectorStore(
            InMemoryBackend(), EmbedderSource(load_embedder), settings=settings
        )

        gateway = await store.embedding_gateway()

        assert gateway.settings == settings
