"""CPU-only local embeddings with a lazily imported ONNX model."""

import asyncio
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING

from ontogr_mem._extras import import_extra
from ontogr_mem.vectorstore.errors import (
    EmbedderUnavailableError,
    EmbeddingError,
    PermanentEmbeddingError,
)
from ontogr_mem.vectorstore.vecmath import to_unit


if TYPE_CHECKING:
    from fastembed import TextEmbedding


class LocalEmbedder:
    """Embed English text with Granite on CPU.

    Attributes:
        model_id: Native fastembed model identity stored with each space.
        dim: Vector width, fixed at 384.
        max_input_tokens: Tokenizer limit, fixed at 8192. Longer inputs truncate.
    """

    model_id = "ibm-granite/granite-embedding-small-english-r2"
    dim = 384
    max_input_tokens = 8192

    def __init__(self, model: "TextEmbedding") -> None:
        """Bind a loaded CPU model.

        Args:
            model: Native fastembed model loaded by open, or a test substitute.
        """
        self._model: TextEmbedding | None = model
        # A cancelled caller cannot stop a native call that is already running.
        # One worker thread runs batches in submission order, so a later batch
        # waits for an abandoned one instead of entering the model beside it.
        self._executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="local-embedder"
        )

    @classmethod
    async def open(cls) -> "LocalEmbedder":
        """Import and load the CPU model in a worker thread.

        The first open may download model files. fastembed uses
        FASTEMBED_CACHE_PATH, or fastembed_cache in the system temp directory.

        Returns:
            A local embedder ready for a Tenant stack.

        Raises:
            EmbedderUnavailableError: If the extra or model cannot load.
        """

        def load() -> "TextEmbedding":
            module = import_extra("fastembed", "fastembed")
            return module.TextEmbedding(
                model_name=cls.model_id,
                providers=["CPUExecutionProvider"],
                lazy_load=False,
            )

        try:
            model = await asyncio.to_thread(load)
        except Exception as exc:
            raise EmbedderUnavailableError(
                "local embedding model could not load",
                hint=(
                    'Install "ontogr-mem[fastembed]" on a supported platform, '
                    "or set MEMORY_EMBEDDING_MODEL and MEMORY_EMBEDDING_BASE_URL. "
                    "Check model cache access and download connectivity."
                ),
            ) from exc
        return cls(model)

    async def embed_batch(self, texts: Sequence[str]) -> list[list[float]]:
        """Return aligned unit vectors without blocking the event loop.

        Batches run one at a time on a single worker thread. If the awaiting
        caller is cancelled, its batch keeps running on that thread, and later
        batches wait for it to finish.

        Args:
            texts: One gateway batch. The tokenizer truncates at 8192 tokens.

        Returns:
            One unit vector per text, or an empty list for empty input.

        Raises:
            PermanentEmbeddingError: If the model fails or returns bad vectors.
            EmbedderUnavailableError: If this embedder has been closed.
        """
        # Read the model and submit the batch without yielding to the loop. This
        # keeps aclose from shutting the executor down between the check and the
        # submission.
        model = self._model
        if model is None:
            raise EmbedderUnavailableError(
                "local embedder is closed", hint="Open a new LocalEmbedder."
            )
        if not texts:
            return []
        documents = list(texts)

        def embed() -> list[list[float]]:
            vectors = [
                list(to_unit(vector.tolist(), self.dim))
                for vector in model.embed(documents, batch_size=32, parallel=None)
            ]
            if len(vectors) != len(documents):
                raise PermanentEmbeddingError("local embedding count mismatch")
            return vectors

        try:
            return await asyncio.get_running_loop().run_in_executor(
                self._executor, embed
            )
        except EmbeddingError:
            raise
        except Exception as exc:
            raise PermanentEmbeddingError("local embedding failed") from exc

    async def aclose(self) -> None:
        """Close the embedder and wait for batches already submitted to finish.

        Batches started after this call raise EmbedderUnavailableError. Batches
        submitted before it, queued or running, still finish and return. The
        wait runs in a worker thread, so the event loop stays free. Calling this
        method again is safe.
        """
        self._model = None
        await asyncio.to_thread(self._executor.shutdown, wait=True)
