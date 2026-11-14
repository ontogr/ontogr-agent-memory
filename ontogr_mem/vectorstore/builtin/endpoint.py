"""An embedding adapter for an explicit OpenAI-compatible endpoint."""

from collections.abc import Sequence

from openai import (
    APIConnectionError,
    APIError,
    APIStatusError,
    APITimeoutError,
    AsyncOpenAI,
)

from ontogr_mem.vectorstore.builtin.settings import EmbeddingSettings
from ontogr_mem.vectorstore.embedder import EmbeddingPurpose
from ontogr_mem.vectorstore.errors import (
    EmbedderUnavailableError,
    EmbeddingTimeoutError,
    InvalidVectorError,
    PermanentEmbeddingError,
    TransientEmbeddingError,
)
from ontogr_mem.vectorstore.vecmath import to_unit


class EndpointEmbedder:
    """Embed batches through one endpoint with a fixed model and width.

    Attributes:
        model_id: Endpoint model name stored in the embedding space.
        dim: Length of every output vector.
        max_input_tokens: Declared input limit used by the embedding gateway.
    """

    def __init__(
        self,
        client: AsyncOpenAI,
        *,
        model_id: str,
        dim: int,
        max_input_tokens: int,
        owns_client: bool = False,
    ) -> None:
        """Bind a client with known dimensions.

        Args:
            client: SDK client, or an injected compatible test client.
            model_id: Model name passed to the endpoint.
            dim: Expected width, between 1 and 4000.
            max_input_tokens: Positive declared input limit.
            owns_client: Whether aclose also closes this SDK client.
        """
        if not 1 <= dim <= 4000 or max_input_tokens < 1:
            raise ValueError("invalid embedding dimensions or input limit")
        self.model_id = model_id
        self.dim = dim
        self.max_input_tokens = max_input_tokens
        self._client = client
        self._owns_client = owns_client

    @classmethod
    async def open(
        cls,
        settings: EmbeddingSettings,
        *,
        client: AsyncOpenAI | None = None,
    ) -> "EndpointEmbedder":
        """Open the endpoint, discovering its width once when unset.

        Args:
            settings: Explicit model, URL and independent embedding key.
            client: Borrowed client for callers that manage the SDK lifetime. A
                borrowed client keeps the retry and header settings it was built
                with.

        Returns:
            An embedder whose dimensions are ready for a Tenant stack.

        Raises:
            EmbedderUnavailableError: If model or URL is missing.
            PermanentEmbeddingError: If the discovery reply is invalid.
            TransientEmbeddingError: If the endpoint is unavailable.
            EmbeddingTimeoutError: If discovery times out.
        """
        if not settings.model or not settings.base_url:
            raise EmbedderUnavailableError(
                "embedding endpoint model and base URL are required",
                hint="Set MEMORY_EMBEDDING_MODEL and MEMORY_EMBEDDING_BASE_URL.",
            )
        selected = client
        if selected is None:
            selected = AsyncOpenAI(
                base_url=settings.base_url,
                api_key=(
                    settings.api_key.get_secret_value()
                    if settings.api_key is not None
                    else "not-required"
                ),
                # The embedding gateway owns retries; the SDK must not add its own.
                max_retries=0,
            )
            # The SDK fills these from OPENAI_ORG_ID and OPENAI_PROJECT_ID, and
            # this client must not send ambient values.
            selected.organization = None
            selected.project = None
        embedder = cls(
            selected,
            model_id=settings.model,
            dim=settings.dimensions or 1,
            max_input_tokens=settings.max_input_tokens,
            owns_client=client is None,
        )
        try:
            if settings.dimensions is None:
                vectors = await embedder._request(["embedding dimension probe"])
                width = len(vectors[0])
                if not 1 <= width <= 4000:
                    raise PermanentEmbeddingError("unsupported embedding width")
                embedder.dim = width
                to_unit(vectors[0], width)
        except (InvalidVectorError, TypeError, ValueError) as exc:
            await embedder.aclose()
            raise PermanentEmbeddingError("invalid embedding probe vector") from exc
        except BaseException:
            await embedder.aclose()
            raise
        return embedder

    async def embed_batch(
        self, texts: Sequence[str], *, purpose: EmbeddingPurpose
    ) -> list[list[float]]:
        """Return one unit vector per text, in input order.

        The endpoint uses the same request for query and document purposes.

        Args:
            texts: One gateway batch. The gateway handles long-text splitting.
            purpose: Whether the text will be searched or stored.

        Returns:
            Aligned unit vectors, or an empty list for an empty input.

        Raises:
            PermanentEmbeddingError: If input or returned vectors are invalid.
            TransientEmbeddingError: If the endpoint is unavailable.
            EmbeddingTimeoutError: If the endpoint times out.
        """
        if not texts:
            return []
        vectors = await self._request(texts)
        try:
            return [list(to_unit(vector, self.dim)) for vector in vectors]
        except (InvalidVectorError, TypeError, ValueError) as exc:
            raise PermanentEmbeddingError("invalid embedding response vector") from exc

    async def aclose(self) -> None:
        """Close an owned SDK client once; leave a borrowed client open."""
        if self._owns_client:
            self._owns_client = False
            await self._client.close()

    async def _request(self, texts: Sequence[str]) -> list[list[float]]:
        """Map provider failures and validate the ordered response boundary."""
        try:
            response = await self._client.embeddings.create(
                model=self.model_id,
                input=list(texts),
                encoding_format="float",
            )
        except APITimeoutError as exc:
            raise EmbeddingTimeoutError("embedding endpoint timed out") from exc
        except APIConnectionError as exc:
            raise TransientEmbeddingError("embedding endpoint unavailable") from exc
        except APIStatusError as exc:
            if exc.status_code in {408, 429} or exc.status_code >= 500:
                raise TransientEmbeddingError(
                    f"embedding endpoint returned HTTP {exc.status_code}"
                ) from exc
            raise PermanentEmbeddingError(
                f"embedding endpoint returned HTTP {exc.status_code}"
            ) from exc
        except APIError as exc:
            raise PermanentEmbeddingError(
                "invalid embedding endpoint response"
            ) from exc
        try:
            rows = sorted(response.data, key=lambda row: row.index)
            vectors = [list(row.embedding) for row in rows]
        except (AttributeError, TypeError, ValueError) as exc:
            raise PermanentEmbeddingError(
                "invalid embedding endpoint response"
            ) from exc
        if [row.index for row in rows] != list(range(len(texts))):
            raise PermanentEmbeddingError(
                "embedding response indices do not match input"
            )
        return vectors
