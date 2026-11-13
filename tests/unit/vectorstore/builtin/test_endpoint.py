"""Endpoint embedder behavior against fake clients and a mock HTTP transport."""

import json
import math
from collections.abc import Callable
from types import SimpleNamespace
from typing import Any, NoReturn

import httpx2
import pytest
from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AuthenticationError,
    BadRequestError,
    InternalServerError,
    NotFoundError,
    RateLimitError,
)
from openai import AsyncOpenAI as RealAsyncOpenAI
from pydantic import SecretStr

from ontogr_mem.vectorstore.builtin.endpoint import EndpointEmbedder
from ontogr_mem.vectorstore.builtin.settings import EmbeddingSettings
from ontogr_mem.vectorstore.errors import (
    EmbedderUnavailableError,
    EmbeddingTimeoutError,
    PermanentEmbeddingError,
    TransientEmbeddingError,
)


MODEL = "test-model"
BASE_URL = "http://embeddings.invalid/v1"
ENDPOINT_URL = "http://embeddings.invalid/v1/embeddings"
ENDPOINT_OPENAI = "ontogr_mem.vectorstore.builtin.endpoint.AsyncOpenAI"


class FakeEmbeddings:
    """Answers `embeddings.create` by calling a script with the request input."""

    def __init__(self, respond: Callable[[list[str]], object]) -> None:
        """Store the script that answers each request."""
        self._respond = respond

    async def create(self, **request: Any) -> object:
        """Return the scripted answer for `request["input"]`, or raise it."""
        return self._respond(request["input"])


class FakeClient:
    """Stands in for `AsyncOpenAI` and counts calls to `close`."""

    def __init__(self, respond: Callable[[list[str]], object]) -> None:
        """Script the embeddings endpoint with `respond`."""
        self.embeddings = FakeEmbeddings(respond)
        self.close_count = 0

    async def close(self) -> None:
        """Count one close call."""
        self.close_count += 1


class ClientFactory:
    """Stands in for `AsyncOpenAI` and returns one prepared client."""

    def __init__(self, client: FakeClient) -> None:
        """Store the client that every call returns."""
        self.client = client

    def __call__(self, **kwargs: Any) -> FakeClient:
        """Return the prepared client, ignoring the constructor keywords."""
        return self.client


def _settings(
    *,
    model: str | None = MODEL,
    base_url: str | None = BASE_URL,
    api_key: SecretStr | None = None,
    dimensions: int | None = 3,
) -> EmbeddingSettings:
    """Build settings that ignore any `.env` file.

    The default dimension of 3 keeps `open` from sending a probe request.
    """
    return EmbeddingSettings(
        model=model,
        base_url=base_url,
        api_key=api_key,
        dimensions=dimensions,
        _env_file=None,
    )


def _embedder(client: FakeClient, dim: int = 3) -> EndpointEmbedder:
    """Build an embedder around `client` without calling `open`."""
    return EndpointEmbedder(client, model_id=MODEL, dim=dim, max_input_tokens=8192)


def _no_request(texts: list[str]) -> NoReturn:
    """Fail when the embedder sends a request the test does not expect."""
    raise AssertionError("no request expected")


def _raises(error: Exception) -> Callable[[list[str]], object]:
    """Return a script that raises `error` for every request."""

    def respond(texts: list[str]) -> NoReturn:
        """Raise the stored error."""
        raise error

    return respond


def _response(*rows: tuple[int, list[float]]) -> SimpleNamespace:
    """Build an embeddings response with rows in the order given."""
    return SimpleNamespace(
        data=[SimpleNamespace(index=index, embedding=vector) for index, vector in rows]
    )


def _request() -> httpx2.Request:
    """Return the request that the error fixtures are attached to."""
    return httpx2.Request("POST", ENDPOINT_URL)


def _status_error(
    error_class: type[APIStatusError], status_code: int
) -> APIStatusError:
    """Build the SDK status error for a response with `status_code`.

    The body is the sentinel string. The message repeats it, so a test fails
    when the endpoint embedder forwards `str(error)` to its caller.
    """
    return error_class(
        f"Error code: {status_code} - sentinel-body",
        response=httpx2.Response(status_code, request=_request()),
        body="sentinel-body",
    )


def _embedding_reply(request: httpx2.Request) -> httpx2.Response:
    """Answer each input text with the vector [1, 2, 2], in request order."""
    inputs = json.loads(request.content)["input"]
    return httpx2.Response(
        200,
        json={
            "object": "list",
            "model": MODEL,
            "usage": {"prompt_tokens": 1, "total_tokens": 1},
            "data": [
                {"object": "embedding", "index": index, "embedding": [1.0, 2.0, 2.0]}
                for index in range(len(inputs))
            ],
        },
    )


def _wire_client(
    monkeypatch: pytest.MonkeyPatch,
    handler: Callable[[httpx2.Request], httpx2.Response],
) -> None:
    """Make `open` build the real SDK client on a mock HTTP transport.

    The handler receives each request as it would go on the wire, so a test
    can check the URL, headers and body that the SDK sends.
    """

    def factory(**kwargs: Any) -> RealAsyncOpenAI:
        """Build the real SDK client with `handler` as its transport."""
        transport = httpx2.MockTransport(handler)
        return RealAsyncOpenAI(
            **kwargs, http_client=httpx2.AsyncClient(transport=transport)
        )

    monkeypatch.setattr(ENDPOINT_OPENAI, factory)


class TestEndpointEmbedder:
    """Behavior of `EndpointEmbedder` against fake endpoints and clients."""

    async def test_empty_batch_returns_no_vectors_without_a_request(self) -> None:
        """An empty batch returns an empty list and never calls the endpoint."""
        embedder = _embedder(FakeClient(_no_request))

        assert await embedder.embed_batch([]) == []

    async def test_orders_rows_by_index_and_normalizes_each_vector(self) -> None:
        """Rows are matched to texts by index and each vector has length one."""
        client = FakeClient(
            lambda texts: _response((1, [0.0, 0.0, 2.0]), (0, [3.0, 4.0, 0.0]))
        )

        vectors = await _embedder(client).embed_batch(["a", "b"])

        assert len(vectors) == 2
        assert vectors[0] == pytest.approx([0.6, 0.8, 0.0])
        assert vectors[1] == pytest.approx([0.0, 0.0, 1.0])

    @pytest.mark.parametrize(
        "rows",
        [
            pytest.param([(0, [1.0, 0.0, 0.0]), (2, [0.0, 1.0, 0.0])], id="missing"),
            pytest.param([(0, [1.0, 0.0, 0.0]), (0, [0.0, 1.0, 0.0])], id="duplicate"),
            pytest.param(
                [(0, [1.0, 0.0, 0.0]), (1, [0.0, 1.0, 0.0]), (2, [0.0, 0.0, 1.0])],
                id="extra_row",
            ),
        ],
    )
    async def test_rejects_rows_whose_indices_are_not_zero_to_n_minus_one(
        self, rows: list[tuple[int, list[float]]]
    ) -> None:
        """Each text needs one row, and the row indices must be 0 to n minus 1."""
        embedder = _embedder(FakeClient(lambda texts: _response(*rows)))

        with pytest.raises(PermanentEmbeddingError):
            await embedder.embed_batch(["a", "b"])

    @pytest.mark.parametrize(
        "vector",
        [
            pytest.param([1.0, 0.0], id="too_short"),
            pytest.param([1.0, 0.0, 0.0, 0.0], id="too_long"),
        ],
    )
    async def test_rejects_a_vector_whose_width_differs_from_dim(
        self, vector: list[float]
    ) -> None:
        """A vector must have exactly `dim` components."""
        embedder = _embedder(FakeClient(lambda texts: _response((0, vector))), dim=3)

        with pytest.raises(PermanentEmbeddingError):
            await embedder.embed_batch(["a"])

    @pytest.mark.parametrize(
        "vector",
        [
            pytest.param([math.nan, 0.0, 0.0], id="nan"),
            pytest.param([math.inf, 0.0, 0.0], id="infinity"),
            pytest.param([0.0, 0.0, 0.0], id="all_zero"),
        ],
    )
    async def test_rejects_non_finite_and_zero_vectors(
        self, vector: list[float]
    ) -> None:
        """A vector with NaN or infinity, or with zero norm, is rejected."""
        embedder = _embedder(FakeClient(lambda texts: _response((0, vector))))

        with pytest.raises(PermanentEmbeddingError):
            await embedder.embed_batch(["a"])

    async def test_embed_batch_rejects_non_numeric_vector_values(self) -> None:
        """A vector whose values are not numbers is a permanent error."""
        vector: list[Any] = ["a", "b", "c"]
        client = FakeClient(lambda texts: _response((0, vector)))

        with pytest.raises(PermanentEmbeddingError):
            await _embedder(client).embed_batch(["a"])

    async def test_configured_dimensions_are_used_without_a_probe(self) -> None:
        """A configured dimension sets `dim` and sends no request."""
        embedder = await EndpointEmbedder.open(
            _settings(dimensions=3), client=FakeClient(_no_request)
        )

        assert embedder.dim == 3

    async def test_probe_vector_sets_the_dimension_when_none_is_configured(
        self,
    ) -> None:
        """Without a configured dimension, the probe response sets `dim`."""
        client = FakeClient(lambda texts: _response((0, [1.0, 2.0, 3.0, 4.0, 5.0])))

        embedder = await EndpointEmbedder.open(
            _settings(dimensions=None), client=client
        )

        assert embedder.dim == 5

    @pytest.mark.parametrize(
        "vector",
        [
            pytest.param([1.0] * 4001, id="over_4000"),
            pytest.param([], id="empty"),
        ],
    )
    async def test_rejects_a_probe_vector_outside_the_valid_width(
        self, vector: list[float]
    ) -> None:
        """Discovery accepts only widths from 1 to 4000."""
        client = FakeClient(lambda texts: _response((0, vector)))

        with pytest.raises(PermanentEmbeddingError):
            await EndpointEmbedder.open(_settings(dimensions=None), client=client)

    async def test_probe_rejects_non_numeric_values_and_closes_owned_client(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A non-numeric probe vector is a permanent error that closes the client."""
        vector: list[Any] = ["a", "b", "c"]
        client = FakeClient(lambda texts: _response((0, vector)))
        monkeypatch.setattr(ENDPOINT_OPENAI, ClientFactory(client))

        with pytest.raises(PermanentEmbeddingError):
            await EndpointEmbedder.open(_settings(dimensions=None))

        assert client.close_count == 1

    async def test_failed_probe_closes_an_owned_client(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A rejected probe raises a permanent error and closes the owned client."""
        client = FakeClient(_raises(_status_error(AuthenticationError, 401)))
        monkeypatch.setattr(ENDPOINT_OPENAI, ClientFactory(client))

        with pytest.raises(PermanentEmbeddingError):
            await EndpointEmbedder.open(_settings(dimensions=None))

        assert client.close_count == 1

    async def test_failed_probe_leaves_a_borrowed_client_open(self) -> None:
        """A client passed in by the caller is never closed by `open`."""
        client = FakeClient(_raises(_status_error(AuthenticationError, 401)))

        with pytest.raises(PermanentEmbeddingError):
            await EndpointEmbedder.open(_settings(dimensions=None), client=client)

        assert client.close_count == 0

    async def test_aclose_leaves_a_borrowed_client_open(self) -> None:
        """Closing the embedder does not close a client it does not own."""
        client = FakeClient(_no_request)
        embedder = await EndpointEmbedder.open(_settings(dimensions=3), client=client)

        await embedder.aclose()

        assert client.close_count == 0

    async def test_owned_client_is_closed_once_when_aclose_repeats(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A second `aclose` does not close the owned client again."""
        client = FakeClient(_no_request)
        monkeypatch.setattr(ENDPOINT_OPENAI, ClientFactory(client))
        embedder = await EndpointEmbedder.open(_settings(dimensions=3))

        await embedder.aclose()
        await embedder.aclose()

        assert client.close_count == 1

    @pytest.mark.parametrize(
        ("make_error", "expected_error"),
        [
            pytest.param(
                lambda: APITimeoutError(request=_request()),
                EmbeddingTimeoutError,
                id="timeout",
            ),
            pytest.param(
                lambda: APIConnectionError(
                    message=f"Could not reach {ENDPOINT_URL}.", request=_request()
                ),
                TransientEmbeddingError,
                id="connection",
            ),
            pytest.param(
                lambda: _status_error(APIStatusError, 408),
                TransientEmbeddingError,
                id="request_timeout_408",
            ),
            pytest.param(
                lambda: _status_error(RateLimitError, 429),
                TransientEmbeddingError,
                id="rate_limit_429",
            ),
            pytest.param(
                lambda: _status_error(InternalServerError, 500),
                TransientEmbeddingError,
                id="server_500",
            ),
            pytest.param(
                lambda: _status_error(InternalServerError, 503),
                TransientEmbeddingError,
                id="unavailable_503",
            ),
            pytest.param(
                lambda: _status_error(BadRequestError, 400),
                PermanentEmbeddingError,
                id="bad_request_400",
            ),
            pytest.param(
                lambda: _status_error(AuthenticationError, 401),
                PermanentEmbeddingError,
                id="unauthorized_401",
            ),
            pytest.param(
                lambda: _status_error(NotFoundError, 404),
                PermanentEmbeddingError,
                id="not_found_404",
            ),
        ],
    )
    async def test_maps_sdk_errors_without_exposing_body_or_url(
        self,
        make_error: Callable[[], Exception],
        expected_error: type[Exception],
    ) -> None:
        """Each SDK error maps to its embedding error and hides body and URL."""
        embedder = _embedder(FakeClient(_raises(make_error())))

        with pytest.raises(expected_error) as failure:
            await embedder.embed_batch(["a"])

        message = str(failure.value)
        assert "sentinel-body" not in message
        assert "embeddings.invalid" not in message

    @pytest.mark.parametrize(
        ("api_key", "expected_authorization"),
        [
            pytest.param(
                SecretStr("sentinel-key"), "Bearer sentinel-key", id="explicit_key"
            ),
            pytest.param(None, "Bearer not-required", id="no_key"),
        ],
    )
    async def test_request_uses_settings_and_ignores_openai_variables(
        self,
        monkeypatch: pytest.MonkeyPatch,
        api_key: SecretStr | None,
        expected_authorization: str,
    ) -> None:
        """The request goes to the configured URL with no ambient OpenAI values."""
        monkeypatch.setenv("OPENAI_API_KEY", "ambient-sentinel-key")
        monkeypatch.setenv("OPENAI_BASE_URL", "http://ambient-sentinel.invalid/v1")
        monkeypatch.setenv("OPENAI_ORG_ID", "ambient-sentinel-org")
        monkeypatch.setenv("OPENAI_PROJECT_ID", "ambient-sentinel-project")
        seen: list[httpx2.Request] = []

        def handler(request: httpx2.Request) -> httpx2.Response:
            """Record the request and answer with the fixed test vector."""
            seen.append(request)
            return _embedding_reply(request)

        _wire_client(monkeypatch, handler)
        embedder = await EndpointEmbedder.open(_settings(api_key=api_key, dimensions=3))

        (vector,) = await embedder.embed_batch(["a"])
        await embedder.aclose()

        (request,) = seen
        assert str(request.url) == f"{BASE_URL}/embeddings"
        assert request.headers["authorization"] == expected_authorization
        assert not any(name.lower().startswith("openai-") for name in request.headers)
        observed = [str(request.url), *request.headers.values()]
        assert not any("ambient-sentinel" in text for text in observed)
        assert vector == pytest.approx([1 / 3, 2 / 3, 2 / 3])

    async def test_sdk_retries_are_disabled(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A 503 reply is sent once, not retried by the SDK, and is transient."""
        seen: list[httpx2.Request] = []

        def handler(request: httpx2.Request) -> httpx2.Response:
            """Record the request and answer with HTTP 503."""
            seen.append(request)
            return httpx2.Response(503, request=request)

        _wire_client(monkeypatch, handler)
        embedder = await EndpointEmbedder.open(_settings(dimensions=3))

        with pytest.raises(TransientEmbeddingError):
            await embedder.embed_batch(["a"])
        await embedder.aclose()

        assert len(seen) == 1

    @pytest.mark.parametrize(
        ("model", "base_url"),
        [
            pytest.param(None, BASE_URL, id="missing_model"),
            pytest.param(MODEL, None, id="missing_base_url"),
        ],
    )
    async def test_missing_model_or_base_url_raises_a_hint_naming_both(
        self, model: str | None, base_url: str | None
    ) -> None:
        """The unavailable error names both environment variables."""
        with pytest.raises(EmbedderUnavailableError) as unavailable:
            await EndpointEmbedder.open(
                _settings(model=model, base_url=base_url),
                client=FakeClient(_no_request),
            )

        assert "MEMORY_EMBEDDING_MODEL" in unavailable.value.hint
        assert "MEMORY_EMBEDDING_BASE_URL" in unavailable.value.hint

    @pytest.mark.parametrize(
        ("dim", "max_input_tokens"),
        [
            pytest.param(0, 8192, id="dim_zero"),
            pytest.param(4001, 8192, id="dim_above_4000"),
            pytest.param(3, 0, id="max_input_tokens_zero"),
        ],
    )
    def test_constructor_rejects_dim_or_token_limit_out_of_range(
        self, dim: int, max_input_tokens: int
    ) -> None:
        """`dim` must be 1 to 4000 and `max_input_tokens` must be at least 1."""
        with pytest.raises(ValueError):
            EndpointEmbedder(
                FakeClient(_no_request),
                model_id=MODEL,
                dim=dim,
                max_input_tokens=max_input_tokens,
            )
