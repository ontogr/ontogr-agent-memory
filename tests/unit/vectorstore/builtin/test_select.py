"""Endpoint or local embedder selection in `select_embedder`."""

import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import httpx2
import openai
import pytest

from ontogr_mem.vectorstore.builtin.endpoint import EndpointEmbedder
from ontogr_mem.vectorstore.builtin.local import LocalEmbedder
from ontogr_mem.vectorstore.builtin.select import select_embedder
from ontogr_mem.vectorstore.builtin.settings import EmbeddingSettings
from ontogr_mem.vectorstore.errors import (
    EmbedderUnavailableError,
    PermanentEmbeddingError,
)


EMBEDDING_VARIABLES = (
    "MEMORY_EMBEDDING_MODEL",
    "MEMORY_EMBEDDING_BASE_URL",
    "MEMORY_EMBEDDING_API_KEY",
    "MEMORY_EMBEDDING_DIMENSIONS",
    "MEMORY_EMBEDDING_MAX_INPUT_TOKENS",
)
PROBE_VECTOR = [0.1, 0.2, 0.3]


@pytest.fixture(autouse=True)
def clean_embedding_environment(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Run each test in an empty directory with no embedding variables set.

    A `.env` file in the working directory and variables exported in the shell
    both change which embedder is selected, so both are removed here.
    """
    monkeypatch.chdir(tmp_path)
    for name in EMBEDDING_VARIABLES:
        monkeypatch.delenv(name, raising=False)


class FakeTextEmbedding:
    """Stands in for `fastembed.TextEmbedding` and counts constructions."""

    constructions = 0

    def __init__(self, model_name: str, providers: list[str], lazy_load: bool) -> None:
        """Count the construction without loading a model."""
        FakeTextEmbedding.constructions += 1


@pytest.fixture
def fake_fastembed(monkeypatch: pytest.MonkeyPatch) -> type[FakeTextEmbedding]:
    """Replace the `fastembed` module with a stub and reset the construction count."""
    # Reset per test so a count from an earlier test cannot leak into this one.
    monkeypatch.setattr(FakeTextEmbedding, "constructions", 0)
    # The stub keeps the real package, and the network probes its imports
    # perform, from loading.
    stub = ModuleType("fastembed")
    stub.TextEmbedding = FakeTextEmbedding
    monkeypatch.setitem(sys.modules, "fastembed", stub)
    return FakeTextEmbedding


class FakeEmbeddings:
    """The `embeddings` attribute of a fake OpenAI client."""

    def __init__(self, error: Exception | None) -> None:
        """Keep the error that every `create` call raises, if any."""
        self.error = error

    async def create(self, **request: Any) -> SimpleNamespace:
        """Return one probe vector per input text, or raise the stored error."""
        if self.error is not None:
            raise self.error
        return SimpleNamespace(
            data=[
                SimpleNamespace(index=index, embedding=PROBE_VECTOR)
                for index in range(len(request["input"]))
            ]
        )


class FakeAsyncOpenAI:
    """Stands in for `openai.AsyncOpenAI` with a fake embeddings endpoint."""

    def __init__(self, error: Exception | None) -> None:
        """Create a client whose embeddings calls fail with `error` when set."""
        self.embeddings = FakeEmbeddings(error)

    async def close(self) -> None:
        """Close the client. The fake holds no connection to release."""


def patch_async_openai(
    monkeypatch: pytest.MonkeyPatch, error: Exception | None = None
) -> None:
    """Make the endpoint embedder build a fake client instead of a real one."""

    def factory(**client_options: Any) -> FakeAsyncOpenAI:
        """Return a fake client and ignore the connection options."""
        return FakeAsyncOpenAI(error)

    monkeypatch.setattr("ontogr_mem.vectorstore.builtin.endpoint.AsyncOpenAI", factory)


class TestSelectEmbedder:
    """The endpoint is chosen only when both its model and its base URL are set."""

    async def test_returns_endpoint_embedder_when_model_and_base_url_are_set(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A model and a base URL together select the endpoint embedder."""
        patch_async_openai(monkeypatch)
        # Dimensions are set so the endpoint embedder skips its probe request.
        settings = EmbeddingSettings(
            _env_file=None,
            model="test-model",
            base_url="http://embeddings.invalid/v1",
            dimensions=3,
        )

        embedder = await select_embedder(settings)

        assert isinstance(embedder, EndpointEmbedder)
        assert embedder.model_id == "test-model"
        assert embedder.dim == 3

    async def test_reads_endpoint_settings_from_environment_by_default(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Without a settings argument, the environment selects the endpoint."""
        patch_async_openai(monkeypatch)
        monkeypatch.setenv("MEMORY_EMBEDDING_MODEL", "test-model")
        monkeypatch.setenv("MEMORY_EMBEDDING_BASE_URL", "http://embeddings.invalid/v1")
        monkeypatch.setenv("MEMORY_EMBEDDING_DIMENSIONS", "3")

        embedder = await select_embedder()

        assert isinstance(embedder, EndpointEmbedder)
        assert embedder.model_id == "test-model"
        assert embedder.dim == 3

    @pytest.mark.usefixtures("fake_fastembed")
    async def test_returns_local_embedder_when_no_endpoint_is_configured(
        self,
    ) -> None:
        """Without endpoint settings, the local granite model is used."""
        embedder = await select_embedder(EmbeddingSettings(_env_file=None))

        assert isinstance(embedder, LocalEmbedder)
        assert embedder.model_id == "ibm-granite/granite-embedding-small-english-r2"
        assert embedder.dim == 384

    @pytest.mark.usefixtures("fake_fastembed")
    @pytest.mark.parametrize(
        "settings_fields",
        [
            pytest.param({"model": "test-model"}, id="only-model"),
            pytest.param(
                {"base_url": "http://embeddings.invalid/v1"}, id="only-base-url"
            ),
            pytest.param({"api_key": "sentinel-key"}, id="only-api-key"),
            pytest.param({"dimensions": 3}, id="only-dimensions"),
        ],
    )
    async def test_partial_endpoint_settings_select_the_local_embedder(
        self, settings_fields: dict[str, Any]
    ) -> None:
        """One endpoint field alone never selects the endpoint embedder."""
        settings = EmbeddingSettings(_env_file=None, **settings_fields)

        embedder = await select_embedder(settings)

        assert isinstance(embedder, LocalEmbedder)

    @pytest.mark.usefixtures("fake_fastembed")
    async def test_language_model_and_openai_variables_do_not_select_the_endpoint(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Language-model and ambient OpenAI variables never configure embeddings."""
        monkeypatch.setenv("MEMORY_LLM_MODEL", "sentinel-llm-model")
        monkeypatch.setenv("MEMORY_LLM_BASE_URL", "http://llm.invalid/v1")
        monkeypatch.setenv("MEMORY_LLM_API_KEY", "sentinel-llm-key")
        monkeypatch.setenv("OPENAI_API_KEY", "sentinel-openai-key")
        monkeypatch.setenv("OPENAI_BASE_URL", "http://openai.invalid/v1")

        embedder = await select_embedder()

        assert isinstance(embedder, LocalEmbedder)

    async def test_missing_fastembed_hint_names_the_endpoint_settings(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The error hint names the endpoint settings and the fastembed extra."""
        # A None entry makes `import fastembed` raise ImportError, as a missing
        # package does.
        monkeypatch.setitem(sys.modules, "fastembed", None)

        with pytest.raises(EmbedderUnavailableError) as raised:
            await select_embedder(EmbeddingSettings(_env_file=None))

        hint = raised.value.hint
        assert "MEMORY_EMBEDDING_MODEL" in hint
        assert "MEMORY_EMBEDDING_BASE_URL" in hint
        assert "ontogr-mem[fastembed]" in hint

    async def test_endpoint_failure_is_raised_without_falling_back_to_local(
        self,
        monkeypatch: pytest.MonkeyPatch,
        fake_fastembed: type[FakeTextEmbedding],
    ) -> None:
        """A failing endpoint probe raises its error and never loads the local model."""
        authentication_error = openai.AuthenticationError(
            "denied",
            response=httpx2.Response(
                401,
                request=httpx2.Request(
                    "POST", "http://embeddings.invalid/v1/embeddings"
                ),
            ),
            body=None,
        )
        patch_async_openai(monkeypatch, error=authentication_error)
        settings = EmbeddingSettings(
            _env_file=None,
            model="test-model",
            base_url="http://embeddings.invalid/v1",
        )

        with pytest.raises(PermanentEmbeddingError):
            await select_embedder(settings)

        # The local model must not load after the endpoint fails.
        assert fake_fastembed.constructions == 0
