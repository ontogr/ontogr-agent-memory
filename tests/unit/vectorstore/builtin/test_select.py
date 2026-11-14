"""Provider selection in `select_embedder`."""

from pathlib import Path
from typing import Any

import pytest

from ontogr_mem.vectorstore.builtin.endpoint import EndpointEmbedder
from ontogr_mem.vectorstore.builtin.local import LocalEmbedder
from ontogr_mem.vectorstore.builtin.select import select_embedder
from ontogr_mem.vectorstore.builtin.settings import EmbeddingSettings
from ontogr_mem.vectorstore.errors import EmbedderUnavailableError


EMBEDDING_VARIABLES = (
    "MEMORY_EMBEDDING_MODEL",
    "MEMORY_EMBEDDING_BASE_URL",
    "MEMORY_EMBEDDING_API_KEY",
    "MEMORY_EMBEDDING_DIMENSIONS",
    "MEMORY_EMBEDDING_MAX_INPUT_TOKENS",
    "MEMORY_EMBEDDING_PROVIDER",
    "MEMORY_EMBEDDING_FASTEMBED_KWARGS",
    "MEMORY_EMBEDDING_BATCH_SIZE",
)


@pytest.fixture(autouse=True)
def clean_embedding_environment(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Run each test without embedding variables or a working-directory .env."""
    monkeypatch.chdir(tmp_path)
    for name in EMBEDDING_VARIABLES:
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def provider_openers(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Replace provider open methods and record the selected settings."""
    calls: dict[str, list[EmbeddingSettings]] = {"endpoint": [], "fastembed": []}
    results = {"endpoint": object(), "fastembed": object()}
    failures: dict[str, Exception] = {}

    async def open_endpoint(
        embedder_class: type[EndpointEmbedder], settings: EmbeddingSettings
    ) -> object:
        calls["endpoint"].append(settings)
        if "endpoint" in failures:
            raise failures["endpoint"]
        return results["endpoint"]

    async def open_fastembed(
        embedder_class: type[LocalEmbedder], settings: EmbeddingSettings
    ) -> object:
        calls["fastembed"].append(settings)
        if "fastembed" in failures:
            raise failures["fastembed"]
        return results["fastembed"]

    monkeypatch.setattr(EndpointEmbedder, "open", classmethod(open_endpoint))
    monkeypatch.setattr(LocalEmbedder, "open", classmethod(open_fastembed))
    return {"calls": calls, "results": results, "failures": failures}


class TestSelectEmbedder:
    """Provider choice follows the explicit setting and endpoint completeness."""

    async def test_auto_selects_endpoint_when_model_and_base_url_are_set(
        self, provider_openers: dict[str, Any]
    ) -> None:
        """Auto mode opens the endpoint only when both required fields exist."""
        settings = EmbeddingSettings(
            _env_file=None,
            model="endpoint-model-sentinel",
            base_url="https://endpoint.invalid/v1",
            fastembed_kwargs={"model_name": "inactive-local-model"},
        )

        embedder = await select_embedder(settings)

        assert embedder is provider_openers["results"]["endpoint"]
        assert provider_openers["calls"] == {"endpoint": [settings], "fastembed": []}

    async def test_auto_selects_fastembed_when_no_endpoint_fields_are_set(
        self, provider_openers: dict[str, Any]
    ) -> None:
        """Auto mode opens FastEmbed when neither endpoint field is present."""
        settings = EmbeddingSettings(
            _env_file=None,
            api_key="inactive-key-sentinel",
            dimensions=768,
            fastembed_kwargs={"model_name": "local-model"},
        )

        embedder = await select_embedder(settings)

        assert embedder is provider_openers["results"]["fastembed"]
        assert provider_openers["calls"] == {"endpoint": [], "fastembed": [settings]}

    async def test_explicit_endpoint_opens_with_complete_configuration(
        self, provider_openers: dict[str, Any]
    ) -> None:
        """Explicit endpoint selection passes complete settings to its opener."""
        settings = EmbeddingSettings(
            _env_file=None,
            provider="endpoint",
            model="endpoint-model",
            base_url="https://endpoint.invalid/v1",
            fastembed_kwargs={"model_name": "inactive-local-model"},
        )

        embedder = await select_embedder(settings)

        assert embedder is provider_openers["results"]["endpoint"]
        assert provider_openers["calls"] == {"endpoint": [settings], "fastembed": []}

    @pytest.mark.parametrize(
        "settings_fields",
        [
            pytest.param({"model": "endpoint-model-sentinel"}, id="model-only"),
            pytest.param(
                {"base_url": "https://endpoint.invalid/v1"}, id="base-url-only"
            ),
        ],
    )
    async def test_auto_rejects_partial_endpoint_configuration_safely(
        self,
        provider_openers: dict[str, Any],
        settings_fields: dict[str, Any],
    ) -> None:
        """Partial endpoint settings raise a safe configuration error."""
        settings = EmbeddingSettings(_env_file=None, **settings_fields)

        with pytest.raises(EmbedderUnavailableError) as error:
            await select_embedder(settings)

        message = f"{error.value} {error.value.hint}"
        assert "MEMORY_EMBEDDING_MODEL" in message
        assert "MEMORY_EMBEDDING_BASE_URL" in message
        assert "endpoint-model-sentinel" not in message
        assert "endpoint.invalid" not in message
        assert provider_openers["calls"] == {"endpoint": [], "fastembed": []}

    @pytest.mark.parametrize(
        "settings_fields",
        [
            pytest.param({}, id="neither-field"),
            pytest.param({"model": "endpoint-model-sentinel"}, id="model-only"),
            pytest.param(
                {"base_url": "https://endpoint.invalid/v1"}, id="base-url-only"
            ),
        ],
    )
    async def test_explicit_endpoint_requires_both_endpoint_fields(
        self,
        provider_openers: dict[str, Any],
        settings_fields: dict[str, Any],
    ) -> None:
        """Explicit endpoint selection never falls back to FastEmbed."""
        settings = EmbeddingSettings(
            _env_file=None, provider="endpoint", **settings_fields
        )

        with pytest.raises(EmbedderUnavailableError) as error:
            await select_embedder(settings)

        assert "MEMORY_EMBEDDING_MODEL" in error.value.hint
        assert "MEMORY_EMBEDDING_BASE_URL" in error.value.hint
        assert provider_openers["calls"] == {"endpoint": [], "fastembed": []}

    @pytest.mark.parametrize(
        "settings_fields",
        [
            pytest.param({}, id="no-endpoint-fields"),
            pytest.param({"model": "inactive-model"}, id="model-only"),
            pytest.param(
                {"base_url": "https://endpoint.invalid/v1"}, id="base-url-only"
            ),
            pytest.param(
                {
                    "model": "inactive-model",
                    "base_url": "https://endpoint.invalid/v1",
                },
                id="complete-endpoint",
            ),
        ],
    )
    async def test_explicit_fastembed_ignores_endpoint_settings(
        self,
        provider_openers: dict[str, Any],
        settings_fields: dict[str, Any],
    ) -> None:
        """Explicit FastEmbed selection passes settings through unchanged."""
        settings = EmbeddingSettings(
            _env_file=None,
            provider="fastembed",
            fastembed_kwargs={"model_name": "local-model"},
            **settings_fields,
        )

        embedder = await select_embedder(settings)

        assert embedder is provider_openers["results"]["fastembed"]
        assert provider_openers["calls"] == {"endpoint": [], "fastembed": [settings]}

    async def test_reads_explicit_provider_from_environment(
        self,
        monkeypatch: pytest.MonkeyPatch,
        provider_openers: dict[str, Any],
    ) -> None:
        """The environment can explicitly choose FastEmbed without endpoint data."""
        monkeypatch.setenv("MEMORY_EMBEDDING_PROVIDER", "fastembed")
        monkeypatch.setenv("MEMORY_EMBEDDING_FASTEMBED_KWARGS", '{"threads":2}')

        embedder = await select_embedder()

        assert embedder is provider_openers["results"]["fastembed"]
        selected = provider_openers["calls"]["fastembed"][0]
        assert selected.provider == "fastembed"
        assert selected.fastembed_kwargs == {"threads": 2}
        assert provider_openers["calls"]["endpoint"] == []

    @pytest.mark.parametrize(
        ("provider", "selected", "other"),
        [
            pytest.param("auto", "endpoint", "fastembed", id="auto-endpoint"),
            pytest.param("endpoint", "endpoint", "fastembed", id="endpoint"),
            pytest.param("fastembed", "fastembed", "endpoint", id="fastembed"),
        ],
    )
    async def test_selected_provider_failure_does_not_fall_back(
        self,
        provider: str,
        selected: str,
        other: str,
        provider_openers: dict[str, Any],
    ) -> None:
        """A selected provider's failure propagates without opening the other."""
        failure = RuntimeError("selected provider failed")
        provider_openers["failures"][selected] = failure
        settings = EmbeddingSettings(
            _env_file=None,
            provider=provider,
            model="endpoint-model",
            base_url="https://endpoint.invalid/v1",
        )

        with pytest.raises(RuntimeError, match="selected provider failed") as error:
            await select_embedder(settings)

        assert error.value is failure
        assert provider_openers["calls"][selected] == [settings]
        assert provider_openers["calls"][other] == []
