"""Embedding settings read MEMORY_EMBEDDING_ variables and validate their ranges."""

import pytest
from pydantic import ValidationError

from ontogr_mem.vectorstore.builtin.settings import EmbeddingSettings


@pytest.fixture(autouse=True)
def clean_environment(monkeypatch):
    """Remove embedding variables a developer machine may have set."""
    for name in (
        "MEMORY_EMBEDDING_MODEL",
        "MEMORY_EMBEDDING_BASE_URL",
        "MEMORY_EMBEDDING_API_KEY",
        "MEMORY_EMBEDDING_DIMENSIONS",
        "MEMORY_EMBEDDING_MAX_INPUT_TOKENS",
        "MEMORY_EMBEDDING_PROVIDER",
        "MEMORY_EMBEDDING_FASTEMBED_KWARGS",
        "MEMORY_EMBEDDING_BATCH_SIZE",
    ):
        monkeypatch.delenv(name, raising=False)


class TestEmbeddingSettings:
    """Behavior of EmbeddingSettings."""

    def test_defaults_when_no_embedding_variables_are_set(self):
        """Optional fields are unset and the input token limit is 8192."""
        settings = EmbeddingSettings(_env_file=None)

        assert settings.model is None
        assert settings.base_url is None
        assert settings.api_key is None
        assert settings.dimensions is None
        assert settings.max_input_tokens == 8192
        assert settings.provider == "auto"
        assert settings.fastembed_kwargs == {}
        assert settings.batch_size == 32

    def test_python_and_environment_values_are_equivalent(self, monkeypatch):
        """Python options and their JSON environment form resolve identically."""
        options = {
            "model_name": "sample/model",
            "providers": [["CUDAExecutionProvider", {"device_id": 0}]],
            "threads": 2,
        }
        python_settings = EmbeddingSettings(
            _env_file=None,
            provider="fastembed",
            fastembed_kwargs=options,
            batch_size=8,
        )
        monkeypatch.setenv("MEMORY_EMBEDDING_PROVIDER", "fastembed")
        monkeypatch.setenv(
            "MEMORY_EMBEDDING_FASTEMBED_KWARGS",
            '{"model_name":"sample/model","providers":[["CUDAExecutionProvider",'
            '{"device_id":0}]],"threads":2}',
        )
        monkeypatch.setenv("MEMORY_EMBEDDING_BATCH_SIZE", "8")

        environment_settings = EmbeddingSettings(_env_file=None)

        assert environment_settings.provider == python_settings.provider
        assert environment_settings.fastembed_kwargs == python_settings.fastembed_kwargs
        assert environment_settings.batch_size == python_settings.batch_size
        assert environment_settings.model is None

    def test_fastembed_kwargs_default_is_isolated_between_settings(self):
        """Each settings instance owns its mutable native-options dictionary."""
        first = EmbeddingSettings(_env_file=None)
        second = EmbeddingSettings(_env_file=None)

        first.fastembed_kwargs["threads"] = 2

        assert second.fastembed_kwargs == {}

    def test_hides_fastembed_kwargs_in_repr(self):
        """Native options do not appear in settings representations."""
        settings = EmbeddingSettings(
            _env_file=None,
            fastembed_kwargs={"cache_dir": "/private/model-cache-sentinel"},
        )

        assert "/private/model-cache-sentinel" not in repr(settings)
        assert "/private/model-cache-sentinel" not in str(settings)

    @pytest.mark.parametrize("provider", ["auto", "fastembed", "endpoint"])
    def test_accepts_supported_providers(self, provider):
        """The configured provider uses one of the supported names."""
        assert EmbeddingSettings(_env_file=None, provider=provider).provider == provider

    def test_rejects_unsupported_providers_without_echoing_values(self):
        """An unsupported provider fails without showing the configured value."""
        value = "other-provider-sentinel"
        with pytest.raises(ValidationError) as error:
            EmbeddingSettings(_env_file=None, provider=value)

        assert value not in str(error.value)

    @pytest.mark.parametrize("value", ["0", "-1"])
    def test_rejects_nonpositive_batch_sizes(self, monkeypatch, value):
        """Batch sizes must be positive integers."""
        monkeypatch.setenv("MEMORY_EMBEDDING_BATCH_SIZE", value)

        with pytest.raises(ValidationError):
            EmbeddingSettings(_env_file=None)

    @pytest.mark.parametrize("value", ['"kwargs-sentinel"', '["kwargs-sentinel"]'])
    def test_rejects_non_object_fastembed_kwargs_without_echoing_values(
        self, monkeypatch, value
    ):
        """The environment accepts JSON objects and hides rejected raw values."""
        monkeypatch.setenv("MEMORY_EMBEDDING_FASTEMBED_KWARGS", value)

        with pytest.raises(ValidationError) as error:
            EmbeddingSettings(_env_file=None)

        assert "kwargs-sentinel" not in str(error.value)

    def test_rejects_non_object_python_fastembed_kwargs_without_echoing_value(self):
        """Python callers must pass an object without exposing rejected values."""
        value = "python-kwargs-sentinel"

        with pytest.raises(ValidationError) as error:
            EmbeddingSettings(_env_file=None, fastembed_kwargs=value)

        assert value not in str(error.value)

    def test_reads_all_embedding_environment_variables(self, monkeypatch):
        """Each MEMORY_EMBEDDING_ variable sets its matching field."""
        monkeypatch.setenv("MEMORY_EMBEDDING_MODEL", "test-model")
        monkeypatch.setenv("MEMORY_EMBEDDING_BASE_URL", "http://embeddings.invalid/v1")
        monkeypatch.setenv("MEMORY_EMBEDDING_API_KEY", "sentinel-key")
        monkeypatch.setenv("MEMORY_EMBEDDING_DIMENSIONS", "768")
        monkeypatch.setenv("MEMORY_EMBEDDING_MAX_INPUT_TOKENS", "512")
        monkeypatch.setenv("MEMORY_EMBEDDING_PROVIDER", "fastembed")
        monkeypatch.setenv(
            "MEMORY_EMBEDDING_FASTEMBED_KWARGS",
            '{"model_name":"test-model","threads":2}',
        )
        monkeypatch.setenv("MEMORY_EMBEDDING_BATCH_SIZE", "7")

        settings = EmbeddingSettings(_env_file=None)

        assert settings.model == "test-model"
        assert settings.base_url == "http://embeddings.invalid/v1"
        assert settings.api_key.get_secret_value() == "sentinel-key"
        assert settings.dimensions == 768
        assert settings.max_input_tokens == 512
        assert settings.provider == "fastembed"
        assert settings.fastembed_kwargs == {"model_name": "test-model", "threads": 2}
        assert settings.batch_size == 7

    @pytest.mark.parametrize(
        ("variable", "attribute"),
        [
            ("MEMORY_EMBEDDING_MODEL", "model"),
            ("MEMORY_EMBEDDING_BASE_URL", "base_url"),
            ("MEMORY_EMBEDDING_API_KEY", "api_key"),
            ("MEMORY_EMBEDDING_DIMENSIONS", "dimensions"),
        ],
    )
    @pytest.mark.parametrize("blank", ["", "   "])
    def test_blank_value_becomes_none(self, monkeypatch, variable, attribute, blank):
        """Empty or whitespace-only values mean the setting is missing."""
        monkeypatch.setenv(variable, blank)

        settings = EmbeddingSettings(_env_file=None)

        assert getattr(settings, attribute) is None

    def test_hides_api_key_in_repr_and_str(self, monkeypatch):
        """The API key text is absent from repr and str but still readable."""
        monkeypatch.setenv("MEMORY_EMBEDDING_API_KEY", "sentinel-key")

        settings = EmbeddingSettings(_env_file=None)

        assert "sentinel-key" not in repr(settings)
        assert "sentinel-key" not in str(settings)
        assert settings.api_key.get_secret_value() == "sentinel-key"

    @pytest.mark.parametrize(("value", "expected"), [("1", 1), ("4000", 4000)])
    def test_accepts_dimensions_at_the_bounds(self, monkeypatch, value, expected):
        """Dimensions of 1 and 4000 are the inclusive limits."""
        monkeypatch.setenv("MEMORY_EMBEDDING_DIMENSIONS", value)

        settings = EmbeddingSettings(_env_file=None)

        assert settings.dimensions == expected

    @pytest.mark.parametrize("value", ["0", "4001"])
    def test_rejects_dimensions_outside_the_bounds(self, monkeypatch, value):
        """Dimensions below 1 or above 4000 fail validation."""
        monkeypatch.setenv("MEMORY_EMBEDDING_DIMENSIONS", value)

        with pytest.raises(ValidationError):
            EmbeddingSettings(_env_file=None)

    def test_rejects_zero_max_input_tokens(self, monkeypatch):
        """At least one input token is required."""
        monkeypatch.setenv("MEMORY_EMBEDDING_MAX_INPUT_TOKENS", "0")

        with pytest.raises(ValidationError):
            EmbeddingSettings(_env_file=None)

    def test_accepts_one_max_input_token(self, monkeypatch):
        """A limit of one token is the smallest accepted value."""
        monkeypatch.setenv("MEMORY_EMBEDDING_MAX_INPUT_TOKENS", "1")

        settings = EmbeddingSettings(_env_file=None)

        assert settings.max_input_tokens == 1

    def test_ignores_unrelated_environment_variables(self, monkeypatch):
        """Unknown keys and other API key variables never populate api_key."""
        monkeypatch.setenv("MEMORY_EMBEDDING_UNKNOWN_FIELD", "x")
        monkeypatch.setenv("MEMORY_LLM_API_KEY", "llm-secret-sentinel")
        monkeypatch.setenv("OPENAI_API_KEY", "ambient-sentinel")

        settings = EmbeddingSettings(_env_file=None)

        assert settings.api_key is None
