"""Tests for model settings resolution."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from ontogr_mem.llm import LLMConfig, ModelRole


def make(**values: object) -> LLMConfig:
    """Helper: make."""
    return LLMConfig(_env_file=None, **values)


def test_defaults() -> None:
    """Defaults."""
    cfg = make()
    assert cfg.output_mode == "tool"
    assert cfg.max_concurrency == 4
    assert cfg.timeout_s == 300.0
    caps = {role: cfg.resolve(role).max_output_tokens for role in ModelRole}
    assert caps[ModelRole.FACT_EXTRACTION] == 16384
    assert caps[ModelRole.ENTITY_EXTRACTION] == 2048
    assert caps[ModelRole.ANSWER] is None
    assert all(cfg.resolve(role).model is None for role in ModelRole)


def test_global_env_is_inherited(monkeypatch: pytest.MonkeyPatch) -> None:
    """Global env is inherited."""
    monkeypatch.setenv("MEMORY_LLM_MODEL", "m")
    monkeypatch.setenv("MEMORY_LLM_BASE_URL", "http://x")
    monkeypatch.setenv("MEMORY_LLM_API_KEY", "k")
    monkeypatch.setenv("MEMORY_LLM_MAX_OUTPUT_TOKENS", "99")
    monkeypatch.setenv("MEMORY_LLM_EXTRA_BODY", '{"a": 1}')
    cfg = make()
    for role in ModelRole:
        r = cfg.resolve(role)
        assert (r.model, r.base_url, r.max_output_tokens) == ("m", "http://x", 99)
        assert r.api_key is not None and r.api_key.get_secret_value() == "k"
        assert r.extra_body == {"a": 1}


def test_all_eight_role_variables_win(monkeypatch: pytest.MonkeyPatch) -> None:
    """All eight role variables win."""
    monkeypatch.setenv("MEMORY_LLM_MODEL", "g")
    monkeypatch.setenv("MEMORY_LLM_BASE_URL", "http://g")
    monkeypatch.setenv("MEMORY_LLM_API_KEY", "gk")
    monkeypatch.setenv("MEMORY_LLM_OUTPUT_MODE", "tool")
    monkeypatch.setenv("MEMORY_LLM_MAX_OUTPUT_TOKENS", "5")
    monkeypatch.setenv("MEMORY_LLM_REASONING_EFFORT", "low")
    monkeypatch.setenv("MEMORY_LLM_TIMEOUT_S", "10")
    monkeypatch.setenv("MEMORY_LLM_EXTRA_BODY", '{"g": 1}')
    p = "MEMORY_LLM_FACT_EXTRACTION_"
    monkeypatch.setenv(p + "MODEL", "anthropic:r")
    monkeypatch.setenv(p + "BASE_URL", "http://r")
    monkeypatch.setenv(p + "API_KEY", "rk")
    monkeypatch.setenv(p + "OUTPUT_MODE", "native")
    monkeypatch.setenv(p + "MAX_OUTPUT_TOKENS", "7")
    monkeypatch.setenv(p + "REASONING_EFFORT", "high")
    monkeypatch.setenv(p + "TIMEOUT_S", "20")
    monkeypatch.setenv(p + "EXTRA_BODY", '{"r": 2}')
    cfg = make()
    r = cfg.resolve(ModelRole.FACT_EXTRACTION)
    assert r.model == "anthropic:r"
    assert r.base_url == "http://r"
    assert r.api_key is not None and r.api_key.get_secret_value() == "rk"
    assert (r.output_mode, r.max_output_tokens) == ("native", 7)
    assert (r.reasoning_effort, r.timeout_s) == ("high", 20.0)
    assert r.extra_body == {"r": 2}
    other = cfg.resolve(ModelRole.ENTITY_EXTRACTION)
    assert (other.model, other.max_output_tokens, other.timeout_s) == ("g", 5, 10.0)
    assert other.extra_body == {"g": 1}


def test_blank_values_are_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    """Blank values are missing."""
    monkeypatch.setenv("MEMORY_LLM_MODEL", "g")
    monkeypatch.setenv("MEMORY_LLM_BASE_URL", "http://g")
    monkeypatch.setenv("MEMORY_LLM_TIMEOUT_S", "  ")
    monkeypatch.setenv("MEMORY_LLM_MAX_OUTPUT_TOKENS", "")
    monkeypatch.setenv("MEMORY_LLM_EXTRA_BODY", " ")
    monkeypatch.setenv("MEMORY_LLM_JUDGE_MODEL", "")
    monkeypatch.setenv("MEMORY_LLM_JUDGE_BASE_URL", "  ")
    monkeypatch.setenv("MEMORY_LLM_JUDGE_TIMEOUT_S", " ")
    monkeypatch.setenv("MEMORY_LLM_JUDGE_EXTRA_BODY", " ")
    r = make().resolve(ModelRole.JUDGE)
    assert (r.model, r.base_url, r.timeout_s, r.extra_body) == (
        "g",
        "http://g",
        300.0,
        None,
    )
    assert r.max_output_tokens == 2048


def test_empty_object_replaces_global(monkeypatch: pytest.MonkeyPatch) -> None:
    """Empty object replaces global."""
    monkeypatch.setenv("MEMORY_LLM_EXTRA_BODY", '{"a": 1}')
    monkeypatch.setenv("MEMORY_LLM_JUDGE_EXTRA_BODY", "{}")
    cfg = make()
    assert cfg.resolve(ModelRole.JUDGE).extra_body == {}
    assert cfg.resolve(ModelRole.ONTOLOGY).extra_body == {"a": 1}


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("MEMORY_LLM_EXTRA_BODY", "[1]"),
        ("MEMORY_LLM_EXTRA_BODY", "{bad"),
        ("MEMORY_LLM_JUDGE_EXTRA_BODY", "[1]"),
        ("MEMORY_LLM_JUDGE_EXTRA_BODY", "{bad"),
        ("MEMORY_LLM_MAX_OUTPUT_TOKENS", "0"),
        ("MEMORY_LLM_JUDGE_MAX_OUTPUT_TOKENS", "0"),
        ("MEMORY_LLM_TIMEOUT_S", "0"),
        ("MEMORY_LLM_JUDGE_TIMEOUT_S", "0"),
        ("MEMORY_LLM_OUTPUT_MODE", "bogus"),
        ("MEMORY_LLM_JUDGE_OUTPUT_MODE", "bogus"),
        ("MEMORY_LLM_MAX_CONCURRENCY", "0"),
    ],
)
def test_invalid_values_fail(
    monkeypatch: pytest.MonkeyPatch, name: str, value: str
) -> None:
    """Invalid values fail."""
    monkeypatch.setenv(name, value)
    with pytest.raises((ValidationError, ValueError)):
        make()


def test_init_beats_env_and_env_beats_dotenv(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Init beats env and env beats dotenv."""
    dotenv = tmp_path / ".env"
    dotenv.write_text(
        "MEMORY_LLM_MODEL=from-dotenv\nMEMORY_LLM_BASE_URL=http://d\n"
        "MEMORY_LLM_JUDGE_MODEL=judge-dotenv\nMEMORY_LLM_TIMEOUT_S=11\n"
    )
    monkeypatch.setenv("MEMORY_LLM_MODEL", "from-env")
    monkeypatch.setenv("MEMORY_LLM_JUDGE_MODEL", "judge-env")
    cfg = LLMConfig(_env_file=dotenv)
    assert cfg.model == "from-env"
    assert cfg.resolve(ModelRole.JUDGE).model == "judge-env"
    assert cfg.timeout_s == 11.0
    cfg = LLMConfig(_env_file=dotenv, model="from-init")
    assert cfg.model == "from-init"
    monkeypatch.delenv("MEMORY_LLM_MODEL")
    monkeypatch.delenv("MEMORY_LLM_JUDGE_MODEL")
    assert LLMConfig(_env_file=dotenv).model == "from-dotenv"


def test_explicit_construction_uses_same_validators() -> None:
    """Explicit construction uses same validators."""
    cfg = make(
        model="m",
        base_url="http://x",
        extra_body='{"a": 1}',
        roles={"judge": {"timeout_s": 5, "extra_body": "{}"}},
    )
    assert cfg.resolve(ModelRole.JUDGE).timeout_s == 5
    assert cfg.resolve(ModelRole.JUDGE).extra_body == {}
    assert cfg.resolve(ModelRole.ANSWER).extra_body == {"a": 1}


def test_secret_is_hidden() -> None:
    """Secret is hidden."""
    cfg = make(api_key="sentinel-secret-value")
    assert "sentinel-secret-value" not in repr(cfg)
    assert "sentinel-secret-value" not in repr(cfg.resolve(ModelRole.JUDGE))


def test_bare_openai_prefix_names_valid_prefixes() -> None:
    """Bare openai prefix names valid prefixes."""
    with pytest.raises(ValueError, match="openai-chat") as info:
        make(model="openai:gpt", base_url="http://x")
    assert "openai-responses" in str(info.value)
    with pytest.raises(ValueError, match="openai-chat"):
        make(base_url="http://x", roles={"judge": {"model": "openai:gpt"}})


@pytest.mark.parametrize(
    "prefix", ["", "openai-chat:", "openai-responses:", "anthropic:"]
)
def test_missing_base_url_fails_with_hint(prefix: str) -> None:
    """Missing base url fails with hint."""
    with pytest.raises(ValueError, match="MEMORY_LLM_BASE_URL"):
        make(model=prefix + "m")
    with pytest.raises(ValueError, match="MEMORY_LLM_JUDGE_BASE_URL"):
        make(roles={"judge": {"model": prefix + "m"}})


def test_blank_base_url_counts_as_missing() -> None:
    """Blank base url counts as missing."""
    with pytest.raises(ValueError, match="BASE_URL"):
        make(model="m", base_url="  ")


def test_model_construct_reads_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    """Model construct reads nothing."""
    monkeypatch.setenv("MEMORY_LLM_MODEL", "m")
    monkeypatch.setenv("MEMORY_LLM_BASE_URL", "http://x")
    cfg = LLMConfig.model_construct()
    assert all(cfg.resolve(role).model is None for role in ModelRole)
