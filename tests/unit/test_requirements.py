"""Requirement checks for optional runtime features."""

import asyncio

from ontogr_mem.llm.config import LLMConfig
from ontogr_mem.llm.models import build_models
from ontogr_mem.requirements import TEXT_WRITE_ROLES, check_requirements
from ontogr_mem.store_urls import parse_graph_url, parse_state_url
from ontogr_mem.vectorstore.builtin.settings import EmbeddingSettings


class TestCheckRequirements:
    """Reports extras and model configuration needed by a command."""

    def test_collects_backend_embedding_and_mcp_requirements(self):
        """The report follows stable order and describes keyless model setup."""
        models = build_models(LLMConfig.model_construct())
        try:
            report = check_requirements(
                models=models,
                graph=parse_graph_url("redis://127.0.0.1:6379"),
                state=parse_state_url("postgresql://user:pass@127.0.0.1/db"),
                embedding=EmbeddingSettings(provider="fastembed"),
                mcp=True,
                required_roles=frozenset(TEXT_WRITE_ROLES),
            )
        finally:
            asyncio.run(models.aclose())

        assert report.required == ("falkordb", "postgres", "fastembed", "mcp")
        assert report.missing_roles == TEXT_WRITE_ROLES
        assert report.mode == "keyless"
