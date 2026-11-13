"""Entity and fact extraction against a configured live language model."""

import pytest
from pydantic_ai import models

from ontogr_mem.data_models import TypeView
from ontogr_mem.entity_extraction.extractors import LLMExtractor
from ontogr_mem.fact_extraction import LLMFactExtractor
from ontogr_mem.llm import LLMConfig, build_models
from tests.support.extraction import catalog, text_episode


pytestmark = [pytest.mark.slow, pytest.mark.enable_socket]

TEXT = "Ana switched to Postgres in March 2024 because Acme needs transactions."
EXPECTED_NAMES = {"ana", "postgres", "acme"}


async def test_entities_and_facts_are_extracted_from_a_real_model() -> None:
    """A live model names the entities and grounds at least one fact in the text."""
    config = LLMConfig()
    if config.model is None:
        pytest.skip("the global model is not configured")

    with models.override_allow_model_requests(True):
        role_models = build_models(config)
        try:
            episode = text_episode(TEXT)
            types = catalog()
            extracted = await LLMExtractor(role_models).extract(episode, types)

            names = {entity.name.lower() for entity in extracted.entities}
            assert len(names & EXPECTED_NAMES) >= 2

            result = await LLMFactExtractor(role_models).extract(
                episode, extracted.entities, TypeView(types)
            )

            grounded = [
                fact
                for fact in result.facts
                if fact.evidence_span is not None
                and TEXT[fact.evidence_span[0] : fact.evidence_span[1]]
                == fact.evidence_quote
            ]
            assert grounded
        finally:
            await role_models.aclose()
