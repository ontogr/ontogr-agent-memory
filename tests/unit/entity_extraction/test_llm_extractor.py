"""Tests for the model entity extractor."""

import pytest
from pydantic_ai.exceptions import ModelHTTPError

from ontogr_mem.data_models import UNKNOWN_TYPE_ID, Method
from ontogr_mem.entity_extraction.extractors import LLMExtractor
from ontogr_mem.entity_extraction.settings import EntityExtractionSettings
from ontogr_mem.llm import ModelCallFailedError, ModelOutputError
from tests.support.extraction import (
    catalog,
    entities_reply,
    entity_models,
    text_episode,
)


TEXT = "Ana moved the Orders table to Postgres."


def reply(*entities):
    """Build a scripted reply."""
    return entities_reply(*entities)


class TestLLMExtractor:
    """Behavior of LLMExtractor."""

    async def test_binds_entities_to_spans_and_types(self):
        """Spans come from the text, not from the model."""
        models, _ = entity_models(
            [
                reply(
                    {"name": "Ana", "type_id": 0, "confidence": 0.9},
                    {"name": "Postgres", "type_id": 1, "confidence": 0.8},
                )
            ]
        )

        result = await LLMExtractor(models).extract(text_episode(TEXT), catalog())

        assert [(e.name, e.type_id, e.spans, e.method) for e in result.entities] == [
            ("Ana", 0, ((0, 3),), Method.LLM),
            ("Postgres", 1, ((30, 38),), Method.LLM),
        ]
        assert result.usage.buckets["entities/entity_extraction"].calls == 1

    async def test_discards_entities_that_are_not_in_the_text_or_have_a_bad_id(self):
        """A name missing from the episode and an id outside the catalog are dropped."""
        models, _ = entity_models(
            [
                reply(
                    {"name": "Bob", "type_id": 0, "confidence": 0.9},
                    {"name": "Ana", "type_id": 9, "confidence": 0.9},
                    {"name": "Postgres", "type_id": 1, "confidence": 0.9},
                )
            ]
        )

        result = await LLMExtractor(models).extract(text_episode(TEXT), catalog())

        assert [e.name for e in result.entities] == ["Postgres"]

    async def test_keeps_the_type_label_only_when_no_catalog_type_fits(self):
        """A named type is kept for an Unknown entity and ignored for a typed one."""
        models, _ = entity_models(
            [
                reply(
                    {"name": "Orders", "type_id": -1, "type_label": "Table"},
                    {"name": "Ana", "type_id": 0, "type_label": "Human"},
                )
            ]
        )

        result = await LLMExtractor(models).extract(text_episode(TEXT), catalog())

        assert [(e.name, e.type_id, e.type_label) for e in result.entities] == [
            ("Orders", UNKNOWN_TYPE_ID, "Table"),
            ("Ana", UNKNOWN_TYPE_ID, None),
        ]

    async def test_low_confidence_entity_is_typed_unknown(self):
        """Below the threshold the model's type is not trusted."""
        models, _ = entity_models(
            [reply({"name": "Ana", "type_id": 0, "confidence": 0.6})]
        )

        result = await LLMExtractor(models).extract(text_episode(TEXT), catalog())

        assert result.entities[0].type_id == UNKNOWN_TYPE_ID

    async def test_reasks_a_malformed_reply_and_keeps_both_requests_in_usage(self):
        """A bad shape is re-asked whole; usage counts both requests."""
        models, scripted = entity_models(
            ["not json", reply({"name": "Ana", "type_id": 0})]
        )

        result = await LLMExtractor(models).extract(text_episode(TEXT), catalog())

        assert [e.name for e in result.entities] == ["Ana"]
        assert len(scripted.calls) == 2
        assert result.usage.buckets["entities/entity_extraction"].calls == 2

    async def test_one_malformed_item_invalidates_the_whole_reply(self):
        """A boolean type id is refused rather than coerced."""
        models, scripted = entity_models(
            [
                reply({"name": "Ana", "type_id": True}),
                reply({"name": "Ana", "type_id": 0, "confidence": 0.9}),
            ]
        )

        result = await LLMExtractor(models).extract(text_episode(TEXT), catalog())

        assert len(scripted.calls) == 2
        assert [e.type_id for e in result.entities] == [0]

    async def test_raises_output_error_with_usage_when_reasks_run_out(self):
        """Three bad replies end in a mapped failure that keeps the counts."""
        models, scripted = entity_models(["bad", "bad", "bad"])

        with pytest.raises(ModelOutputError) as info:
            await LLMExtractor(models).extract(text_episode(TEXT), catalog())

        assert len(scripted.calls) == 3
        assert info.value.usage is not None and info.value.usage.requests == 3

    async def test_provider_context_rejection_is_a_call_failure(self):
        """A rejected prompt is not truncated and retried locally."""
        models, scripted = entity_models(
            [ModelHTTPError(400, "m", {"error": "context length"})]
        )

        with pytest.raises(ModelCallFailedError) as info:
            await LLMExtractor(models).extract(text_episode(TEXT), catalog())

        assert info.value.status_code == 400
        assert len(scripted.calls) == 1

    async def test_blank_episode_makes_no_call(self):
        """An empty text needs no model."""
        models, scripted = entity_models()

        result = await LLMExtractor(models).extract(text_episode("  "), catalog())

        assert result.entities == []
        assert scripted.calls == []


class TestPreviousEpisodes:
    """Earlier episodes are context, never evidence."""

    async def test_previous_episodes_are_in_the_prompt_oldest_first(self):
        """The block lists earlier episodes in the order given."""
        models, scripted = entity_models([reply()])
        previous = [
            text_episode("First earlier note.", "p1", minutes=-2),
            text_episode("Second earlier note.", "p2", minutes=-1),
        ]

        await LLMExtractor(models).extract(text_episode(TEXT), catalog(), previous)

        prompt = scripted.calls[0].prompt
        assert prompt.index("First earlier note.") < prompt.index(
            "Second earlier note."
        )
        assert prompt.index("Second earlier note.") < prompt.index(TEXT)
        assert prompt.count("<episode_text>") == 1

    async def test_no_block_without_previous_episodes(self):
        """An empty context adds no previous_episodes block."""
        models, scripted = entity_models([reply()])

        await LLMExtractor(models).extract(text_episode(TEXT), catalog())

        assert "<previous_episodes>" not in scripted.calls[0].prompt

    async def test_a_name_found_only_in_a_previous_episode_is_not_an_entity(self):
        """Spans are searched in the current episode only."""
        models, _ = entity_models(
            [reply({"name": "Carol", "type_id": 0, "confidence": 0.9})]
        )
        previous = [text_episode("Carol joined.", "p1", minutes=-1)]

        result = await LLMExtractor(models).extract(
            text_episode(TEXT), catalog(), previous
        )

        assert result.entities == []

    async def test_a_long_prompt_reaches_the_model_intact(self):
        """No local fitting: every earlier episode is delivered whole."""
        models, scripted = entity_models([reply()])
        previous = [
            text_episode(f"{tag} " + "x" * 300, f"p{tag}", minutes=-3 + i)
            for i, tag in enumerate("ABC")
        ]

        await LLMExtractor(models).extract(text_episode(TEXT), catalog(), previous)

        prompt = scripted.calls[0].prompt
        for tag in "ABC":
            assert f"{tag} " + "x" * 300 in prompt
        assert TEXT in prompt

    async def test_instructions_come_from_the_agent_not_the_prompt(self):
        """The system text is the agent's instructions."""
        models, scripted = entity_models([reply()])

        await LLMExtractor(models).extract(text_episode(TEXT), catalog())

        assert scripted.calls[0].system
        assert scripted.calls[0].system not in scripted.calls[0].prompt


class TestSettings:
    """Settings change extractor behavior."""

    async def test_threshold_comes_from_settings(self):
        """A lower threshold keeps the model's type."""
        settings = EntityExtractionSettings(threshold=0.5)
        models, _ = entity_models(
            [reply({"name": "Ana", "type_id": 0, "confidence": 0.6})]
        )

        result = await LLMExtractor(models, settings).extract(
            text_episode(TEXT), catalog()
        )

        assert result.entities[0].type_id == 0
