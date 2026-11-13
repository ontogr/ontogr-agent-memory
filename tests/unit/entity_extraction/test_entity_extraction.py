"""Tests for the entity extraction pipeline."""

from dataclasses import replace

import pytest
from pydantic_ai.exceptions import ModelHTTPError

from ontogr_mem.data_models import Kind, Method, Status
from ontogr_mem.entity_extraction import (
    EntityExtraction,
    EntityExtractionSettings,
    UnknownKindError,
)
from ontogr_mem.entity_extraction.extractors import (
    CompositeExtractor,
    EscalationThresholds,
    FieldEntity,
    GLiNERExtractor,
    LLMExtractor,
    RawSpan,
    RecordMapper,
    RecordMapping,
)
from ontogr_mem.llm import RoleModels
from tests.support.episodes import make_episode
from tests.support.extraction import (
    FakeGliner,
    FakePrevious,
    catalog,
    entities_reply,
    entity_models,
    text_episode,
)


ACME = entities_reply({"name": "Acme", "type_id": 2, "confidence": 0.9})
MAPPER = RecordMapper({"chat": RecordMapping((FieldEntity("vendor", "Company"),))})


def pipeline(models: RoleModels, *, previous=None, max_previous=3):
    """Build a pipeline around a model pool."""
    return EntityExtraction(
        [LLMExtractor(models)],
        MAPPER,
        catalog(),
        previous,
        EntityExtractionSettings(max_previous=max_previous),
    )


class TestEntityExtraction:
    """Behavior of EntityExtraction."""

    async def test_routes_text_to_the_extractor_and_records_to_the_mapper(self):
        """Message and document use the model; record and event use the mapper."""
        models, scripted = entity_models(answer=lambda call: ACME)
        message = text_episode("Acme hired Bo.", "m1")
        document = make_episode("d1", content="Acme grew.")
        document = replace(document, kind=Kind.DOCUMENT)
        record = make_episode("r1", content={"vendor": "Acme"})

        report = await pipeline(models).extract([message, document, record])

        assert [r.episode_id for r in report.results] == ["m1", "d1", "r1"]
        assert [r.status for r in report.results] == [Status.OK] * 3
        assert len(scripted.calls) == 2
        assert report.results[2].entities[0].method is Method.RECORD

    async def test_unknown_kind_raises(self):
        """A kind with no path is a programming error, not a failed episode."""
        episode = make_episode("x1", content="hi")
        episode = replace(episode, kind="mystery")
        models, _ = entity_models()

        with pytest.raises(UnknownKindError):
            await pipeline(models).extract([episode])

    async def test_maps_failures_to_statuses(self):
        """A malformed reply is parse_failed, a provider failure is error."""

        def answer(call):
            if "Alpha" in call.prompt:
                return "bad"
            raise ModelHTTPError(500, "m", {"error": "down"})

        models, _ = entity_models(answer=answer)

        report = await pipeline(models).extract(
            [text_episode("Alpha one.", "e1"), text_episode("Beta two.", "e2")]
        )

        assert sorted(r.status.value for r in report.results) == [
            "error",
            "parse_failed",
        ]
        assert sorted(report.failed_episode_ids) == ["e1", "e2"]
        assert report.status_counts()["ok"] == 0

    async def test_malformed_output_after_reasks_is_parse_failed_and_others_continue(
        self,
    ):
        """Output that stays malformed fails only its episode, with usage kept."""
        models, scripted = entity_models(
            answer=lambda call: "bad" if "Alpha" in call.prompt else ACME
        )
        broken = text_episode("Alpha broke the table.", "broken")
        small = text_episode("Acme hired Bo.", "small")

        report = await pipeline(models).extract([broken, small])

        failed, ok = report.results
        assert (failed.episode_id, failed.status) == ("broken", Status.PARSE_FAILED)
        assert failed.usage.buckets["entities/entity_extraction"].calls == 3
        assert (ok.episode_id, ok.status) == ("small", Status.OK)
        assert len(scripted.calls) == 4

    async def test_provider_failure_after_a_reask_is_error_with_usage(self):
        """A provider failure ends the episode as error and keeps completed usage."""
        models, scripted = entity_models(
            ["bad", ModelHTTPError(503, "m", {"error": "unavailable"})]
        )

        report = await pipeline(models).extract(
            [text_episode("Alpha broke the table.", "down")]
        )

        (result,) = report.results
        assert result.status is Status.ERROR
        assert len(scripted.calls) == 2
        # Only the completed malformed reply is counted; the failed call has no usage.
        assert result.usage.buckets["entities/entity_extraction"].calls == 1

    async def test_skips_text_in_a_language_no_extractor_handles(self):
        """Language routing is explicit."""
        extractor = GLiNERExtractor(model=FakeGliner([]))
        extraction = EntityExtraction([extractor], MAPPER, catalog())

        report = await extraction.extract(
            [text_episode("Hallo", "g1", metadata={"language": "de"})]
        )

        assert report.results[0].status is Status.SKIPPED

    async def test_runs_every_extractor_that_handles_the_language(self):
        """The extractors are merged, not picked one after the other."""
        first = GLiNERExtractor(model=FakeGliner([RawSpan(0, 3, "Ana", "Person", 0.9)]))
        second = GLiNERExtractor(model=FakeGliner([RawSpan(4, 6, "Bo", "Person", 0.9)]))
        extraction = EntityExtraction([first, second], MAPPER, catalog())

        report = await extraction.extract([text_episode("Ana Bo")])

        assert {e.name for e in report.results[0].entities} == {"Ana", "Bo"}

    async def test_message_episode_gets_an_actor_entity(self):
        """The speaker is an entity of the episode, typed Person."""
        models, _ = entity_models(answer=lambda call: ACME)

        report = await pipeline(models).extract([text_episode("Acme hired Bo.")])

        names = {e.name: e for e in report.results[0].entities}
        actor = names["Ana"]
        assert (actor.local_id, actor.type_id, actor.confidence) == ("actor", 0, 1.0)
        assert actor.attributes == {"actor_id": "u1", "role": "user"}

    async def test_empty_result_is_reported_as_empty(self):
        """A reply with no entities is EMPTY, not an error."""
        models, _ = entity_models(answer=lambda call: entities_reply())

        report = await pipeline(models).extract([text_episode("Nothing here.")])

        assert report.results[0].status is Status.EMPTY

    async def test_counts_escalation_triggers_across_the_batch(self):
        """The report sums trigger names over episodes."""
        extractor = CompositeExtractor(
            [GLiNERExtractor(model=FakeGliner([]))],
            thresholds=EscalationThresholds(min_entities_per_1k_chars=100),
        )
        extraction = EntityExtraction([extractor], MAPPER, catalog())

        report = await extraction.extract(
            [text_episode("word " * 100, "a"), text_episode("word " * 100, "b")]
        )

        assert report.trigger_stats == {"too_few_entities": 2}


class TestPreviousEpisodesInThePipeline:
    """The pipeline gives earlier episodes to the model as context."""

    def history(self):
        """Build three earlier episodes."""
        return [
            text_episode(f"Earlier {n}.", f"p{n}", minutes=n - 10) for n in range(3)
        ]

    async def test_earlier_episodes_reach_the_prompt_oldest_first(self):
        """Up to `max_previous` earlier episodes are fetched and shown."""
        models, scripted = entity_models(answer=lambda call: entities_reply())
        previous = FakePrevious(self.history())

        await pipeline(models, previous=previous, max_previous=2).extract(
            [text_episode("Acme now.", minutes=0)]
        )

        prompt = scripted.calls[0].prompt
        assert previous.limits == [2]
        assert "Earlier 0." not in prompt
        assert prompt.index("Earlier 1.") < prompt.index("Earlier 2.")

    async def test_zero_max_previous_turns_the_context_off(self):
        """No fetch and no block."""
        models, scripted = entity_models(answer=lambda call: entities_reply())
        previous = FakePrevious(self.history())

        await pipeline(models, previous=previous, max_previous=0).extract(
            [text_episode("Acme now.", minutes=0)]
        )

        assert previous.limits == []
        assert "<previous_episodes>" not in scripted.calls[0].prompt
