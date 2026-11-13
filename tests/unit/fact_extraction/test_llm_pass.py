"""The model fact pass: re-asks, skips, context and drops."""

import pytest
from pydantic_ai.exceptions import ModelHTTPError, ModelRequestTimeout

from ontogr_mem.data_models import Kind, Status
from ontogr_mem.fact_extraction import (
    FactDropReason,
    FactExtractionSettings,
    LLMFactExtractor,
    SkipReason,
    UnknownKindError,
)
from ontogr_mem.llm import RoleModels
from tests.support.facts import (
    ANA,
    PG,
    T0,
    fact,
    fact_models,
    make_episode,
    reply,
)


def extractor(models: RoleModels, settings=None) -> LLMFactExtractor:
    """Build an extractor around the scripted model pool."""
    return LLMFactExtractor(models, settings)


async def extract(
    ex: LLMFactExtractor,
    episode=None,
    entities=None,
    types=None,
    previous=(),
    *,
    force: bool = False,
):
    """Run the extractor with the test defaults."""
    return await ex.extract(
        episode or make_episode(),
        [ANA, PG] if entities is None else entities,
        types,
        previous,
        force=force,
    )


class TestLLMPass:
    """Behavior of the model fact pass."""

    async def test_ok_path_binds_facts_and_records_one_call_in_the_facts_bucket(self):
        """A good reply becomes facts and one usage bucket."""
        models, scripted = fact_models([reply(fact())])

        result = await extract(extractor(models))

        assert result.status is Status.OK
        assert [f.text for f in result.facts] == ["Ana switched to Postgres"]
        assert result.usage.buckets["facts/fact_extraction"].calls == 1
        assert len(scripted.calls) == 1

    async def test_no_facts_is_empty_not_error(self):
        """An empty reply is a final, successful outcome."""
        models, _ = fact_models([reply()])

        result = await extract(extractor(models))

        assert result.status is Status.EMPTY
        assert result.facts == []

    async def test_malformed_replies_are_reasked_whole_until_one_validates(self):
        """Two malformed replies are re-asked; the third is bound and kept."""
        models, scripted = fact_models(
            ["not json", reply(fact(fact_kind="oops")), reply(fact())]
        )

        result = await extract(extractor(models))

        assert result.status is Status.OK
        assert [f.text for f in result.facts] == ["Ana switched to Postgres"]
        assert len(scripted.calls) == 3
        assert result.usage.buckets["facts/fact_extraction"].calls == 3

    async def test_exhausted_reasks_are_parse_failed_with_usage_recorded(self):
        """Three invalid replies are retryable, never an empty success."""
        models, scripted = fact_models(["nope", "still nope", "no again"])

        result = await extract(extractor(models))

        assert result.status is Status.PARSE_FAILED
        assert result.facts == []
        assert result.detail
        assert len(scripted.calls) == 3
        assert result.usage.buckets["facts/fact_extraction"].calls == 3

    async def test_provider_failure_is_error_after_a_single_call(self):
        """A provider rejection ends the episode without a local second attempt."""
        models, scripted = fact_models([ModelHTTPError(503, "m", {"error": "down"})])

        result = await extract(extractor(models))

        assert result.status is Status.ERROR
        assert "503" in result.detail
        assert len(scripted.calls) == 1

    async def test_timeout_ends_the_episode_as_error_without_more_calls(self):
        """A timeout is not retried within the episode."""
        models, scripted = fact_models([ModelRequestTimeout("m", 1.0)])

        result = await extract(extractor(models))

        assert result.status is Status.ERROR
        assert len(scripted.calls) == 1

    async def test_domain_drops_do_not_trigger_a_reask(self):
        """Items that fail binding are dropped from one reply with no second call."""
        models, scripted = fact_models(
            [reply(fact(evidence_quote="never said"), fact(entity_refs=["zz"]))]
        )

        result = await extract(extractor(models))

        assert result.status is Status.EMPTY
        assert result.facts == []
        assert {d.reason for d in result.dropped} == {
            FactDropReason.UNGROUNDED,
            FactDropReason.UNKNOWN_ENTITY_REF,
        }
        assert len(scripted.calls) == 1

    async def test_fact_output_cap_defaults_to_16384_tokens(self):
        """The fact role asks the provider for at most 16384 output tokens."""
        models, scripted = fact_models([reply()])

        await extract(extractor(models))

        settings = scripted.calls[0].info.model_settings or {}
        assert settings.get("max_tokens") == 16384

    async def test_document_without_candidates_is_skipped_with_a_stable_code(self):
        """A document below the candidate threshold never calls the model."""
        models, scripted = fact_models()

        result = await extract(
            extractor(models), make_episode(kind=Kind.DOCUMENT), entities=[]
        )

        assert result.status is Status.SKIPPED
        assert result.skip_reason is SkipReason.NO_CANDIDATES
        assert scripted.calls == []

    async def test_message_without_entities_still_runs_because_the_actor_is_a_candidate(
        self,
    ):
        """A message always has its speaker as a candidate."""
        models, _ = fact_models(
            [reply(fact(entity_refs=["actor"], evidence_quote="I switched"))]
        )

        result = await extract(extractor(models), entities=[])

        assert result.status is Status.OK
        assert result.facts[0].entity_refs == ("actor",)

    async def test_actor_candidate_is_added_even_when_ner_found_the_same_name(self):
        """The speaker is added under its own id, not merged by name."""
        models, _ = fact_models(
            [reply(fact(entity_refs=["actor"], evidence_quote="I switched"))]
        )

        result = await extract(extractor(models), entities=[ANA])

        assert result.status is Status.OK
        assert result.facts[0].entity_refs == ("actor",)

    async def test_force_runs_an_episode_with_no_candidates(self):
        """Force overrides the candidate threshold."""
        models, scripted = fact_models([reply()])

        result = await extract(
            extractor(models),
            make_episode(kind=Kind.DOCUMENT),
            entities=[],
            force=True,
        )

        assert result.status is Status.EMPTY
        assert len(scripted.calls) == 1

    async def test_min_entities_threshold_comes_from_settings(self):
        """Settings can require more than one candidate."""
        models, _ = fact_models()

        result = await extract(
            LLMFactExtractor(models, FactExtractionSettings(min_entities=3)),
            make_episode(kind=Kind.DOCUMENT),
            entities=[ANA, PG],
        )

        assert result.skip_reason is SkipReason.NO_CANDIDATES

    async def test_language_settings_skip_unsupported_episodes(self):
        """A language outside the configured set is skipped."""
        models, _ = fact_models()
        episode = make_episode(metadata={"language": "de"})

        result = await extract(
            LLMFactExtractor(
                models,
                FactExtractionSettings(languages=frozenset({"en"})),
            ),
            episode,
        )

        assert result.skip_reason is SkipReason.UNSUPPORTED_LANGUAGE

    async def test_extractor_settings_control_defaults_instructions_and_version(self):
        """Configured values affect binding, the prompt, and the status version."""
        settings = FactExtractionSettings(
            default_confidence=0.9,
            instructions="Keep only facts about databases.",
            version="fact-custom",
        )
        models, scripted = fact_models([reply(fact(confidence=None))])
        ex = LLMFactExtractor(models, settings)

        result = await extract(ex)

        assert result.facts[0].confidence == 0.9
        assert "Keep only facts about databases." in scripted.calls[0].prompt
        assert ex.version == "fact-custom"

    async def test_unknown_kind_raises(self):
        """An unknown kind is a programming error, not a skipped episode."""
        models, _ = fact_models()

        with pytest.raises(UnknownKindError, match="no LLM fact path"):
            await extract(extractor(models), make_episode(kind="telepathy"))

    async def test_previous_episodes_come_from_the_provider_oldest_first(self):
        """The prompt shows the earlier episodes in time order."""
        previous = (
            make_episode("earliest note", episode_id="p1"),
            make_episode("middle note", episode_id="p2"),
            make_episode("latest note", episode_id="p3"),
        )
        models, scripted = fact_models([reply()])

        await extract(extractor(models), previous=previous)

        prompt = scripted.calls[0].prompt
        assert (
            prompt.index("earliest note")
            < prompt.index("middle note")
            < prompt.index("latest note")
        )

    async def test_long_previous_episode_reaches_the_prompt_intact(self):
        """Earlier context is delivered whole, with no local fitting."""
        long_note = "Earlier note. " + "detail " * 800 + "end."
        previous = (make_episode(long_note, episode_id="p1"),)
        models, scripted = fact_models([reply()])

        await extract(extractor(models), previous=previous)

        prompt = scripted.calls[0].prompt
        assert long_note in prompt
        assert "I switched to Postgres" in prompt

    async def test_quote_found_only_in_a_previous_episode_is_dropped(self):
        """Earlier context cannot ground a fact's evidence."""
        previous = (
            make_episode("Earlier, Ana said she hated MySQL.", episode_id="p0"),
        )
        models, _ = fact_models([reply(fact(evidence_quote="she hated MySQL"))])

        result = await extract(extractor(models), previous=previous)

        (d,) = result.dropped
        assert d.reason is FactDropReason.UNGROUNDED
        assert d.detail == "quote found only in a previous episode"

    async def test_reference_time_in_the_result_is_the_episode_time(self):
        """The default start of a state is the episode time."""
        models, _ = fact_models([reply(fact(fact_kind="state"))])

        result = await extract(extractor(models))

        assert result.facts[0].valid_from == T0

    async def test_dropped_items_are_returned_on_the_result(self):
        """A bound drop stays on the result next to the kept facts."""
        models, _ = fact_models([reply(fact(), fact(evidence_quote="never said"))])

        result = await extract(extractor(models))

        assert len(result.facts) == 1
        assert [d.reason for d in result.dropped] == [FactDropReason.UNGROUNDED]
