"""The model fact pass: attempts, skips, context and drops."""

import pytest

from ontogr_mem.data_models import Kind, Status
from ontogr_mem.fact_extraction import (
    FactDropReason,
    FactExtractionSettings,
    LLMFactExtractor,
    SkipReason,
    UnknownKindError,
)
from ontogr_mem.llm import LLMCallError
from tests.support.facts import (
    ANA,
    PG,
    T0,
    ScriptedLLM,
    fact,
    gate,
    make_episode,
    reply,
)


def extractor(llm: ScriptedLLM, **kw: object) -> LLMFactExtractor:
    """Build an extractor around a scripted client."""
    return LLMFactExtractor(gate(llm), **kw)  # type: ignore[arg-type]


async def extract(
    ex: LLMFactExtractor,
    episode=None,
    entities=None,
    types=None,
    previous=(),
    **kw: object,
):
    """Run the extractor with the test defaults."""
    return await ex.extract(
        episode or make_episode(),
        [ANA, PG] if entities is None else entities,
        types,
        previous,
        **kw,  # type: ignore[arg-type]
    )


def strict_flags(llm: ScriptedLLM) -> list[bool]:
    """Return whether every request asked for the strict schema."""
    return [
        request.schema is not None
        and request.schema.get("additionalProperties") is False
        for request in llm.requests
    ]


class TestLLMPass:
    """Behavior of the model fact pass."""

    async def test_ok_path_binds_facts_and_records_usage_by_stage_and_size(self):
        """A good reply becomes facts and one usage bucket."""
        llm = ScriptedLLM([reply(fact())])

        result = await extract(extractor(llm))

        assert result.status is Status.OK
        assert [f.text for f in result.facts] == ["Ana switched to Postgres"]
        bucket = result.usage.buckets["facts/medium"]
        assert (bucket.calls, bucket.output_tokens) == (1, 1)
        assert bucket.input_tokens == len(llm.requests[0].prompt)
        assert llm.requests[0].model_size.value == "medium"

    async def test_no_facts_is_empty_not_error(self):
        """An empty reply is a final, successful outcome."""
        result = await extract(extractor(ScriptedLLM([reply()])))

        assert result.status is Status.EMPTY
        assert result.facts == []

    async def test_chain_is_strict_then_non_strict_then_json_with_errors_fed_back(
        self,
    ):
        """A schema failure loosens the request and finally feeds errors back."""
        llm = ScriptedLLM(["not json", reply(fact(fact_kind="oops")), reply(fact())])

        result = await extract(extractor(llm))

        assert result.status is Status.OK
        assert strict_flags(llm) == [True, False, False]
        assert [r.schema is not None for r in llm.requests] == [True, True, False]
        assert "validation_errors" in llm.requests[2].prompt
        assert "fact_kind must be one of" in llm.requests[2].prompt

    async def test_invalid_json_after_the_chain_is_parse_failed_never_empty(self):
        """Three invalid replies are retryable, never an empty success."""
        llm = ScriptedLLM(["nope", "still nope", "no again"])

        result = await extract(extractor(llm))

        assert result.status is Status.PARSE_FAILED
        assert result.facts == []
        assert result.detail
        assert result.usage.buckets["facts/medium"].calls == 3

    async def test_provider_failure_moves_to_the_next_attempt(self):
        """A provider rejection is retried with a looser request."""
        llm = ScriptedLLM([LLMCallError("rejects strict"), reply(fact())])

        result = await extract(extractor(llm))

        assert result.status is Status.OK
        assert strict_flags(llm) == [True, False]

    async def test_all_provider_failures_are_error_apart_from_empty(self):
        """A provider that never answers ends as a retryable error."""
        llm = ScriptedLLM([LLMCallError("down")] * 3)

        result = await extract(extractor(llm))

        assert result.status is Status.ERROR
        assert "LLMCallError" in result.detail

    async def test_timeout_ends_the_episode_as_error_without_more_calls(self):
        """A timeout is not retried within the episode."""
        llm = ScriptedLLM([TimeoutError("slow")])

        result = await extract(extractor(llm))

        assert result.status is Status.ERROR
        assert len(llm.requests) == 1

    async def test_document_without_candidates_is_skipped_with_a_stable_code(self):
        """A document below the candidate threshold never calls the model."""
        llm = ScriptedLLM()

        result = await extract(
            extractor(llm), make_episode(kind=Kind.DOCUMENT), entities=[]
        )

        assert result.status is Status.SKIPPED
        assert result.skip_reason is SkipReason.NO_CANDIDATES
        assert llm.requests == []

    async def test_message_without_entities_still_runs_because_the_actor_is_a_candidate(
        self,
    ):
        """A message always has its speaker as a candidate."""
        llm = ScriptedLLM(
            [reply(fact(entity_refs=["actor"], evidence_quote="I switched"))]
        )

        result = await extract(extractor(llm), entities=[])

        assert result.status is Status.OK
        assert result.facts[0].entity_refs == ("actor",)

    async def test_actor_candidate_is_added_even_when_ner_found_the_same_name(self):
        """The speaker is added under its own id, not merged by name."""
        llm = ScriptedLLM(
            [reply(fact(entity_refs=["actor"], evidence_quote="I switched"))]
        )

        result = await extract(extractor(llm), entities=[ANA])

        assert result.status is Status.OK
        assert result.facts[0].entity_refs == ("actor",)

    async def test_force_runs_an_episode_with_no_candidates(self):
        """Force overrides the candidate threshold."""
        llm = ScriptedLLM([reply()])

        result = await extract(
            extractor(llm),
            make_episode(kind=Kind.DOCUMENT),
            entities=[],
            force=True,
        )

        assert result.status is Status.EMPTY
        assert len(llm.requests) == 1

    async def test_min_entities_threshold_comes_from_settings(self):
        """Settings can require more than one candidate."""
        result = await extract(
            LLMFactExtractor(
                gate(ScriptedLLM()), FactExtractionSettings(min_entities=3)
            ),
            make_episode(kind=Kind.DOCUMENT),
            entities=[ANA, PG],
        )

        assert result.skip_reason is SkipReason.NO_CANDIDATES

    async def test_language_settings_skip_unsupported_episodes(self):
        """A language outside the configured set is skipped."""
        episode = make_episode(metadata={"language": "de"})

        result = await extract(
            LLMFactExtractor(
                gate(ScriptedLLM()),
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
        llm = ScriptedLLM([reply(fact(confidence=None))])
        ex = LLMFactExtractor(gate(llm), settings)

        result = await extract(ex)

        assert result.facts[0].confidence == 0.9
        assert "Keep only facts about databases." in llm.requests[0].prompt
        assert ex.version == "fact-custom"

    async def test_unknown_kind_raises(self):
        """An unknown kind is a programming error, not a skipped episode."""
        with pytest.raises(UnknownKindError, match="no LLM fact path"):
            await extract(extractor(ScriptedLLM()), make_episode(kind="telepathy"))

    async def test_episode_that_cannot_fit_is_skipped_oversize_with_numbers(self):
        """An episode over the cap is skipped before any call."""
        llm = ScriptedLLM()
        ex = LLMFactExtractor(gate(llm, max_input_tokens=50))

        result = await extract(ex)

        assert result.status is Status.SKIPPED
        assert result.skip_reason is SkipReason.OVERSIZE
        assert "cap is 50" in result.detail
        assert llm.requests == []

    async def test_gate_level_oversize_is_also_skipped_not_error(self):
        """The gate's own cap produces the same skip."""
        llm = ScriptedLLM()
        ex = LLMFactExtractor(gate(llm, max_input_tokens=50), count_tokens=lambda _: 1)

        result = await extract(ex)

        assert result.status is Status.SKIPPED
        assert result.skip_reason is SkipReason.OVERSIZE

    async def test_only_the_repair_prompt_oversize_is_parse_failed_with_errors(self):
        """A repair prompt that cannot fit keeps the earlier schema errors."""
        llm = ScriptedLLM(["nope", "nope again"])
        ex = LLMFactExtractor(
            gate(llm),
            count_tokens=lambda s: 30_000 if "previous reply was invalid" in s else 1,
        )

        result = await extract(ex)

        assert result.status is Status.PARSE_FAILED
        assert "not valid JSON" in result.detail
        assert len(llm.requests) == 2

    async def test_repair_oversize_after_a_provider_failure_ends_as_error(self):
        """A provider failure before the repair keeps the episode retryable."""
        llm = ScriptedLLM(["nope", LLMCallError("down")])
        ex = LLMFactExtractor(
            gate(llm),
            count_tokens=lambda s: 30_000 if "previous reply was invalid" in s else 1,
        )

        result = await extract(ex)

        assert result.status is Status.ERROR
        assert "LLMCallError" in result.detail
        assert len(llm.requests) == 2

    async def test_oldest_previous_is_dropped_first_when_the_prompt_is_over_budget(
        self,
    ):
        """Older context goes first; the current episode is never cut."""
        big = "x" * 4000
        previous = (
            make_episode(big, episode_id="p1"),
            make_episode("recent context", episode_id="p2"),
        )
        llm = ScriptedLLM([reply()])
        ex = LLMFactExtractor(gate(llm, max_input_tokens=1500))

        result = await extract(ex, previous=previous)

        assert result.status is Status.EMPTY
        prompt = llm.requests[0].prompt
        assert "recent context" in prompt
        assert big not in prompt

    async def test_max_previous_larger_than_available_keeps_all(self):
        """The cap is a maximum, not a requirement."""
        previous = (make_episode("only earlier note", episode_id="p1"),)
        llm = ScriptedLLM([reply()])

        await extract(extractor(llm), previous=previous)

        assert "only earlier note" in llm.requests[0].prompt

    async def test_previous_episodes_come_from_the_provider_oldest_first(self):
        """The prompt shows the earlier episodes in time order."""
        previous = (
            make_episode("earliest note", episode_id="p1"),
            make_episode("middle note", episode_id="p2"),
            make_episode("latest note", episode_id="p3"),
        )
        llm = ScriptedLLM([reply()])

        await extract(extractor(llm), previous=previous)

        prompt = llm.requests[0].prompt
        assert (
            prompt.index("earliest note")
            < prompt.index("middle note")
            < prompt.index("latest note")
        )

    async def test_quote_found_only_in_a_previous_episode_is_dropped(self):
        """Earlier context cannot ground a fact's evidence."""
        previous = (
            make_episode("Earlier, Ana said she hated MySQL.", episode_id="p0"),
        )
        llm = ScriptedLLM([reply(fact(evidence_quote="she hated MySQL"))])

        result = await extract(extractor(llm), previous=previous)

        (d,) = result.dropped
        assert d.reason is FactDropReason.UNGROUNDED
        assert d.detail == "quote found only in a previous episode"

    async def test_reference_time_in_the_result_is_the_episode_time(self):
        """The default start of a state is the episode time."""
        llm = ScriptedLLM([reply(fact(fact_kind="state"))])

        result = await extract(extractor(llm))

        assert result.facts[0].valid_from == T0

    async def test_dropped_items_are_returned_on_the_result(self):
        """A bound drop stays on the result next to the kept facts."""
        llm = ScriptedLLM([reply(fact(), fact(evidence_quote="never said"))])

        result = await extract(extractor(llm))

        assert len(result.facts) == 1
        assert [d.reason for d in result.dropped] == [FactDropReason.UNGROUNDED]
