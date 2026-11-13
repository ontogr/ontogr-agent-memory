"""Record facts, the kind router and the batch driver."""

from collections.abc import Sequence
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from pydantic_ai.exceptions import ModelHTTPError

from ontogr_mem.data_models import Kind, Method, Status
from ontogr_mem.fact_extraction import (
    FactDropReason,
    FactExtraction,
    FactExtractionSettings,
    FactKind,
    FactMapping,
    FactTemplate,
    InMemoryFactStore,
    LLMFactExtractor,
    RecordFactMapper,
    RoutingFactExtractor,
    SkipReason,
    TimeBasis,
    TypeView,
    UnknownKindError,
)
from tests.support.extraction import FakePrevious
from tests.support.facts import (
    ANA,
    PG,
    T0,
    ent,
    fact,
    fact_models,
    make_episode,
    reply,
)
from tests.support.scripted_model import Scripted


ORDER = {
    "customer": "Ana",
    "product": "Postgres",
    "status": "paid",
    "paid_on": "2024-03",
}
TEMPLATE = FactTemplate(
    "{customer} bought {product}",
    ("customer", "product"),
    entity_fields=("customer", "product"),
    time_field="paid_on",
    relation_type_label="USES",
)


def record_episode(payload: dict[str, object] | None = None, **kw: object):
    """Build a structured record episode."""
    return make_episode(
        "ignored",
        kind=Kind.RECORD,
        content=payload or ORDER,
        source_system="shop",
        metadata={"table": "orders"},
        **kw,  # type: ignore[arg-type]
    )


def mapper(template: FactTemplate = TEMPLATE) -> RecordFactMapper:
    """Build a mapper with one template for the orders table."""
    return RecordFactMapper({"shop/orders": FactMapping((template,))})


def router(
    replies: Sequence[Any] = (), **maps: FactMapping
) -> tuple[RoutingFactExtractor, Scripted]:
    """Build a router around a scripted model and record mappings."""
    models, scripted = fact_models(replies)
    extractor = RoutingFactExtractor(LLMFactExtractor(models), RecordFactMapper(maps))
    return extractor, scripted


def driver(
    replies: Sequence[Any], store: InMemoryFactStore | None = None
) -> tuple[FactExtraction, InMemoryFactStore, Scripted]:
    """Build a batch driver around a scripted model and return its recorder."""
    store = store or InMemoryFactStore()
    models, scripted = fact_models(replies)
    return FactExtraction(LLMFactExtractor(models), store), store, scripted


def by_episode(episodes, entities=None):
    """Build the entities map for a batch."""
    if entities is None:
        return {episode.id: [ANA, PG] for episode in episodes}
    return {episode.id: list(entities) for episode in episodes}


class TestRecordFacts:
    """The no-model path for structured episodes."""

    def test_record_is_rendered_with_field_paths_as_evidence(self, catalog) -> None:
        """A template turns fields into one grounded statement."""
        result = mapper().map(record_episode(), [ANA, PG], TypeView(catalog))

        (f,) = result.facts
        assert result.status is Status.OK
        assert f.text == "Ana bought Postgres"
        assert f.entity_refs == ("e0", "e1")
        assert f.relation_type_id == 0
        assert (f.method, f.confidence, f.evidence_span) == (
            Method.RECORD,
            1.0,
            None,
        )
        assert f.evidence_quote == "customer,product"
        assert f.fields["customer"] == "Ana"
        assert f.valid_from == datetime(2024, 3, 1, tzinfo=UTC)
        assert f.time_basis is TimeBasis.STATED
        assert f.fact_kind is FactKind.EVENT
        assert f.asserted_by is None

    def test_record_without_time_field_uses_the_record_time(self) -> None:
        """The record time is the reference when no field holds a time."""
        plain = FactTemplate("{customer} bought {product}", ("customer", "product"))

        (f,) = mapper(plain).map(record_episode(), [], None).facts

        assert (f.valid_from, f.time_basis) == (T0, TimeBasis.REFERENCE)

    def test_bad_record_time_is_cleared_and_recorded(self) -> None:
        """An unparsable time field clears the time and keeps the fact."""
        result = mapper().map(
            record_episode(ORDER | {"paid_on": "soon"}), [ANA, PG], None
        )

        (f,) = result.facts
        assert (f.valid_from, f.time_basis) == (None, TimeBasis.NONE)
        assert FactDropReason.TIME_UNPARSED in [d.reason for d in result.dropped]

    async def test_unmapped_record_is_skipped_never_sent_to_a_model(self) -> None:
        """A record with no mapping is skipped without a model call."""
        extractor, scripted = router(**{"shop/orders": FactMapping((TEMPLATE,))})
        episode = make_episode(
            "x", kind=Kind.RECORD, content={"a": 1}, source_system="crm"
        )

        result = await extractor.extract(episode, [], None)

        assert result.status is Status.SKIPPED
        assert result.skip_reason is SkipReason.UNMAPPED_RECORD
        assert scripted.calls == []

    def test_mapping_falls_back_from_table_to_system(self) -> None:
        """A system mapping covers every table of that system."""
        m = RecordFactMapper({"shop": FactMapping((TEMPLATE,))})

        assert m.map(record_episode(), [ANA, PG], None).status is Status.OK

    def test_missing_field_drops_that_template_with_a_reason(self) -> None:
        """A template with a missing field produces no fact."""
        result = mapper().map(record_episode({"customer": "Ana"}), [ANA], None)

        assert result.status is Status.EMPTY
        assert result.dropped[0].reason is FactDropReason.MISSING_FIELD
        assert result.dropped[0].detail == "product"

    def test_entity_field_with_no_or_several_matches_is_left_out(self) -> None:
        """An ambiguous or unknown field value leaves the ref out and is recorded."""
        twin = ent("e9", "postgres", 1)

        result = mapper().map(record_episode(), [ANA, PG, twin], None)

        (f,) = result.facts
        assert f.entity_refs == ("e0",)
        assert [d.reason for d in result.dropped] == [
            FactDropReason.AMBIGUOUS_ENTITY_REF,
            FactDropReason.TYPES_UNAVAILABLE,
        ]
        none = mapper().map(record_episode(), [ANA], None)
        assert none.dropped[0].reason is FactDropReason.UNKNOWN_ENTITY_REF

    async def test_events_take_the_record_path_too(self) -> None:
        """An event payload is mapped without a model."""
        episode = make_episode(
            "x",
            kind=Kind.EVENT,
            content=ORDER,
            source_system="shop",
            metadata={"table": "orders"},
        )
        extractor, _ = router(**{"shop/orders": FactMapping((TEMPLATE,))})

        result = await extractor.extract(episode, [ANA, PG], None)

        assert result.status is Status.OK

    async def test_router_sends_text_kinds_to_the_llm_and_unknown_kinds_raise(
        self,
    ) -> None:
        """Only structured kinds skip the model; an unknown kind is an error."""
        extractor, scripted = router([reply(fact())])

        result = await extractor.extract(make_episode(), [ANA, PG], None)

        assert result.status is Status.OK and len(scripted.calls) == 1
        unknown, _ = router()
        with pytest.raises(UnknownKindError):
            await unknown.extract(make_episode(kind="telepathy"), [], None)


class TestFactExtraction:
    """Resume, storage and the batch report."""

    async def test_stored_ok_is_not_called_again(self) -> None:
        """A final result comes back without another model call."""
        ex, _, scripted = driver([reply(fact())])
        episode = make_episode()

        first = await ex.extract([episode], by_episode([episode]))
        second = await ex.extract([episode], by_episode([episode]))

        assert len(scripted.calls) == 1
        assert second.results[0].facts == first.results[0].facts

    async def test_stored_empty_is_final_too(self) -> None:
        """An empty result is final, not retryable."""
        ex, _, scripted = driver([reply()])
        episode = make_episode()

        await ex.extract([episode], by_episode([episode]))
        await ex.extract([episode], by_episode([episode]))

        assert len(scripted.calls) == 1

    async def test_error_is_retried_and_the_retry_replaces_the_result(self) -> None:
        """A retryable error is called again and the stored error is replaced."""
        provider_down = ModelHTTPError(503, "fact-model", {"error": "down"})
        ex, _, _ = driver([provider_down, reply(fact())])
        episode = make_episode()

        first = await ex.extract([episode], by_episode([episode]))
        second = await ex.extract([episode], by_episode([episode]))

        assert first.retryable and first.failed_episode_ids == ["ep1"]
        assert second.succeeded
        assert second.results[0].status is Status.OK
        assert second.results[0].facts

    async def test_parse_failed_is_retried(self) -> None:
        """A schema failure is retryable."""
        ex, _, _ = driver(["x", "x", "x", reply(fact())])
        episode = make_episode()

        assert (await ex.extract([episode], by_episode([episode]))).retryable
        assert (await ex.extract([episode], by_episode([episode]))).succeeded

    async def test_skipped_is_re_evaluated_with_the_candidates_of_this_run(
        self,
    ) -> None:
        """A skip is not final: new candidates can make the episode runnable."""
        ex, _, _ = driver([reply(fact())])
        episode = make_episode(kind=Kind.DOCUMENT)

        first = await ex.extract([episode], {})
        second = await ex.extract([episode], by_episode([episode]))

        assert first.results[0].status is Status.SKIPPED
        assert second.results[0].status is Status.OK

    async def test_save_replaces_everything_of_the_episode(self) -> None:
        """A new extractor version replaces the older stored result."""
        store = InMemoryFactStore()
        models, scripted = fact_models([reply(fact()), reply(), reply(fact())])
        episode = make_episode()

        old = FactExtraction(LLMFactExtractor(models), store)
        await old.extract([episode], by_episode([episode]))
        new = FactExtraction(
            LLMFactExtractor(models, FactExtractionSettings(version="fact-2")),
            store,
        )
        await new.extract([episode], by_episode([episode]))

        assert len(scripted.calls) == 2
        await new.extract([episode], by_episode([episode]))
        assert len(scripted.calls) == 2

        # The old version's stored result is gone, so it runs the model again.
        third = await old.extract([episode], by_episode([episode]))
        assert len(scripted.calls) == 3
        assert third.results[0].status is Status.OK

    async def test_catalog_change_is_a_new_status_key(self, catalog) -> None:
        """The same episode is extracted again for a different catalog."""
        models, scripted = fact_models([reply(), reply()])
        store = InMemoryFactStore()
        episode = make_episode()
        plain = FactExtraction(LLMFactExtractor(models), store)
        typed = FactExtraction(LLMFactExtractor(models), store, TypeView(catalog))

        await plain.extract([episode], by_episode([episode]))
        await typed.extract([episode], by_episode([episode]))

        assert len(scripted.calls) == 2

    async def test_one_episode_failure_does_not_affect_another(self) -> None:
        """A failure is scoped to its episode."""
        provider_down = ModelHTTPError(503, "fact-model", {"error": "down"})
        models, _ = fact_models([provider_down, reply(fact())])
        ex = FactExtraction(
            LLMFactExtractor(models),
            InMemoryFactStore(),
            settings=FactExtractionSettings(concurrency=1),
        )
        a = make_episode(episode_id="a")
        b = make_episode(episode_id="b")

        report = await ex.extract([a, b], by_episode([a, b]))

        assert [r.status for r in report.results] == [Status.ERROR, Status.OK]
        assert report.failed_episode_ids == ["a"]

    async def test_provider_failure_in_a_batch_leaves_the_other_episodes_running(
        self,
    ) -> None:
        """A provider error ends its episode as ERROR; the rest still extract."""
        provider_down = ModelHTTPError(503, "fact-model", {"error": "down"})
        models, scripted = fact_models([reply(fact()), provider_down, reply(fact())])
        ex = FactExtraction(
            LLMFactExtractor(models),
            InMemoryFactStore(),
            settings=FactExtractionSettings(concurrency=1),
        )
        episodes = [make_episode(episode_id=name) for name in ("a", "b", "c")]

        report = await ex.extract(episodes, by_episode(episodes))

        assert [r.status for r in report.results] == [
            Status.OK,
            Status.ERROR,
            Status.OK,
        ]
        assert report.failed_episode_ids == ["b"]
        assert len(scripted.calls) == 3

    async def test_exhausted_malformed_replies_end_as_parse_failed_with_usage(
        self,
    ) -> None:
        """Three malformed replies end the episode as PARSE_FAILED, with usage."""
        ex, _, scripted = driver(["bad", "bad", "bad"])
        episode = make_episode()

        report = await ex.extract([episode], by_episode([episode]))

        (result,) = report.results
        assert result.status is Status.PARSE_FAILED
        assert len(scripted.calls) == 3
        assert result.usage.buckets["facts/fact_extraction"].calls == 3

    async def test_unknown_kind_is_not_swallowed_by_the_driver(self) -> None:
        """An unknown kind stops the batch."""
        extractor, _ = router()
        ex = FactExtraction(extractor, InMemoryFactStore())

        with pytest.raises(UnknownKindError):
            await ex.extract([make_episode(kind="telepathy")])

    async def test_remove_episode_clears_its_records(self) -> None:
        """Removing an episode makes the next run call the model again."""
        ex, store, scripted = driver([reply(fact()), reply(fact())])
        episode = make_episode()

        await ex.extract([episode], by_episode([episode]))
        await store.remove_episode("ep1")
        await ex.extract([episode], by_episode([episode]))

        assert len(scripted.calls) == 2

    async def test_previous_episodes_are_fetched_once_per_episode(self) -> None:
        """The context reader is called once for each episode in the batch."""
        previous = FakePrevious([make_episode("earlier note", episode_id="p1")])
        models, _ = fact_models([reply(), reply()])
        ex = FactExtraction(
            LLMFactExtractor(models),
            InMemoryFactStore(),
            previous=previous,
        )
        a = make_episode(episode_id="a")
        b = make_episode(episode_id="b")

        await ex.extract([a, b], by_episode([a, b]))

        assert previous.limits == [3, 3]

    async def test_max_previous_zero_does_not_read_previous_episodes(self) -> None:
        """Zero turns the context reader off."""
        previous = FakePrevious([make_episode("earlier note", episode_id="p1")])
        models, scripted = fact_models([reply()])
        ex = FactExtraction(
            LLMFactExtractor(models),
            InMemoryFactStore(),
            previous=previous,
            settings=FactExtractionSettings(max_previous=0),
        )
        episode = make_episode()

        await ex.extract([episode], by_episode([episode]))

        assert previous.limits == []
        assert "<previous_episodes>" not in scripted.calls[0].prompt
        assert "earlier note" not in scripted.calls[0].prompt

    async def test_long_previous_episodes_reach_the_prompt_intact(self) -> None:
        """Every earlier episode is delivered whole, with no local truncation."""
        texts = [f"Earlier note {i}. " + "x" * 5_000 for i in range(5)]
        earlier = [
            replace(
                make_episode(text, episode_id=f"p{i}"),
                occurred_at=T0 - timedelta(minutes=5 - i),
            )
            for i, text in enumerate(texts)
        ]
        previous = FakePrevious(earlier)
        models, scripted = fact_models([reply(fact())])
        ex = FactExtraction(
            LLMFactExtractor(models),
            InMemoryFactStore(),
            previous=previous,
            settings=FactExtractionSettings(max_previous=5),
        )
        episode = make_episode()

        report = await ex.extract([episode], by_episode([episode]))

        assert report.results[0].status is Status.OK
        prompt = scripted.calls[0].prompt
        for text in texts:
            assert text in prompt

    async def test_report_counts_statuses_drops_usage_and_skip_codes(self) -> None:
        """The report gives the counts a retry policy needs."""
        ex, _, _ = driver([reply(fact(), fact(evidence_quote="never said"))])
        normal = make_episode()
        d1 = make_episode(kind=Kind.DOCUMENT, episode_id="d1")
        d2 = make_episode(kind=Kind.DOCUMENT, episode_id="d2")

        report = await ex.extract(
            [normal, d1, d2],
            {normal.id: [ANA, PG], d1.id: [], d2.id: []},
        )

        assert report.status_counts() == {
            "ok": 1,
            "empty": 0,
            "parse_failed": 0,
            "skipped": 2,
            "error": 0,
        }
        assert report.skipped_reasons() == {"no_candidates": 2}
        drops = report.dropped_by_episode()
        assert drops["ep1"] == {"ungrounded": 1}
        assert all(not counts for episode, counts in drops.items() if episode != "ep1")
        assert report.usage().buckets["facts/fact_extraction"].calls == 1
        assert report.succeeded
