"""Record facts, the kind router and the batch driver."""

from datetime import UTC, datetime

import pytest

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
from ontogr_mem.llm import LLMCallError
from tests.support.extraction import FakePrevious
from tests.support.facts import (
    ANA,
    PG,
    T0,
    ScriptedLLM,
    ent,
    fact,
    gate,
    make_episode,
    reply,
)


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


def router(llm: ScriptedLLM | None = None, **maps: FactMapping) -> RoutingFactExtractor:
    """Build a router around a scripted model and record mappings."""
    return RoutingFactExtractor(
        LLMFactExtractor(gate(llm or ScriptedLLM())), RecordFactMapper(maps)
    )


def driver(
    llm: ScriptedLLM, store: InMemoryFactStore | None = None
) -> tuple[FactExtraction, InMemoryFactStore]:
    """Build a batch driver around a scripted model."""
    store = store or InMemoryFactStore()
    return FactExtraction(LLMFactExtractor(gate(llm)), store), store


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
        llm = ScriptedLLM()
        episode = make_episode(
            "x", kind=Kind.RECORD, content={"a": 1}, source_system="crm"
        )

        result = await router(llm, **{"shop/orders": FactMapping((TEMPLATE,))}).extract(
            episode, [], None
        )

        assert result.status is Status.SKIPPED
        assert result.skip_reason is SkipReason.UNMAPPED_RECORD
        assert llm.requests == []

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

        result = await router(**{"shop/orders": FactMapping((TEMPLATE,))}).extract(
            episode, [ANA, PG], None
        )

        assert result.status is Status.OK

    async def test_router_sends_text_kinds_to_the_llm_and_unknown_kinds_raise(
        self,
    ) -> None:
        """Only structured kinds skip the model; an unknown kind is an error."""
        llm = ScriptedLLM([reply(fact())])

        result = await router(llm).extract(make_episode(), [ANA, PG], None)

        assert result.status is Status.OK and len(llm.requests) == 1
        with pytest.raises(UnknownKindError):
            await router().extract(make_episode(kind="telepathy"), [], None)


class TestFactExtraction:
    """Resume, storage and the batch report."""

    async def test_stored_ok_is_not_called_again(self) -> None:
        """A final result comes back without another model call."""
        llm = ScriptedLLM([reply(fact())])
        ex, _ = driver(llm)
        episode = make_episode()

        first = await ex.extract([episode], by_episode([episode]))
        second = await ex.extract([episode], by_episode([episode]))

        assert len(llm.requests) == 1
        assert second.results[0].facts == first.results[0].facts

    async def test_stored_empty_is_final_too(self) -> None:
        """An empty result is final, not retryable."""
        llm = ScriptedLLM([reply()])
        ex, _ = driver(llm)
        episode = make_episode()

        await ex.extract([episode], by_episode([episode]))
        await ex.extract([episode], by_episode([episode]))

        assert len(llm.requests) == 1

    async def test_error_is_retried_and_the_retry_replaces_the_result(self) -> None:
        """A retryable error is called again and the stored error is replaced."""
        llm = ScriptedLLM([LLMCallError("down")] * 3 + [reply(fact())])
        ex, _ = driver(llm)
        episode = make_episode()

        first = await ex.extract([episode], by_episode([episode]))
        second = await ex.extract([episode], by_episode([episode]))

        assert first.retryable and first.failed_episode_ids == ["ep1"]
        assert second.succeeded
        assert second.results[0].status is Status.OK
        assert second.results[0].facts

    async def test_parse_failed_is_retried(self) -> None:
        """A schema failure is retryable."""
        llm = ScriptedLLM(["x", "x", "x", reply(fact())])
        ex, _ = driver(llm)
        episode = make_episode()

        assert (await ex.extract([episode], by_episode([episode]))).retryable
        assert (await ex.extract([episode], by_episode([episode]))).succeeded

    async def test_skipped_is_re_evaluated_with_the_candidates_of_this_run(
        self,
    ) -> None:
        """A skip is not final: new candidates can make the episode runnable."""
        llm = ScriptedLLM([reply(fact())])
        ex, _ = driver(llm)
        episode = make_episode(kind=Kind.DOCUMENT)

        first = await ex.extract([episode], {})
        second = await ex.extract([episode], by_episode([episode]))

        assert first.results[0].status is Status.SKIPPED
        assert second.results[0].status is Status.OK

    async def test_save_replaces_everything_of_the_episode(self) -> None:
        """A new extractor version replaces the older stored result."""
        store = InMemoryFactStore()
        llm = ScriptedLLM([reply(fact()), reply(), reply(fact())])
        episode = make_episode()

        old, _ = driver(llm, store)
        await old.extract([episode], by_episode([episode]))
        new = FactExtraction(
            LLMFactExtractor(gate(llm), FactExtractionSettings(version="fact-2")),
            store,
        )
        await new.extract([episode], by_episode([episode]))

        assert len(llm.requests) == 2
        await new.extract([episode], by_episode([episode]))
        assert len(llm.requests) == 2

        # The old version's stored result is gone, so it runs the model again.
        third = await old.extract([episode], by_episode([episode]))
        assert len(llm.requests) == 3
        assert third.results[0].status is Status.OK

    async def test_catalog_change_is_a_new_status_key(self, catalog) -> None:
        """The same episode is extracted again for a different catalog."""
        llm = ScriptedLLM([reply(), reply()])
        store = InMemoryFactStore()
        episode = make_episode()
        plain = FactExtraction(LLMFactExtractor(gate(llm)), store)
        typed = FactExtraction(LLMFactExtractor(gate(llm)), store, TypeView(catalog))

        await plain.extract([episode], by_episode([episode]))
        await typed.extract([episode], by_episode([episode]))

        assert len(llm.requests) == 2

    async def test_one_episode_failure_does_not_affect_another(self) -> None:
        """A failure is scoped to its episode."""
        llm = ScriptedLLM([LLMCallError("down")] * 3 + [reply(fact())])
        ex = FactExtraction(
            LLMFactExtractor(gate(llm)),
            InMemoryFactStore(),
            settings=FactExtractionSettings(concurrency=1),
        )
        a = make_episode(episode_id="a")
        b = make_episode(episode_id="b")

        report = await ex.extract([a, b], by_episode([a, b]))

        assert [r.status for r in report.results] == [Status.ERROR, Status.OK]
        assert report.failed_episode_ids == ["a"]

    async def test_unknown_kind_is_not_swallowed_by_the_driver(self) -> None:
        """An unknown kind stops the batch."""
        ex = FactExtraction(router(), InMemoryFactStore())

        with pytest.raises(UnknownKindError):
            await ex.extract([make_episode(kind="telepathy")])

    async def test_remove_episode_clears_its_records(self) -> None:
        """Removing an episode makes the next run call the model again."""
        llm = ScriptedLLM([reply(fact()), reply(fact())])
        ex, store = driver(llm)
        episode = make_episode()

        await ex.extract([episode], by_episode([episode]))
        await store.remove_episode("ep1")
        await ex.extract([episode], by_episode([episode]))

        assert len(llm.requests) == 2

    async def test_previous_episodes_are_fetched_once_per_episode(self) -> None:
        """The context reader is called once for each episode in the batch."""
        previous = FakePrevious([make_episode("earlier note", episode_id="p1")])
        ex = FactExtraction(
            LLMFactExtractor(gate(ScriptedLLM([reply(), reply()]))),
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
        llm = ScriptedLLM([reply()])
        ex = FactExtraction(
            LLMFactExtractor(gate(llm)),
            InMemoryFactStore(),
            previous=previous,
            settings=FactExtractionSettings(max_previous=0),
        )
        episode = make_episode()

        await ex.extract([episode], by_episode([episode]))

        assert previous.limits == []
        assert "<previous_episodes>" not in llm.requests[0].prompt
        assert "earlier note" not in llm.requests[0].prompt

    async def test_oversize_episode_in_a_batch_is_skipped_and_retried(self) -> None:
        """An over-long episode is skipped, retried, and never blocks another."""
        llm = ScriptedLLM([reply(fact()), reply(fact())])
        store = InMemoryFactStore()
        big = make_episode(
            "I switched to Postgres. " + "x" * 100_000,
            kind=Kind.DOCUMENT,
            episode_id="big",
        )
        normal = make_episode()
        small_cap = FactExtraction(
            LLMFactExtractor(gate(llm, max_input_tokens=2000)), store
        )

        report = await small_cap.extract(
            [big, normal], {big.id: [ANA, PG], normal.id: [ANA, PG]}
        )

        assert [r.status for r in report.results] == [Status.SKIPPED, Status.OK]
        assert report.results[0].skip_reason is SkipReason.OVERSIZE
        assert report.results[1].facts

        larger_cap = FactExtraction(
            LLMFactExtractor(gate(llm, max_input_tokens=200_000)), store
        )
        retry = await larger_cap.extract([big], {big.id: [ANA, PG]})

        assert len(llm.requests) == 2
        assert retry.results[0].status is Status.OK

    async def test_report_counts_statuses_drops_usage_and_skip_codes(self) -> None:
        """The report gives the counts a retry policy needs."""
        llm = ScriptedLLM([reply(fact(), fact(evidence_quote="never said"))])
        ex, _ = driver(llm)
        normal = make_episode()
        d1 = make_episode(kind=Kind.DOCUMENT, episode_id="d1")
        d2 = make_episode(kind=Kind.DOCUMENT, episode_id="d2")
        big = make_episode("x" * 100_000, kind=Kind.DOCUMENT, episode_id="big")

        report = await ex.extract(
            [normal, d1, d2, big],
            {normal.id: [ANA, PG], d1.id: [], d2.id: [], big.id: [ANA]},
        )

        assert report.status_counts() == {
            "ok": 1,
            "empty": 0,
            "parse_failed": 0,
            "skipped": 3,
            "error": 0,
        }
        assert report.skipped_reasons() == {"no_candidates": 2, "oversize": 1}
        drops = report.dropped_by_episode()
        assert drops["ep1"] == {"ungrounded": 1}
        assert all(not counts for episode, counts in drops.items() if episode != "ep1")
        assert report.usage().buckets["facts/medium"].calls == 1
        assert report.succeeded
