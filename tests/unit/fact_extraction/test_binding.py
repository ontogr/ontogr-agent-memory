"""Binding parsed facts to candidates, types and the episode."""

from datetime import UTC, datetime

import pytest

from ontogr_mem.data_models import Kind, Method
from ontogr_mem.data_models.facts import FactKind, TimeBasis
from ontogr_mem.fact_extraction import FactDropReason, TypeMode, TypeView
from ontogr_mem.fact_extraction.binding import bind
from ontogr_mem.fact_extraction.parsing import SchemaError, parse_facts
from tests.support.facts import ACME, ANA, PG, fact, make_episode, reply


def bind_facts(
    *facts: dict[str, object],
    episode=None,
    view: TypeView | None = None,
    previous=(),
):
    """Parse and bind facts against the test candidates."""
    episode = episode or make_episode()
    return bind(
        episode,
        [ANA, PG, ACME],
        parse_facts(reply(*facts)),
        view,
        default_confidence=0.5,
        previous=previous,
    )


def reasons(bound) -> list[FactDropReason]:
    """Return the reason of every drop."""
    return [d.reason for d in bound.dropped]


class TestBinding:
    """Grounding, refs, relation types, times, duplicates and provenance."""

    def test_quote_is_matched_ignoring_case_and_whitespace_and_stored_from_episode(
        self,
    ) -> None:
        """The stored quote is the episode's own text at the matched span."""
        bound = bind_facts(fact(evidence_quote="i  SWITCHED\nto postgres"))
        (f,) = bound.facts
        assert f.evidence_quote == "I switched to Postgres"
        assert f.evidence_span == (0, 22)

    def test_ungrounded_quote_drops_the_fact_with_a_reason(self) -> None:
        """A quote that is not in the episode drops the fact."""
        bound = bind_facts(fact(evidence_quote="I moved to MySQL"))
        assert bound.facts == []
        (d,) = bound.dropped
        assert d.reason is FactDropReason.UNGROUNDED
        assert d.detail == "quote not found in episode"

    def test_quote_found_only_in_a_previous_episode_is_ungrounded_and_says_so(
        self,
    ) -> None:
        """A quote from earlier context cannot ground a fact."""
        earlier = make_episode("Earlier, Ana said she hated MySQL.", episode_id="ep0")
        bound = bind_facts(fact(evidence_quote="she hated MySQL"), previous=(earlier,))
        assert bound.facts == []
        assert bound.dropped[0].detail == "quote found only in a previous episode"

    def test_unknown_entity_ref_drops_the_fact(self) -> None:
        """Every ref must come from the candidate list."""
        bound = bind_facts(fact(entity_refs=["e0", "e9"]))
        assert bound.facts == []
        assert reasons(bound) == [FactDropReason.UNKNOWN_ENTITY_REF]

    @pytest.mark.parametrize("refs", [[], ["e0"], ["e0", "e1", "e2"]])
    def test_facts_may_have_any_number_of_refs(self, refs: list[str]) -> None:
        """A fact can mention any number of candidates."""
        (f,) = bind_facts(fact(entity_refs=refs)).facts
        assert f.entity_refs == tuple(refs)

    def test_repeated_refs_are_collapsed_in_order(self) -> None:
        """The first occurrence of a ref decides its position."""
        (f,) = bind_facts(fact(entity_refs=["e1", "e0", "e1"])).facts
        assert f.entity_refs == ("e1", "e0")

    def test_empty_text_is_dropped(self) -> None:
        """A fact with no text is not a claim."""
        bound = bind_facts(fact(text="   "))
        assert bound.facts == []
        assert reasons(bound) == [FactDropReason.EMPTY_TEXT]

    @pytest.mark.parametrize("mode", list(TypeMode))
    def test_valid_type_is_kept_in_every_mode(self, catalog, mode: TypeMode) -> None:
        """A valid dense id is kept whatever the mode."""
        bound = bind_facts(fact(relation_type_id=0), view=TypeView(catalog, mode))
        assert bound.facts[0].relation_type_id == 0
        assert bound.dropped == []

    @pytest.mark.parametrize("mode", [TypeMode.STRICT, TypeMode.PREFERRED])
    def test_wrong_direction_clears_the_type_but_keeps_the_fact(
        self, catalog, mode: TypeMode
    ) -> None:
        """A direction problem clears the type and keeps the claim."""
        bound = bind_facts(
            fact(
                entity_refs=["e1", "e0"], relation_type_id=0, evidence_quote="Postgres"
            ),
            view=TypeView(catalog, mode),
        )
        (f,) = bound.facts
        assert f.relation_type_id is None
        assert reasons(bound) == [FactDropReason.WRONG_DIRECTION]

    def test_open_mode_does_not_enforce_direction(self, catalog) -> None:
        """Open mode treats patterns as advisory."""
        bound = bind_facts(
            fact(
                entity_refs=["e1", "e0"], relation_type_id=0, evidence_quote="Postgres"
            ),
            view=TypeView(catalog, TypeMode.OPEN),
        )
        assert bound.facts[0].relation_type_id == 0

    @pytest.mark.parametrize(
        ("refs", "type_id", "reason"),
        [
            (["e0", "e1"], 9, FactDropReason.BAD_TYPE_ID),
            (["e0"], 0, FactDropReason.BAD_TYPE_ID),
            (["e0", "e1", "e2"], 0, FactDropReason.BAD_TYPE_ID),
            (["e0", "e0"], 0, FactDropReason.SELF_RELATION),
        ],
    )
    def test_strict_mode_keeps_the_fact_and_records_why(
        self, catalog, refs: list[str], type_id: int, reason: FactDropReason
    ) -> None:
        """An unusable type is cleared, not fatal."""
        bound = bind_facts(
            fact(entity_refs=refs, relation_type_id=type_id),
            view=TypeView(catalog, TypeMode.STRICT),
        )
        assert len(bound.facts) == 1
        assert bound.facts[0].relation_type_id is None
        assert reasons(bound) == [reason]

    def test_type_without_a_catalog_is_cleared(self) -> None:
        """A dense id is meaningless without the catalog that issued it."""
        bound = bind_facts(fact(relation_type_id=0), view=None)
        assert bound.facts[0].relation_type_id is None
        assert reasons(bound) == [FactDropReason.TYPES_UNAVAILABLE]

    def test_bad_time_keeps_the_fact_and_records_the_reason(self) -> None:
        """A bad time clears the time fields and keeps the claim."""
        bound = bind_facts(
            fact(
                fact_kind="state",
                valid_from="soon",
                time_expression="in March 2024",
            )
        )
        (f,) = bound.facts
        assert (f.valid_from, f.time_basis) == (None, TimeBasis.NONE)
        assert FactDropReason.TIME_UNPARSED in reasons(bound)

    def test_present_tense_state_uses_the_episode_time_as_reference(self) -> None:
        """An actual state with no stated start begins at the episode time."""
        (f,) = bind_facts(fact(fact_kind="state")).facts
        assert f.valid_from == datetime(2026, 10, 8, 12, tzinfo=UTC)
        assert f.time_basis is TimeBasis.REFERENCE
        assert f.fact_kind is FactKind.STATE

    def test_stated_time_keeps_precision_and_expression(self) -> None:
        """A stated month keeps month precision and its phrase."""
        (f,) = bind_facts(
            fact(valid_from="2024-03", precision="month", time_expression="March 2024")
        ).facts
        assert f.valid_from == datetime(2024, 3, 1, tzinfo=UTC)
        assert f.precision is not None and f.precision.value == "month"
        assert f.time_expression == "March 2024"

    def test_exact_duplicates_merge_and_keep_the_other_quote(self) -> None:
        """A duplicate is folded into the first fact as extra evidence."""
        bound = bind_facts(
            fact(),
            fact(evidence_quote="switched to Postgres"),
        )
        (f,) = bound.facts
        assert [e.quote for e in f.extra_evidence] == ["switched to Postgres"]
        assert reasons(bound) == [FactDropReason.DUPLICATE_MERGED]

    def test_same_text_with_different_time_stays_separate(self) -> None:
        """The time window is part of a fact's identity."""
        bound = bind_facts(
            fact(fact_kind="state", valid_from="2024", time_expression="2024"),
            fact(fact_kind="state"),
        )
        assert len(bound.facts) == 2
        assert bound.facts[0].fact_id != bound.facts[1].fact_id

    def test_fact_ids_are_stable_across_runs_and_differ_per_episode(self) -> None:
        """Ids are deterministic and include the episode."""
        a = bind_facts(fact()).facts[0].fact_id
        b = bind_facts(fact()).facts[0].fact_id
        c = bind_facts(fact(), episode=make_episode(episode_id="ep2")).facts[0].fact_id
        assert a == b
        assert a != c

    def test_llm_facts_carry_provenance_and_default_confidence(self) -> None:
        """A model fact names its episode, speaker, method and default confidence."""
        (f,) = bind_facts(fact()).facts
        assert (f.episode_id, f.method) == ("ep1", Method.LLM)
        assert f.asserted_by == "u1"
        assert f.confidence == 0.5

    def test_document_facts_have_no_asserted_by(self) -> None:
        """Only a message has a speaker."""
        (f,) = bind_facts(fact(), episode=make_episode(kind=Kind.DOCUMENT)).facts
        assert f.asserted_by is None

    def test_model_confidence_is_kept(self) -> None:
        """A model confidence replaces the default."""
        (f,) = bind_facts(fact(confidence=0.9)).facts
        assert f.confidence == 0.9

    @pytest.mark.parametrize("bad", [1.5, -0.1, "high", True])
    def test_out_of_range_confidence_is_a_schema_error(self, bad: object) -> None:
        """A confidence outside 0..1 is rejected before binding."""
        with pytest.raises(SchemaError) as exc:
            parse_facts(reply(fact(confidence=bad)))
        assert "confidence" in exc.value.errors[0]

    def test_termination_is_kept_as_a_negated_actual_state_at_the_reference_time(
        self,
    ) -> None:
        """A termination is a negated actual state."""
        (f,) = bind_facts(
            fact(fact_kind="state", polarity="negated", evidence_quote="I switched")
        ).facts
        assert f.polarity.value == "negated"
        assert f.modality.value == "actual"
        assert f.time_basis is TimeBasis.REFERENCE

    @pytest.mark.parametrize("modality", ["planned", "conditional", "desired"])
    def test_plans_conditions_and_wishes_are_kept_and_not_dated(
        self, modality: str
    ) -> None:
        """A non-actual modality is kept and gets no reference time."""
        (f,) = bind_facts(
            fact(fact_kind="state", modality=modality, evidence_quote="I switched")
        ).facts
        assert f.modality.value == modality
        assert (f.valid_from, f.time_basis) == (None, TimeBasis.NONE)

    def test_label_field_parses(self) -> None:
        """A free-text label is kept beside a null dense id."""
        (raw,) = parse_facts(reply(fact(relation_label="uses")))
        (f,) = bind_facts(fact(relation_label="uses")).facts
        assert (raw.relation_label, raw.relation_type_id) == ("uses", None)
        assert (f.relation_label, f.relation_type_id) == ("uses", None)

    def test_label_requires_string_or_null(self) -> None:
        """A label that is not a string is a schema error."""
        with pytest.raises(SchemaError) as exc:
            parse_facts(reply(fact(relation_label=5)))
        assert "relation_label" in exc.value.errors[0]
