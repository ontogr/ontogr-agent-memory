"""Text-level checks and binding for facts a host extracted from its source text."""

import uuid
from datetime import UTC, datetime

import pytest

from ontogr_mem.data_models import Method, TypeView
from ontogr_mem.data_models.facts import Evidence
from ontogr_mem.fact_extraction.models import FactDropReason
from ontogr_mem.service.contracts import HostFact
from ontogr_mem.service.sourced_facts import (
    BindInput,
    BoundHostFacts,
    Drop,
    Endpoint,
    Grounded,
    GroundedFact,
    bind_host_facts,
    ground_host_facts,
)
from tests.support.facts import ent, fact_catalog, make_episode


SOURCE = "Ana works at Acme. Ana uses Postgres. Bob works at Acme. Bob likes Rust."
AMBIGUOUS_MESSAGE = "entity name used with more than one type"

EPISODE = make_episode("Ana works at Acme. Ana uses Postgres.")
ENTITIES = [ent("h0", "Ana", 0), ent("h1", "Acme", 2), ent("h2", "Postgres", 1)]
VIEW = TypeView(fact_catalog())

# Dense relation ids of `fact_catalog()`.
USES = 0
WORKS_FOR = 1


def host_fact(
    *,
    subject: str = "Ana",
    subject_type: str = "Person",
    relation: str = "WORKS_FOR",
    object_: str = "Acme",
    object_type: str = "Company",
    evidence_quote: str = "Ana works at Acme.",
    statement: str | None = None,
    valid_from: datetime | None = None,
    valid_to: datetime | None = None,
) -> HostFact:
    """Build a host fact that grounds in `SOURCE` unless a field is overridden."""
    return HostFact(
        subject=subject,
        subject_type=subject_type,
        relation=relation,
        object=object_,
        object_type=object_type,
        evidence_quote=evidence_quote,
        statement=statement,
        valid_from=valid_from,
        valid_to=valid_to,
    )


def uses_postgres() -> HostFact:
    """Build the second grounded fact of `SOURCE`."""
    return host_fact(
        relation="USES",
        object_="Postgres",
        object_type="Database",
        evidence_quote="Ana uses Postgres.",
    )


def bob_works_at_acme() -> HostFact:
    """Build the third grounded fact of `SOURCE`."""
    return host_fact(
        subject="Bob",
        evidence_quote="Bob works at Acme.",
    )


def bob_likes_rust() -> HostFact:
    """Build a grounded fact whose names appear in no other fact of `SOURCE`."""
    return host_fact(
        subject="Bob",
        relation="LIKES",
        object_="Rust",
        object_type="Language",
        evidence_quote="Bob likes Rust.",
    )


def bind_input(
    index: int = 0,
    *,
    fact: HostFact | None = None,
    subject_ref: str = "h0",
    object_ref: str = "h1",
    relation_type_id: int = WORKS_FOR,
) -> BindInput:
    """Build a bind input for Ana and Acme unless overridden."""
    return BindInput(
        index=index,
        fact=fact or host_fact(),
        subject_ref=subject_ref,
        object_ref=object_ref,
        relation_type_id=relation_type_id,
    )


def bind_items(*inputs: BindInput, view: TypeView | None = VIEW) -> BoundHostFacts:
    """Bind the inputs against the test episode and its three entities."""
    return bind_host_facts(EPISODE, inputs, ENTITIES, view)


class TestGroundHostFacts:
    """Host facts are checked against the source text and given shared refs."""

    def test_grounds_a_fully_valid_fact(self) -> None:
        """A fact whose quote and names are in the source gets two refs."""
        fact = host_fact()

        result = ground_host_facts(SOURCE, [fact])

        assert result == Grounded(
            facts=(GroundedFact(0, fact, "h0", "h1"),),
            endpoints=(
                Endpoint("h0", "Ana", "Person"),
                Endpoint("h1", "Acme", "Company"),
            ),
            drops=(),
        )

    def test_returns_empty_results_for_no_facts(self) -> None:
        """An empty input list gives empty facts, endpoints and drops."""
        assert ground_host_facts(SOURCE, []) == Grounded((), (), ())

    @pytest.mark.parametrize("blank", ["", "  \n\t "])
    @pytest.mark.parametrize(
        "field",
        [
            "subject",
            "subject_type",
            "relation",
            "object",
            "object_type",
            "evidence_quote",
        ],
    )
    def test_drops_a_fact_with_a_blank_required_field(
        self, field: str, blank: str
    ) -> None:
        """Each required field is checked for empty or whitespace-only text."""
        fact = host_fact().model_copy(update={field: blank})

        result = ground_host_facts(SOURCE, [fact])

        assert result == Grounded(
            (), (), (Drop(0, FactDropReason.MISSING_FIELD, field),)
        )

    def test_names_the_first_blank_field_in_the_documented_order(self) -> None:
        """A fact with several blank fields reports the earliest one."""
        fact = host_fact().model_copy(update={"object": "", "subject_type": " "})

        result = ground_host_facts(SOURCE, [fact])

        assert [drop.as_result() for drop in result.drops] == [
            {"index": 0, "reason": "missing_field", "detail": "subject_type"}
        ]

    def test_drops_a_fact_whose_quote_is_absent_from_the_source(self) -> None:
        """A quote that is not in the source text is ungrounded."""
        fact = host_fact(evidence_quote="Ana left Acme.")

        result = ground_host_facts(SOURCE, [fact])

        assert result == Grounded(
            (),
            (),
            (Drop(0, FactDropReason.UNGROUNDED, "quote not found in episode"),),
        )

    def test_drops_a_fact_whose_subject_is_absent_from_the_source(self) -> None:
        """A subject name that is not in the source text is ungrounded."""
        fact = host_fact(subject="Zed")

        result = ground_host_facts(SOURCE, [fact])

        assert [drop.as_result() for drop in result.drops] == [
            {
                "index": 0,
                "reason": "ungrounded",
                "detail": "subject not found in episode",
            }
        ]

    def test_drops_a_fact_whose_object_is_absent_from_the_source(self) -> None:
        """An object name that is not in the source text is ungrounded."""
        fact = host_fact(object_="Globex")

        result = ground_host_facts(SOURCE, [fact])

        assert [drop.as_result() for drop in result.drops] == [
            {
                "index": 0,
                "reason": "ungrounded",
                "detail": "object not found in episode",
            }
        ]

    def test_reports_the_subject_when_subject_and_object_are_both_absent(
        self,
    ) -> None:
        """The subject is checked before the object."""
        fact = host_fact(subject="Zed", object_="Globex")

        result = ground_host_facts(SOURCE, [fact])

        assert result.drops == (
            Drop(0, FactDropReason.UNGROUNDED, "subject not found in episode"),
        )

    def test_reports_the_quote_before_the_names(self) -> None:
        """The quote is checked before the subject and the object."""
        fact = host_fact(subject="Zed", evidence_quote="Zed works at Acme.")

        result = ground_host_facts(SOURCE, [fact])

        assert result.drops == (
            Drop(0, FactDropReason.UNGROUNDED, "quote not found in episode"),
        )

    def test_grounds_names_and_quote_that_differ_in_case_and_whitespace(
        self,
    ) -> None:
        """Case and runs of whitespace do not stop a fact from grounding."""
        fact = host_fact(
            subject="  ANA ",
            object_="acme",
            evidence_quote="ana   WORKS\nat acme.",
        )

        result = ground_host_facts(SOURCE, [fact])

        assert result == Grounded(
            facts=(GroundedFact(0, fact, "h0", "h1"),),
            endpoints=(
                Endpoint("h0", "ANA", "Person"),
                Endpoint("h1", "acme", "Company"),
            ),
            drops=(),
        )

    def test_grounds_a_decomposed_accent_against_a_precomposed_source(self) -> None:
        """A name written with a combining accent matches the NFC source text."""
        source = "Jos\N{LATIN SMALL LETTER E WITH ACUTE} works at Acme."
        fact = host_fact(
            subject="Jose\N{COMBINING ACUTE ACCENT}",
            evidence_quote="Jose\N{COMBINING ACUTE ACCENT} works at Acme.",
        )

        result = ground_host_facts(source, [fact])

        assert result.drops == ()
        assert result.endpoints[0] == Endpoint(
            "h0", "Jose\N{COMBINING ACUTE ACCENT}", "Person"
        )

    def test_shares_refs_between_facts_that_use_the_same_endpoint(self) -> None:
        """The same name and type give the same ref in every fact."""
        facts = [host_fact(), uses_postgres(), bob_works_at_acme()]

        result = ground_host_facts(SOURCE, facts)

        assert [(item.subject_ref, item.object_ref) for item in result.facts] == [
            ("h0", "h1"),
            ("h0", "h2"),
            ("h3", "h1"),
        ]
        assert result.endpoints == (
            Endpoint("h0", "Ana", "Person"),
            Endpoint("h1", "Acme", "Company"),
            Endpoint("h2", "Postgres", "Database"),
            Endpoint("h3", "Bob", "Person"),
        )

    def test_shares_a_ref_when_names_and_types_differ_only_in_case(self) -> None:
        """Endpoints are compared by normalized name and normalized type."""
        facts = [host_fact(), host_fact(object_="ACME", object_type=" company ")]

        result = ground_host_facts(SOURCE, facts)

        assert [(item.subject_ref, item.object_ref) for item in result.facts] == [
            ("h0", "h1"),
            ("h0", "h1"),
        ]
        assert result.endpoints == (
            Endpoint("h0", "Ana", "Person"),
            Endpoint("h1", "Acme", "Company"),
        )

    def test_numbers_refs_by_first_occurrence_across_facts(self) -> None:
        """Refs follow the first appearance, subject before object, in input order."""
        facts = [
            bob_works_at_acme(),
            host_fact(
                subject="Acme",
                subject_type="Company",
                relation="EMPLOYS",
                object_="Ana",
                object_type="Person",
                evidence_quote="Ana works at Acme.",
            ),
        ]

        result = ground_host_facts(SOURCE, facts)

        assert [(item.subject_ref, item.object_ref) for item in result.facts] == [
            ("h0", "h1"),
            ("h1", "h2"),
        ]
        assert [endpoint.name for endpoint in result.endpoints] == [
            "Bob",
            "Acme",
            "Ana",
        ]

    def test_keeps_original_indices_and_skips_refs_of_dropped_facts(self) -> None:
        """Dropped facts leave gaps in the indices but take no refs."""
        facts = [
            host_fact(subject="Zed", evidence_quote="Zed works at Acme."),
            host_fact(evidence_quote=""),
            bob_likes_rust(),
            host_fact(object_="Globex"),
            host_fact(),
        ]

        result = ground_host_facts(SOURCE, facts)

        assert [
            (item.index, item.subject_ref, item.object_ref) for item in result.facts
        ] == [
            (2, "h0", "h1"),
            (4, "h2", "h3"),
        ]
        assert [drop.as_result() for drop in result.drops] == [
            {
                "index": 0,
                "reason": "ungrounded",
                "detail": "quote not found in episode",
            },
            {"index": 1, "reason": "missing_field", "detail": "evidence_quote"},
            {
                "index": 3,
                "reason": "ungrounded",
                "detail": "object not found in episode",
            },
        ]

    @pytest.mark.parametrize("acme_first", [True, False])
    def test_drops_every_fact_that_uses_one_name_with_two_types(
        self, acme_first: bool
    ) -> None:
        """Both facts are dropped whichever one comes first, and others stay."""
        acme_as_object = host_fact()
        acme_as_subject = host_fact(
            subject="Acme",
            subject_type="Organization",
            relation="USES",
            object_="Postgres",
            object_type="Database",
            evidence_quote="Ana uses Postgres.",
        )
        facts = (
            [acme_as_object, acme_as_subject]
            if acme_first
            else [acme_as_subject, acme_as_object]
        )
        facts.append(bob_likes_rust())

        result = ground_host_facts(SOURCE, facts)

        assert result.drops == (
            Drop(0, FactDropReason.AMBIGUOUS_ENTITY_REF, AMBIGUOUS_MESSAGE),
            Drop(1, FactDropReason.AMBIGUOUS_ENTITY_REF, AMBIGUOUS_MESSAGE),
        )
        assert result.facts == (GroundedFact(2, bob_likes_rust(), "h0", "h1"),)
        assert result.endpoints == (
            Endpoint("h0", "Bob", "Person"),
            Endpoint("h1", "Rust", "Language"),
        )

    def test_ignores_the_type_of_a_name_in_a_fact_that_was_already_dropped(
        self,
    ) -> None:
        """A fact dropped for another reason does not make its names ambiguous."""
        facts = [
            host_fact(object_type="Organization", evidence_quote="Ana left Acme."),
            host_fact(),
        ]

        result = ground_host_facts(SOURCE, facts)

        assert result.drops == (
            Drop(0, FactDropReason.UNGROUNDED, "quote not found in episode"),
        )
        assert result.facts == (GroundedFact(1, host_fact(), "h0", "h1"),)

    def test_keeps_a_fact_that_relates_an_entity_to_itself(self) -> None:
        """A self relation is not dropped here and gets one ref on both sides."""
        fact = host_fact(
            object_="ana",
            object_type="PERSON",
            evidence_quote="Ana works at Acme.",
        )

        result = ground_host_facts(SOURCE, [fact])

        assert result.drops == ()
        assert result.facts == (GroundedFact(0, fact, "h0", "h0"),)
        assert result.endpoints == (Endpoint("h0", "Ana", "Person"),)

    def test_describes_a_drop_as_a_result_dict(self) -> None:
        """A drop turns into the plain dict returned to callers."""
        drop = Drop(3, FactDropReason.UNGROUNDED, "quote not found in episode")

        assert drop.as_result() == {
            "index": 3,
            "reason": "ungrounded",
            "detail": "quote not found in episode",
        }
        assert Drop(0, FactDropReason.SELF_RELATION).as_result() == {
            "index": 0,
            "reason": "self_relation",
            "detail": "",
        }


class TestBindHostFacts:
    """Grounded host facts are bound, merged and given ids without a model."""

    def test_binds_a_valid_item_into_a_record_fact(self) -> None:
        """A valid item becomes one fact located in the episode text."""
        result = bind_items(bind_input())

        assert result.drops == ()
        (fact,) = result.facts
        assert fact.method is Method.RECORD
        assert fact.evidence_quote == "Ana works at Acme."
        assert fact.evidence_span == (0, 18)
        assert fact.relation_type_id == WORKS_FOR
        assert fact.entity_refs == ("h0", "h1")
        assert fact.confidence == 1.0
        assert fact.episode_id == "ep1"
        assert uuid.UUID(fact.fact_id).version == 5

    def test_returns_nothing_for_no_inputs(self) -> None:
        """An empty input list gives empty facts and drops."""
        assert bind_items() == BoundHostFacts((), ())

    def test_drops_an_item_whose_quote_is_not_in_the_episode(self) -> None:
        """A quote that the episode does not contain is ungrounded."""
        item = bind_input(fact=host_fact(evidence_quote="Ana left Acme."))

        result = bind_items(item)

        assert result == BoundHostFacts(
            (),
            (Drop(0, FactDropReason.UNGROUNDED, "quote not found in episode"),),
        )

    def test_drops_an_item_that_links_an_entity_to_itself(self) -> None:
        """The same ref on both sides is a self relation."""
        item = bind_input(subject_ref="h0", object_ref="h0")

        result = bind_items(item)

        assert result == BoundHostFacts(
            (), (Drop(0, FactDropReason.SELF_RELATION, "h0"),)
        )

    def test_drops_an_item_whose_direction_the_catalog_refuses(self) -> None:
        """A database cannot be the source of USES, so the item is dropped."""
        item = bind_input(
            fact=uses_postgres(),
            subject_ref="h2",
            object_ref="h0",
            relation_type_id=USES,
        )

        result = bind_items(item)

        assert result == BoundHostFacts(
            (), (Drop(0, FactDropReason.WRONG_DIRECTION, "relation type 0"),)
        )

    @pytest.mark.parametrize(
        ("relation_type_id", "view", "reason", "detail"),
        [
            (99, VIEW, FactDropReason.BAD_TYPE_ID, "relation type 99"),
            (WORKS_FOR, None, FactDropReason.TYPES_UNAVAILABLE, "id 1"),
        ],
    )
    def test_drops_an_item_whose_relation_type_would_be_cleared(
        self,
        relation_type_id: int,
        view: TypeView | None,
        reason: FactDropReason,
        detail: str,
    ) -> None:
        """A type problem drops the item instead of clearing the type."""
        item = bind_input(relation_type_id=relation_type_id)

        result = bind_items(item, view=view)

        assert result == BoundHostFacts((), (Drop(0, reason, detail),))

    def test_drops_an_item_whose_window_ends_before_it_starts(self) -> None:
        """A reversed validity window drops the item and returns no fact."""
        fact = host_fact(
            valid_from=datetime(2024, 3, 1, tzinfo=UTC),
            valid_to=datetime(2024, 1, 1, tzinfo=UTC),
        )

        result = bind_items(bind_input(fact=fact))

        assert result == BoundHostFacts(
            (), (Drop(0, FactDropReason.TIME_ORDER, "valid_to < valid_from"),)
        )

    def test_keeps_a_timed_item_with_its_times_and_the_quote_as_time_evidence(
        self,
    ) -> None:
        """A valid window is kept and the quote serves as its time expression."""
        fact = host_fact(
            valid_from=datetime(2024, 1, 1, tzinfo=UTC),
            valid_to=datetime(2024, 3, 1, tzinfo=UTC),
        )

        result = bind_items(bind_input(fact=fact))

        assert result.drops == ()
        (bound,) = result.facts
        assert bound.valid_from == datetime(2024, 1, 1, tzinfo=UTC)
        assert bound.valid_to == datetime(2024, 3, 1, tzinfo=UTC)
        assert bound.time_expression == "Ana works at Acme."

    def test_keeps_an_item_that_has_only_a_start(self) -> None:
        """An open-ended window is kept with its start."""
        fact = host_fact(valid_from=datetime(2024, 1, 1, tzinfo=UTC))

        result = bind_items(bind_input(fact=fact))

        assert result.drops == ()
        (bound,) = result.facts
        assert bound.valid_from == datetime(2024, 1, 1, tzinfo=UTC)
        assert bound.valid_to is None

    def test_leaves_an_untimed_item_without_a_time_expression(self) -> None:
        """Without times the fact starts at the episode time and has no phrase."""
        result = bind_items(bind_input())

        (bound,) = result.facts
        assert bound.time_expression is None
        assert bound.valid_from == EPISODE.occurred_at

    def test_uses_the_statement_when_it_has_text(self) -> None:
        """The host's own sentence becomes the fact text."""
        item = bind_input(fact=host_fact(statement="Ana is employed by Acme."))

        result = bind_items(item)

        assert [fact.text for fact in result.facts] == ["Ana is employed by Acme."]

    @pytest.mark.parametrize("statement", [None, "", "  \n "])
    def test_builds_a_default_sentence_without_a_usable_statement(
        self, statement: str | None
    ) -> None:
        """A missing or blank statement is replaced by a sentence from the triple."""
        item = bind_input(fact=host_fact(statement=statement))

        result = bind_items(item)

        assert [fact.text for fact in result.facts] == ["Ana works for Acme"]

    def test_merges_identical_items_and_reports_the_later_index(self) -> None:
        """The second identical item is dropped as a duplicate of the first."""
        result = bind_items(bind_input(2), bind_input(5))

        assert len(result.facts) == 1
        assert result.facts[0].extra_evidence == ()
        assert result.drops == (
            Drop(5, FactDropReason.DUPLICATE_MERGED, "same fact as input index 2"),
        )

    def test_keeps_another_quote_of_a_duplicate_as_extra_evidence_once(self) -> None:
        """A different quote is added once, however many duplicates repeat it."""
        shorter = host_fact(evidence_quote="Ana works at Acme")
        result = bind_items(
            bind_input(0),
            bind_input(1, fact=shorter),
            bind_input(2, fact=shorter),
            bind_input(3),
        )

        (fact,) = result.facts
        assert fact.evidence_quote == "Ana works at Acme."
        assert fact.extra_evidence == (Evidence("Ana works at Acme", (0, 17)),)
        assert [drop.as_result() for drop in result.drops] == [
            {
                "index": 1,
                "reason": "duplicate_merged",
                "detail": "same fact as input index 0",
            },
            {
                "index": 2,
                "reason": "duplicate_merged",
                "detail": "same fact as input index 0",
            },
            {
                "index": 3,
                "reason": "duplicate_merged",
                "detail": "same fact as input index 0",
            },
        ]

    def test_keeps_facts_with_different_endpoints_separate(self) -> None:
        """Items that differ in refs are different facts even with equal text."""
        same_text = host_fact(statement="Ana is linked to it.")
        other = host_fact(
            statement="Ana is linked to it.",
            relation="USES",
            object_="Postgres",
            object_type="Database",
            evidence_quote="Ana uses Postgres.",
        )

        result = bind_items(
            bind_input(0, fact=same_text),
            bind_input(1, fact=other, object_ref="h2", relation_type_id=USES),
        )

        assert result.drops == ()
        assert [fact.entity_refs for fact in result.facts] == [
            ("h0", "h1"),
            ("h0", "h2"),
        ]

    def test_gives_different_facts_different_ids_that_are_stable_across_runs(
        self,
    ) -> None:
        """Ids are unique inside a result and repeat for the same input."""
        items = (
            bind_input(0),
            bind_input(
                1,
                fact=uses_postgres(),
                object_ref="h2",
                relation_type_id=USES,
            ),
        )

        first = bind_items(*items)
        second = bind_items(*items)

        first_ids = [fact.fact_id for fact in first.facts]
        assert len(set(first_ids)) == 2
        assert first_ids == [fact.fact_id for fact in second.facts]

    def test_reports_drops_in_index_order_next_to_kept_facts(self) -> None:
        """Kept facts and drops of one call keep the caller's indices."""
        result = bind_items(
            bind_input(4),
            bind_input(7, fact=host_fact(evidence_quote="Ana left Acme.")),
            bind_input(9, subject_ref="h1", object_ref="h1"),
            bind_input(11),
        )

        assert [fact.entity_refs for fact in result.facts] == [("h0", "h1")]
        assert result.drops == (
            Drop(7, FactDropReason.UNGROUNDED, "quote not found in episode"),
            Drop(9, FactDropReason.SELF_RELATION, "h1"),
            Drop(11, FactDropReason.DUPLICATE_MERGED, "same fact as input index 4"),
        )
