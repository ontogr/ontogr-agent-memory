"""Contract rules for sourced_facts requests and the HostFact item model."""

from datetime import UTC, datetime, timedelta

import pytest
from pydantic import ValidationError

from ontogr_mem.service.contracts import AddMemory, HostFact, validate_request
from ontogr_mem.service.errors import InvalidArgumentError


NOW = datetime(2026, 4, 1, 12, 0, tzinfo=UTC)

ANA_FACT = {
    "subject": "Ana",
    "subject_type": "Person",
    "relation": "works_at",
    "object": "Acme",
    "object_type": "Organization",
    "evidence_quote": "Ana works at Acme.",
}


def _sourced_request(**overrides):
    """Return a valid sourced_facts request body with the overrides applied."""
    request = {
        "kind": "sourced_facts",
        "content": "Ana works at Acme.",
        "facts": [ANA_FACT],
    }
    request.update(overrides)
    return request


class TestSourcedFactContracts:
    """Host facts are accepted only on sourced_facts and validate per item."""

    def test_request_with_one_complete_host_fact_validates(self):
        """A sourced_facts request keeps its host fact."""
        model = AddMemory.model_validate(_sourced_request())

        assert [fact.subject for fact in model.facts or []] == ["Ana"]

    def test_empty_facts_list_is_valid(self):
        """A source with no extracted facts is valid and keeps an empty list."""
        model = AddMemory.model_validate(_sourced_request(facts=[]))

        assert model.facts == []

    def test_request_with_exactly_1000_facts_validates(self):
        """A sourced_facts request may carry up to 1000 host facts."""
        model = AddMemory.model_validate(_sourced_request(facts=[ANA_FACT] * 1000))

        assert len(model.facts or []) == 1000

    def test_request_with_1001_facts_fails_validation(self):
        """A sourced_facts request with more than 1000 host facts is refused."""
        with pytest.raises(ValidationError) as excinfo:
            AddMemory.model_validate(_sourced_request(facts=[ANA_FACT] * 1001))

        assert [error["loc"] for error in excinfo.value.errors()] == [("facts",)]

    def test_request_with_1001_facts_raises_invalid_argument(self):
        """The service boundary reports too many facts as an invalid argument."""
        with pytest.raises(InvalidArgumentError, match="facts"):
            validate_request(AddMemory, _sourced_request(facts=[ANA_FACT] * 1001))

    def test_sourced_facts_without_facts_fails(self):
        """A sourced_facts request must send facts, even an empty list."""
        request = {"kind": "sourced_facts", "content": "Ana works at Acme."}

        with pytest.raises(ValidationError, match="kind=sourced_facts needs facts"):
            AddMemory.model_validate(request)

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("subject", "Ana"),
            ("relation", "works_at"),
            ("object", "Acme"),
            ("statement", "Ana works at Acme."),
            ("valid_from", NOW),
            ("valid_to", NOW + timedelta(days=1)),
            ("single_valued", False),
        ],
    )
    def test_triple_field_on_sourced_facts_fails(self, field, value):
        """Each fact triple field is refused next to facts."""
        with pytest.raises(
            ValidationError, match="kind=sourced_facts uses facts, not triple fields"
        ):
            AddMemory.model_validate(_sourced_request(facts=[], **{field: value}))

    @pytest.mark.parametrize(
        "request_body",
        [
            {"kind": "text", "content": "Ana works at Acme.", "facts": [ANA_FACT]},
            {
                "kind": "fact",
                "subject": "Ana",
                "relation": "works_at",
                "object": "Acme",
                "facts": [ANA_FACT],
            },
        ],
        ids=["text", "fact"],
    )
    def test_facts_on_another_kind_fails(self, request_body):
        """Facts are accepted only when the kind is sourced_facts."""
        with pytest.raises(
            ValidationError, match="facts is only valid for kind=sourced_facts"
        ):
            AddMemory.model_validate(request_body)

    @pytest.mark.parametrize("content", [None, "   "])
    def test_sourced_facts_without_content_fails(self, content):
        """The source text is required for sourced_facts, as it is for text."""
        with pytest.raises(
            ValidationError, match="kind=sourced_facts needs non-empty content"
        ):
            AddMemory.model_validate(_sourced_request(content=content, facts=[]))

    @pytest.mark.parametrize("missing_field", list(ANA_FACT))
    def test_host_fact_missing_a_required_field_fails(self, missing_field):
        """Every field of a host fact is required, including evidence_quote."""
        fact = {key: value for key, value in ANA_FACT.items() if key != missing_field}

        with pytest.raises(ValidationError) as excinfo:
            HostFact.model_validate(fact)

        assert [error["loc"] for error in excinfo.value.errors()] == [(missing_field,)]

    def test_host_fact_with_an_unknown_field_fails(self):
        """Unknown keys are refused rather than dropped."""
        with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
            HostFact.model_validate({**ANA_FACT, "confidence": 0.9})

    def test_host_fact_with_a_non_string_subject_fails(self):
        """A number is not a valid subject name."""
        with pytest.raises(ValidationError) as excinfo:
            HostFact.model_validate({**ANA_FACT, "subject": 5})

        assert [error["loc"] for error in excinfo.value.errors()] == [("subject",)]

    def test_host_fact_with_every_text_field_at_4096_characters_validates(self):
        """Each text field of a host fact may hold exactly 4096 characters."""
        text = "a" * 4096
        every_text_field = dict.fromkeys(ANA_FACT, text) | {"statement": text}

        fact = HostFact.model_validate(every_text_field)

        assert fact.model_dump(include=set(every_text_field)) == every_text_field

    @pytest.mark.parametrize(
        "field",
        [
            "subject",
            "subject_type",
            "relation",
            "object",
            "object_type",
            "evidence_quote",
            "statement",
        ],
    )
    def test_host_fact_text_field_of_4097_characters_fails(self, field):
        """A text field one character over 4096 is refused and named in the error."""
        with pytest.raises(ValidationError) as excinfo:
            HostFact.model_validate({**ANA_FACT, field: "a" * 4097})

        assert [error["loc"] for error in excinfo.value.errors()] == [(field,)]

    def test_over_long_host_fact_field_raises_invalid_argument(self):
        """An over-long host fact field reaches the caller as invalid argument."""
        fact = {**ANA_FACT, "evidence_quote": "a" * 4097}

        with pytest.raises(InvalidArgumentError, match="evidence_quote"):
            validate_request(AddMemory, _sourced_request(facts=[fact]))

    def test_host_fact_with_an_empty_subject_is_valid(self):
        """An empty name becomes a per-item drop later, not a request error."""
        fact = HostFact.model_validate({**ANA_FACT, "subject": ""})

        assert fact.subject == ""

    def test_z_suffixed_times_parse_to_utc(self):
        """ISO times ending in Z become timezone-aware UTC datetimes."""
        fact = HostFact.model_validate(
            {
                **ANA_FACT,
                "valid_from": "2026-04-01T12:00:00Z",
                "valid_to": "2026-05-01T12:00:00Z",
            }
        )

        assert fact.valid_from == datetime(2026, 4, 1, 12, tzinfo=UTC)
        assert fact.valid_to == datetime(2026, 5, 1, 12, tzinfo=UTC)
        assert fact.valid_from.utcoffset() == timedelta(0)

    @pytest.mark.parametrize("field", ["valid_from", "valid_to"])
    def test_naive_time_fails(self, field):
        """A time without a timezone offset is refused."""
        with pytest.raises(ValidationError, match="datetime must have a timezone"):
            HostFact.model_validate({**ANA_FACT, field: "2026-04-01T12:00:00"})

    def test_reversed_validity_window_on_an_item_is_kept_as_given(self):
        """A host fact keeps a reversed validity window unchanged."""
        fact = {**ANA_FACT, "valid_from": NOW, "valid_to": NOW - timedelta(days=1)}

        model = AddMemory.model_validate(_sourced_request(facts=[fact]))

        assert [(fact.valid_from, fact.valid_to) for fact in model.facts or []] == [
            (NOW, NOW - timedelta(days=1))
        ]
