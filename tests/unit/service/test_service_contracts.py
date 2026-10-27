"""Tests for the shared request and result models of every agent-facing verb."""

from datetime import UTC, datetime, timedelta, timezone

import pytest
from pydantic import ValidationError

from ontogr_mem.service import contracts
from ontogr_mem.service.contracts import (
    MAX_CONTENT_BYTES,
    AddMemory,
    ClearTenant,
    ConsolidateMemory,
    ContractModel,
    EntityOut,
    EntityRef,
    EntityView,
    EpisodeOut,
    EpisodePage,
    ErrorCode,
    ErrorInfo,
    FactOut,
    ForgetMemory,
    ForgetMode,
    ForgetResult,
    GetEntity,
    InvalidateMemory,
    JobState,
    JobView,
    ListEpisodes,
    MemoryContext,
    MemoryFeedback,
    MemoryKind,
    ScopedRequest,
    SearchMemory,
    StatusOut,
    UpdateMemory,
    WriteReceipt,
    WriteRequest,
    validate_request,
)
from ontogr_mem.service.errors import (
    InvalidArgumentError,
    PayloadTooLargeError,
    RateLimitedError,
)


NOW = datetime(2026, 4, 1, 12, 0, tzinfo=UTC)

WRITE_EXAMPLE = WriteReceipt(
    job_id="j1",
    status=JobState.RUNNING,
    tenant="t1",
    idempotent_replay=True,
    trace_id="tr1",
    result={"op": "add"},
    error=ErrorInfo(code=ErrorCode.CONFLICT, message="m", hint="h"),
)

FACT_EXAMPLE = FactOut(
    fact_id="f1",
    tenant="t1",
    statement="Alice lives in Paris",
    subject=EntityRef(entity_id="e1", name="Alice"),
    relation="lives_in",
    object=EntityRef(entity_id="e2", name="Paris"),
    valid_from=NOW,
    valid_to=NOW + timedelta(days=365),
    recorded_at=NOW,
    expired_at=NOW,
    status="active",
    disputed=True,
    counterpart_ids=["f2"],
    evidence_count=3,
    episode_ids=["e1"],
    score=0.9,
)

ENTITY_EXAMPLE = EntityOut(
    entity_id="e1",
    tenant="t1",
    name="Alice",
    type_label="Person",
    summary="A person",
    aliases=["Al", "Ally"],
)

EXAMPLES = (
    ErrorInfo(code=ErrorCode.CONFLICT, message="m", hint="h"),
    ScopedRequest(tenant="t1"),
    WriteRequest(tenant="t1", wait_seconds=30.0),
    AddMemory(
        tenant="t1",
        wait_seconds=5.0,
        kind=MemoryKind.FACT,
        content="Alice lives in Paris",
        subject="Alice",
        relation="lives_in",
        object="Paris",
        statement="Alice lives in Paris",
        valid_from=NOW,
        valid_to=NOW + timedelta(days=365),
        reference_time=NOW,
        single_valued=True,
        speaker="alice",
        source_description="chat",
        session_id="s1",
        thread="th1",
        idempotency_key="key-1",
    ),
    UpdateMemory(
        tenant="t1",
        wait_seconds=1.0,
        fact_id="f1",
        object="Paris",
        statement="Alice moved",
        effective_at=NOW,
        idempotency_key="key-2",
    ),
    ForgetMemory(
        tenant="t1",
        wait_seconds=1.0,
        fact_id="f1",
        mode=ForgetMode.ERASE,
        dry_run=True,
        confirm_token="tok",
        reason="privacy request",
    ),
    InvalidateMemory(tenant="t1", fact_id="f1", reason="stale", wait_seconds=1.0),
    ConsolidateMemory(tenant="t1", wait_seconds=2.0),
    ClearTenant(
        tenant="t1",
        wait_seconds=2.0,
        dry_run=True,
        confirm_token="tok",
    ),
    MemoryFeedback(tenant="t1", entity_id="e1", score=0.5, note="useful"),
    SearchMemory(
        tenant="t1",
        query="where does Alice live",
        valid_at=NOW,
        overlaps_start=NOW,
        overlaps_end=NOW + timedelta(days=1),
        known_at=NOW,
        graph_depth=2,
        disputed="exclude",
        mentions=["Alice"],
        budget_tokens=4000,
        format="json",
    ),
    GetEntity(
        tenant="t1",
        entity_id="e1",
        include_history=True,
        max_facts=50,
    ),
    ListEpisodes(tenant="t1", session_id="s1", cursor="c2", limit=50),
    WRITE_EXAMPLE,
    JobView(
        job_id="j1",
        tenant="t1",
        op="consolidate_memory",
        status=JobState.DONE,
        attempts=2,
        result={"stage": "dedup"},
        error=None,
        trace_id="tr1",
        created_at=NOW,
        updated_at=NOW,
    ),
    ForgetResult(
        mode=ForgetMode.DELETE,
        dry_run=False,
        fact_ids=["f1"],
        episode_ids=["e1"],
        collateral_fact_ids=["f2"],
        confirm_token="tok",
        receipt=WRITE_EXAMPLE,
        not_cleaned=["residue-node"],
    ),
    EntityRef(entity_id="e1", name="Alice"),
    FACT_EXAMPLE,
    ENTITY_EXAMPLE,
    MemoryContext(
        state="found",
        facts=[FACT_EXAMPLE],
        entities=[ENTITY_EXAMPLE],
        rendered="<memory_context></memory_context>",
        tokens_used=120,
        degraded=["vector"],
        trace_id="tr1",
    ),
    EntityView(entity=ENTITY_EXAMPLE, facts=[FACT_EXAMPLE]),
    EpisodeOut(
        episode_id="e1",
        tenant="t1",
        kind="message",
        session_id="s1",
        thread="th1",
        source_description="chat",
        job_id="j1",
        created_at=NOW,
    ),
    EpisodePage(
        episodes=[
            EpisodeOut(
                episode_id="e1",
                tenant="t1",
                kind="message",
                created_at=NOW,
            )
        ],
        next_cursor="c2",
    ),
    StatusOut(
        status="degraded",
        backend="falkordb",
        llm_available=False,
        tenants=["t1"],
        queued_jobs=3,
        detail={"lanes": ["vector"]},
    ),
)


def contract_models():
    """Every public contract model defined in the contracts module."""
    found: set[type] = set()
    pending = [ContractModel]
    while pending:
        for sub in pending.pop().__subclasses__():
            if sub.__module__ != contracts.__name__ or sub.__name__.startswith("_"):
                continue
            if sub not in found:
                found.add(sub)
                pending.append(sub)
    return found


class TestRoundTrip:
    """Every model survives a JSON round trip with its values unchanged."""

    @pytest.mark.parametrize("example", EXAMPLES, ids=lambda e: type(e).__name__)
    def test_model_survives_a_json_round_trip(self, example):
        """Parsing the JSON form gives back the same instance."""
        parsed = type(example).model_validate_json(example.model_dump_json())

        assert parsed == example

    def test_every_contract_model_has_an_example(self):
        """A new model without an example fails this test."""
        assert {type(example) for example in EXAMPLES} == contract_models()


class TestTimeAndTenantFields:
    """Datetimes must be timezone-aware and tenants must match the format."""

    def test_a_naive_datetime_fails_validation(self):
        """A datetime without a timezone is rejected."""
        with pytest.raises(ValidationError):
            AddMemory(
                kind=MemoryKind.TEXT,
                content="ok",
                reference_time=datetime(2026, 4, 1),
            )

    def test_an_aware_datetime_comes_back_in_utc(self):
        """An offset datetime is stored as the same instant in UTC."""
        model = AddMemory(
            kind=MemoryKind.TEXT,
            content="ok",
            reference_time=datetime(
                2026, 4, 1, 14, tzinfo=timezone(timedelta(hours=2))
            ),
        )

        assert model.reference_time == datetime(2026, 4, 1, 12, tzinfo=UTC)

    @pytest.mark.parametrize("tenant", ["bad tenant!", "", "*", "a" * 61])
    def test_a_bad_tenant_fails_validation(self, tenant):
        """A request tenant must be 1 to 60 allowed characters, no wildcard."""
        with pytest.raises(ValidationError):
            ScopedRequest(tenant=tenant)

    def test_a_sixty_character_tenant_passes(self):
        """The longest allowed tenant id validates."""
        assert ScopedRequest(tenant="a" * 60).tenant == "a" * 60


class TestAddMemory:
    """The per-kind field rules and the valid-time window of add_memory."""

    @pytest.mark.parametrize(
        "fields",
        [
            {"kind": MemoryKind.TEXT},
            {"kind": MemoryKind.MESSAGE, "content": "   "},
            {"kind": MemoryKind.FACT, "subject": "Alice"},
            {"kind": MemoryKind.FACT, "subject": "Alice", "relation": "lives_in"},
            {
                "kind": MemoryKind.TEXT,
                "content": "ok",
                "valid_from": NOW,
                "valid_to": NOW,
            },
            {
                "kind": MemoryKind.TEXT,
                "content": "ok",
                "valid_from": NOW,
                "valid_to": NOW - timedelta(days=1),
            },
        ],
    )
    def test_fields_that_do_not_match_the_kind_fail(self, fields):
        """Text needs content, fact needs its triple, and the window is open."""
        with pytest.raises(ValidationError):
            AddMemory(**fields)

    def test_kind_fact_names_the_missing_fields(self):
        """The failure names every field the fact triple is missing."""
        with pytest.raises(ValidationError) as excinfo:
            AddMemory(kind=MemoryKind.FACT)

        text = str(excinfo.value)
        assert "subject" in text
        assert "relation" in text
        assert "object" in text

    def test_content_of_exactly_the_byte_cap_passes(self):
        """The cap counts bytes and allows exactly that many."""
        model = AddMemory(kind=MemoryKind.TEXT, content="a" * MAX_CONTENT_BYTES)

        assert model.content is not None

    def test_content_over_the_cap_in_bytes_fails_even_below_in_chars(self):
        """A two-byte-per-char string over the byte cap is refused."""
        data = {"kind": "text", "content": "é" * 32768 + "a"}

        with pytest.raises(PayloadTooLargeError):
            validate_request(AddMemory, data)


class TestExactlyOneTarget:
    """Three requests must name exactly one of their two possible targets."""

    @pytest.mark.parametrize(
        ("kwargs", "ok"),
        [
            ({}, False),
            ({"fact_id": "f1", "episode_id": "e1"}, False),
            ({"fact_id": "f1"}, True),
            ({"episode_id": "e1"}, True),
        ],
    )
    def test_forget_names_exactly_one_target(self, kwargs, ok):
        """A forget either fails for zero or two targets or passes."""
        if ok:
            assert ForgetMemory(**kwargs).mode is ForgetMode.INVALIDATE
        else:
            with pytest.raises(ValidationError):
                ForgetMemory(**kwargs)

    @pytest.mark.parametrize(
        ("kwargs", "ok"),
        [
            ({}, False),
            ({"entity_id": "e1", "fact_id": "f1"}, False),
            ({"entity_id": "e1"}, True),
            ({"fact_id": "f1"}, True),
        ],
    )
    def test_feedback_names_exactly_one_target(self, kwargs, ok):
        """A score lands on one entity or one fact, never both or neither."""
        if ok:
            assert MemoryFeedback(score=0.0, **kwargs).score == 0.0
        else:
            with pytest.raises(ValidationError):
                MemoryFeedback(score=0.0, **kwargs)

    @pytest.mark.parametrize(
        ("kwargs", "ok"),
        [
            ({}, False),
            ({"entity_id": "e1", "name": "Alice"}, False),
            ({"entity_id": "e1"}, True),
            ({"name": "Alice"}, True),
        ],
    )
    def test_get_entity_names_exactly_one_target(self, kwargs, ok):
        """An entity lookup uses one id or one name, never both or neither."""
        if ok:
            assert GetEntity(**kwargs) is not None
        else:
            with pytest.raises(ValidationError):
                GetEntity(**kwargs)


class TestSearchMemory:
    """The bounds of the recall request."""

    def test_a_blank_query_fails(self):
        """The query must hold at least one character."""
        with pytest.raises(ValidationError):
            SearchMemory(query="")

    def test_a_two_thousand_character_query_passes(self):
        """The longest allowed query validates."""
        assert SearchMemory(query="a" * 2000).query == "a" * 2000

    def test_a_two_thousand_and_one_character_query_fails(self):
        """One character over the cap is refused."""
        with pytest.raises(ValidationError):
            SearchMemory(query="a" * 2001)

    @pytest.mark.parametrize("limit", [0, 101])
    def test_a_limit_out_of_range_fails(self, limit):
        """The limit runs from 1 to the result cap."""
        with pytest.raises(ValidationError):
            SearchMemory(query="q", limit=limit)

    def test_a_limit_at_the_cap_passes(self):
        """The result cap is an allowed limit."""
        assert SearchMemory(query="q", limit=100).limit == 100

    def test_max_results_is_not_a_field_name(self):
        """The retrieval field names are the wire names; max_results is not."""
        with pytest.raises(InvalidArgumentError):
            validate_request(SearchMemory, {"query": "q", "max_results": 5})

    def test_overlaps_start_at_or_after_overlaps_end_fails(self):
        """The overlap window runs from an earlier instant to a later one."""
        with pytest.raises(ValidationError):
            SearchMemory(query="q", overlaps_start=NOW, overlaps_end=NOW)

    def test_overlaps_end_without_overlaps_start_fails(self):
        """A window needs both ends."""
        with pytest.raises(ValidationError):
            SearchMemory(query="q", overlaps_end=NOW)

    @pytest.mark.parametrize("budget_tokens", [99, 32001])
    def test_a_budget_out_of_range_fails(self, budget_tokens):
        """The token budget runs from 100 to 32000."""
        with pytest.raises(ValidationError):
            SearchMemory(query="q", budget_tokens=budget_tokens)

    @pytest.mark.parametrize("budget_tokens", [100, 32000])
    def test_a_budget_at_the_bounds_passes(self, budget_tokens):
        """Both ends of the budget range are allowed."""
        assert (
            SearchMemory(query="q", budget_tokens=budget_tokens).budget_tokens
            == budget_tokens
        )


class TestValidateRequest:
    """The mapping-to-model entry point every door calls."""

    def test_a_valid_mapping_returns_the_model(self):
        """The requested type comes back built from the mapping."""
        model = validate_request(ScopedRequest, {"tenant": "t1"})

        assert model == ScopedRequest(tenant="t1")

    def test_an_unknown_key_fails_with_invalid_argument(self):
        """Extra fields are errors, not ignored."""
        with pytest.raises(InvalidArgumentError):
            validate_request(ScopedRequest, {"tenant": "t1", "extra": 1})

    def test_a_size_failure_is_payload_too_large(self):
        """Over-cap content reaches the caller as its own code."""
        data = {"kind": "text", "content": "a" * (MAX_CONTENT_BYTES + 1)}

        with pytest.raises(PayloadTooLargeError):
            validate_request(AddMemory, data)

    def test_a_size_failure_wins_over_other_errors(self):
        """The caller must shrink the content before the other fixes matter."""
        data = {
            "kind": "text",
            "content": "a" * (MAX_CONTENT_BYTES + 1),
            "extra": 1,
        }

        with pytest.raises(PayloadTooLargeError):
            validate_request(AddMemory, data)

    def test_the_error_hint_tells_the_caller_to_fix_the_named_fields(self):
        """Every validation failure carries the same actionable hint."""
        with pytest.raises(InvalidArgumentError) as excinfo:
            validate_request(ScopedRequest, {"tenant": ""})

        assert excinfo.value.hint == "Fix the named fields and call again."


class TestErrorInfo:
    """The wire form of a raised service error."""

    def test_from_error_keeps_code_message_and_hint(self):
        """The translation changes the shape, not the content."""
        info = ErrorInfo.from_error(RateLimitedError("m", hint="h"))

        assert info == ErrorInfo(code="rate_limited", message="m", hint="h")
