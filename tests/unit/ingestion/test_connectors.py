"""Tests for connectors, ids and validation."""

from datetime import UTC, datetime, timedelta, timezone

import pytest

from ontogr_mem.data_models import Actor, EpisodeDraft, Kind, Role, Source
from ontogr_mem.ingestion import (
    ChatMessage,
    DocumentInput,
    DraftValidationError,
    EventInput,
    RecordInput,
    ToolOutputInput,
    UnknownKindError,
    connector_for,
    content_hash,
    default_registry,
    doc_id,
    normalize,
    seal,
    validate_batch,
)


T0 = datetime(2026, 10, 8, 12, 0, tzinfo=UTC)
T1 = T0 + timedelta(minutes=5)


def collect(kind: Kind, item: object) -> list[EpisodeDraft]:
    """Run the default connector for `kind`."""
    return list(connector_for(default_registry(), kind).collect(item))


def one(kind: Kind, item: object) -> EpisodeDraft:
    """Run the default connector and expect a single draft."""
    drafts = collect(kind, item)
    assert len(drafts) == 1
    return drafts[0]


class TestConnectors:
    """What each built-in connector maps."""

    def test_chat_keeps_speaker_and_session_out_of_content(self):
        """The text is raw; speaker and session are structured fields."""
        draft = one(
            Kind.MESSAGE,
            ChatMessage("m1", "s9", "u7", Role.USER, "hello", T0, "Ana"),
        )

        assert draft.content == "hello"
        assert draft.actor == Actor("u7", Role.USER, "Ana")
        assert draft.source.ref == "m1"
        assert draft.source.description == "session s9"
        assert draft.metadata == {"session_id": "s9"}
        assert draft.occurred_at == T0

    def test_document_yields_one_episode_per_non_empty_part(self):
        """Blank parts leave gaps in chunk_index instead of renumbering."""
        drafts = collect(
            Kind.DOCUMENT, DocumentInput("file:///a.md", ["alpha", "  ", "gamma"], T0)
        )

        assert [d.source.ref for d in drafts] == [
            "file:///a.md#chunk-0",
            "file:///a.md#chunk-2",
        ]
        assert {d.occurred_at for d in drafts} == {T0}
        assert drafts[1].metadata == {
            "doc_id": doc_id("file:///a.md"),
            "chunk_index": 2,
        }
        assert drafts[0].actor.role is Role.SYSTEM

    def test_tool_output_actor_is_the_tool_and_args_are_metadata(self):
        """Arguments describe the call, not the output."""
        draft = one(
            Kind.TOOL_OUTPUT,
            ToolOutputInput("call-1", "search", {"hits": 3}, T0, {"q": "x"}),
        )

        assert draft.actor == Actor("search", Role.TOOL)
        assert draft.source.system == "agent_tools"
        assert draft.content == {"hits": 3}
        assert draft.metadata == {"args": {"q": "x"}}

    def test_event_uses_event_time_and_keeps_the_payload(self):
        """The raw payload, including its type, is the content."""
        payload = {"type": "invoice.paid", "amount": 5}

        draft = one(Kind.EVENT, EventInput("stripe", "evt_1", T0, payload))

        assert draft.occurred_at == T0
        assert draft.content == payload
        assert draft.actor == Actor("stripe", Role.SERVICE)
        assert (draft.source.system, draft.source.ref) == ("stripe", "evt_1")

    def test_record_keeps_the_row_as_a_dict(self):
        """The row is not flattened to text."""
        row = {"id": 4, "name": "Ana"}

        draft = one(Kind.RECORD, RecordInput("crm", "people", "4", row, T0))

        assert draft.content == row
        assert draft.source.ref == "4"
        assert draft.metadata == {"table": "people"}

    def test_unknown_kind_raises_without_fallback(self):
        """A bad kind and an unregistered kind both raise."""
        with pytest.raises(UnknownKindError):
            connector_for(default_registry(), "nope")
        with pytest.raises(UnknownKindError):
            connector_for({}, Kind.MESSAGE)

    def test_caller_can_override_a_connector(self):
        """An override replaces only its own kind."""

        class Custom:
            def collect(self, item: str) -> list[EpisodeDraft]:
                return []

        registry = default_registry({Kind.MESSAGE: Custom()})

        assert list(connector_for(registry, Kind.MESSAGE).collect("x")) == []
        assert connector_for(registry, Kind.EVENT) is not None


def event_episode_id(event: str, when: datetime, payload: dict[str, object]) -> str:
    """Seal an event and return its episode id."""
    draft = one(Kind.EVENT, EventInput("p", event, when, payload))
    return seal(draft, ingested_at=T0).id


class TestEpisodeIds:
    """What the episode id depends on."""

    def test_known_draft_has_a_fixed_id(self):
        """The id for a fixed draft is a literal."""
        draft = EpisodeDraft(
            Kind.MESSAGE,
            T0,
            Source("chat", "m1", "chat"),
            Actor("u", Role.USER),
            "hi",
        )

        assert seal(draft, ingested_at=T1).id == "c0e254c9e74b213cecda07e36231a9a7"

    def test_changing_ingested_at_or_description_keeps_the_id(self):
        """Neither is part of the id."""
        base = EpisodeDraft(
            Kind.MESSAGE,
            T0,
            Source("chat", "m1", "chat", "one"),
            Actor("u", Role.USER),
            "hi",
        )
        other = EpisodeDraft(
            Kind.MESSAGE,
            T0,
            Source("chat", "m1", "chat", "two"),
            Actor("u", Role.USER),
            "hi",
        )

        assert seal(base, ingested_at=T0).id == seal(other, ingested_at=T1).id

    def test_one_microsecond_changes_the_id(self):
        """The occurrence time counts to the microsecond."""
        assert event_episode_id("e1", T0, {"a": 1}) != event_episode_id(
            "e1", T0 + timedelta(microseconds=1), {"a": 1}
        )

    def test_key_order_and_zone_offset_do_not_change_the_id(self):
        """The same instant and payload give the same id."""
        plus_5_30 = timezone(timedelta(hours=5, minutes=30))

        assert event_episode_id("e1", T0, {"a": 1, "b": 2}) == event_episode_id(
            "e1", T0.astimezone(plus_5_30), {"b": 2, "a": 1}
        )

    def test_normalization(self):
        """Text and JSON are normalized before hashing."""
        assert normalize("  é\r\nx\r ") == "é\nx"
        assert normalize({"b": 1, "a": "é"}) == '{"a":"é","b":1}'
        assert content_hash("{}") != content_hash({})


class TestValidation:
    """Draft validation."""

    @pytest.mark.parametrize(
        ("mutation", "message"),
        [
            ({"occurred_at": datetime(2026, 1, 1)}, "timezone-aware"),
            ({"source": Source("s", "", "c")}, "source.ref"),
            ({"source": Source("s", "a\x1fb", "c")}, "x1f"),
            ({"content": "  "}, "content is empty"),
            ({"metadata": {"content_hash": "x"}}, "reserved"),
        ],
    )
    def test_invalid_drafts_are_rejected(self, mutation, message):
        """Each problem is named, with the draft index in a batch."""
        fields: dict[str, object] = {
            "kind": Kind.MESSAGE,
            "occurred_at": T0,
            "source": Source("chat", "m1", "chat"),
            "actor": Actor("u", Role.USER),
            "content": "hi",
        }
        draft = EpisodeDraft(**{**fields, **mutation})

        with pytest.raises(DraftValidationError, match=message):
            seal(draft, ingested_at=T0)
        with pytest.raises(DraftValidationError, match=r"draft\[0\]"):
            validate_batch([draft])
