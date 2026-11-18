"""Scope resolution and retrieval gate behavior."""

from datetime import UTC, datetime
from types import SimpleNamespace

import pytest

from ontogr_mem.data_models.entities import Entity
from ontogr_mem.data_models.facts import Modality
from ontogr_mem.retrieval.errors import UnknownFilterLabelError
from ontogr_mem.retrieval.gate import FactGate
from ontogr_mem.retrieval.models import RetrievalRequest
from ontogr_mem.retrieval.scope import SearchScope, build_scope
from ontogr_mem.service.contracts import SearchMemory


NOW = datetime(2026, 10, 23, 12, 0, tzinfo=UTC)
"""The present time every request in this module uses."""


class TestBuildScope:
    """Resolve request filters into a search scope."""

    async def test_resolves_filters_once_and_converts_modalities(self) -> None:
        """Resolve type labels once and convert modality names."""
        links = SimpleNamespace(
            entity_family=lambda label: (
                frozenset({1}) if label.lower() == "person" else None
            ),
            relation_family=lambda label: (
                frozenset({2}) if label.lower() == "works" else None
            ),
        )
        calls = 0

        async def load_links():
            nonlocal calls
            calls += 1
            return links

        request = SimpleNamespace(
            entity_types=("Person",),
            exclude_entity_types=(),
            relation_types=("works",),
            exclude_relation_types=(),
            modalities=("planned", "actual"),
        )
        scope = await build_scope("tenant", request, load_links)
        assert scope.entity_types == frozenset({1})
        assert scope.relation_types == frozenset({2})
        assert scope.modalities == frozenset({Modality.PLANNED, Modality.ACTUAL})
        assert calls == 1

    async def test_does_not_load_links_without_type_filters(self) -> None:
        """Skip ontology reads when the request has no type filters."""

        async def fail():
            raise AssertionError("unexpected ontology read")

        request = SimpleNamespace(
            entity_types=(),
            exclude_entity_types=(),
            relation_types=(),
            exclude_relation_types=(),
            modalities=("actual",),
        )
        assert await build_scope("tenant", request, fail) == SearchScope()

    async def test_resolves_a_public_search_request(self) -> None:
        """Resolve type labels and modalities from a SearchMemory request."""
        links = SimpleNamespace(
            entity_family=lambda label: frozenset({7}) if label == "Person" else None,
            relation_family=lambda label: None,
        )

        async def load_links():
            return links

        request = SearchMemory(
            query="query", entity_types=["Person"], modalities=["planned"]
        )
        scope = await build_scope("tenant", request, load_links)

        assert scope.entity_types == frozenset({7})
        assert scope.modalities == frozenset({Modality.PLANNED})

    async def test_unknown_label_names_tenant_and_label(self) -> None:
        """Report an unknown filter label with its tenant context."""
        links = SimpleNamespace(
            entity_family=lambda label: None, relation_family=lambda label: None
        )

        async def load_links():
            return links

        request = SimpleNamespace(
            entity_types=("Missing",),
            exclude_entity_types=(),
            relation_types=(),
            exclude_relation_types=(),
            modalities=("actual",),
        )
        with pytest.raises(UnknownFilterLabelError, match="Missing.*tenant"):
            await build_scope("tenant", request, load_links)


class TestRetrievalRequestScope:
    """Use the resolved scope as the only source of retrieval filters."""

    def test_uses_default_scope_modalities(self) -> None:
        """Use actual-only modalities when no scope is supplied."""
        request = RetrievalRequest(query="query", now=NOW)

        assert request.to_temporal_query().modalities == frozenset({Modality.ACTUAL})

    def test_uses_scope_modalities(self) -> None:
        """Use modalities from the resolved scope."""
        request = RetrievalRequest(
            query="query",
            now=NOW,
            scope=SearchScope(modalities=frozenset({Modality.PLANNED})),
        )

        assert request.to_temporal_query().modalities == frozenset({Modality.PLANNED})

    @pytest.mark.parametrize(
        "label_field",
        [
            "entity_types",
            "exclude_entity_types",
            "relation_types",
            "exclude_relation_types",
        ],
    )
    def test_rejects_unresolved_label_fields(self, label_field: str) -> None:
        """Raw labels cannot reach a request without first being resolved."""
        with pytest.raises(TypeError, match=label_field):
            RetrievalRequest(query="query", now=NOW, **{label_field: ("Person",)})


class TestFactGate:
    """Filter fact views and entity cards by resolved types."""

    async def test_filters_facts_using_merged_entity_results(self) -> None:
        """Use the surviving entity type when filtering merged entities."""
        entities = [
            SimpleNamespace(
                entity_id="old", ontology_type_id=9, merged_into="survivor"
            ),
            SimpleNamespace(entity_id="survivor", ontology_type_id=7, merged_into=None),
            SimpleNamespace(entity_id="other", ontology_type_id=8, merged_into=None),
        ]

        class Reader:
            async def get_entities(
                self, ids: list[str], *, follow_merged: bool = False
            ) -> list[object]:
                assert not follow_merged
                return [entity for entity in entities if entity.entity_id in ids]

        views = [
            SimpleNamespace(subject_id="old", object_id="other", relation_type_id=3),
            SimpleNamespace(subject_id="other", object_id="other", relation_type_id=3),
        ]
        gate = FactGate(SearchScope(entity_types=frozenset({7})), Reader())
        assert await gate.admit(views) == views[:1]

    async def test_filters_cards_by_own_type(self) -> None:
        """Filter entity cards by each entity's own type."""

        class Reader:
            async def get_entities(
                self, ids: list[str], *, follow_merged: bool
            ) -> list[object]:
                return [
                    Entity(
                        entity_id="a",
                        type_label="",
                        ontology_type_id=1,
                        canonical_name="a",
                        strict_key="a",
                        loose_key="a",
                        created_at=NOW,
                        updated_at=NOW,
                    ),
                    Entity(
                        entity_id="b",
                        type_label="",
                        ontology_type_id=2,
                        canonical_name="b",
                        strict_key="b",
                        loose_key="b",
                        created_at=NOW,
                        updated_at=NOW,
                    ),
                ]

        gate = FactGate(SearchScope(entity_types=frozenset({1})), Reader())
        assert await gate.admit_entities(["a", "b"]) == ["a"]

    async def test_stops_when_a_merge_target_cannot_be_fetched(self) -> None:
        """Stop following merges when a stored target has no entity row."""
        stale = SimpleNamespace(
            entity_id="stale", ontology_type_id=1, merged_into="gone"
        )
        calls = 0

        class Reader:
            async def get_entities(
                self, ids: list[str], *, follow_merged: bool
            ) -> list[object]:
                nonlocal calls
                calls += 1
                if calls > 4:
                    raise AssertionError("merge traversal did not terminate")
                return [stale] if "stale" in ids else []

        gate = FactGate(SearchScope(entity_types=frozenset({1})), Reader())

        assert await gate.admit_entities(["stale"]) == []
        assert calls == 2
