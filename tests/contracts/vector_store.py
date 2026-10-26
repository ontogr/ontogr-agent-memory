"""Contract for vector store backends, checked through `VectorStore`."""

import asyncio

import pytest

from ontogr_mem.vectorstore import (
    CapabilityError,
    EdgeExtras,
    EmbedItem,
    FailReason,
    Hit,
    IndexDriftError,
    InvalidVectorError,
    NodeExtras,
    SearchFilters,
    SpaceMismatchError,
    SpaceSpec,
    StoreNotReadyError,
    TemporalFilter,
    Validity,
    ValidityUpdate,
    VectorKind,
)
from tests.support.embedder import MockEmbedder
from tests.support.vectors import (
    A,
    Factory,
    ProbeMismatchEmbedder,
    VectorHarness,
    fixed_embedder,
    ready,
)


def node_item(
    node_id: str,
    text: str,
    *,
    entity_type: int | None = None,
    status: str | None = None,
) -> EmbedItem:
    """Return a node item with the properties a node filter reads."""
    return EmbedItem(node_id, text, NodeExtras(entity_type, status))


def edge_item(
    fact_id: str,
    text: str,
    *,
    validity: Validity | None = None,
    subject: str | None = None,
    target: str | None = None,
    relation: int | None = None,
) -> EmbedItem:
    """Return an edge item whose extras match what the harness seeds."""
    validity = validity or Validity("active", 0, 0)
    subject = subject or f"s-{fact_id}"
    target = target or f"o-{fact_id}"
    return EmbedItem(
        fact_id,
        text,
        EdgeExtras(subject, target, validity, relation),
    )


async def seed_edges(
    harness: VectorHarness,
    facts: list[str],
    *,
    validity: Validity | None = None,
    subject: str | None = None,
    target: str | None = None,
    relation: int | None = None,
) -> None:
    """Seed the graph edges that will carry the vectors."""
    await harness.seed(
        VectorKind.EDGE,
        facts,
        validity=validity,
        subject_id=subject,
        object_id=target,
        relation_type_id=relation,
    )


class VectorStoreContract:
    """Checks every vector backend must pass.

    Subclasses provide the `make_harness` fixture, a factory that creates a
    harness on a fresh tenant for the embedder it is given.
    """

    @pytest.fixture
    def make_harness(self) -> Factory:
        """Return a factory for harnesses on fresh tenants."""
        raise NotImplementedError

    async def test_search_orders_by_score_and_maps_cosine_to_zero_one(
        self, make_harness
    ):
        """Cosine 1, 0.5, 0 and -1 score 1.0, 0.75, 0.5 and 0.0."""
        h = await ready(make_harness)
        ids = ["i-a", "i-half", "i-orth", "i-opp"]
        await h.seed(VectorKind.NODE, ids)
        await h.store.upsert(
            VectorKind.NODE,
            [
                node_item("i-a", "a"),
                node_item("i-half", "half"),
                node_item("i-orth", "orthogonal"),
                node_item("i-opp", "opposite"),
            ],
        )

        hits = await h.store.search(VectorKind.NODE, A, k=10)

        assert [x.id for x in hits] == ids
        assert [x.rank for x in hits] == [0, 1, 2, 3]
        assert [x.score for x in hits] == pytest.approx([1.0, 0.75, 0.5, 0.0])

    async def test_equal_scores_are_ordered_by_id(self, make_harness):
        """Ties never depend on insertion order."""
        h = await ready(make_harness)
        await h.seed(VectorKind.NODE, ["b", "a", "c"])
        await h.store.upsert(
            VectorKind.NODE,
            [node_item(i, "a") for i in ("b", "a", "c")],
        )

        hits = await h.store.search(VectorKind.NODE, A, k=10)

        assert [x.id for x in hits] == ["a", "b", "c"]

    async def test_min_score_is_inclusive_and_k_cuts_the_result(self, make_harness):
        """`min_score` keeps equal scores; `k` limits the count."""
        h = await ready(make_harness)
        await h.seed(VectorKind.NODE, ["i-a", "i-half", "i-orth"])
        await h.store.upsert(
            VectorKind.NODE,
            [
                node_item("i-a", "a"),
                node_item("i-half", "half"),
                node_item("i-orth", "orthogonal"),
            ],
        )

        kept = await h.store.search(VectorKind.NODE, A, k=10, min_score=0.75)
        top1 = await h.store.search(VectorKind.NODE, A, k=1)
        nothing = await h.store.search(VectorKind.NODE, A, k=10, min_score=0.9999)

        assert [x.id for x in kept] == ["i-a", "i-half"]
        assert [x.id for x in top1] == ["i-a"]
        assert [x.id for x in nothing] == ["i-a"]

    async def test_search_with_no_vectors_returns_an_empty_list(self, make_harness):
        """No hits is an empty list, not an error."""
        h = await ready(make_harness)

        assert await h.store.search(VectorKind.NODE, A, k=5) == []

    async def test_missing_node_properties_do_not_match_explicit_filters(
        self, make_harness
    ):
        """A missing property cannot satisfy a filter on that property."""
        h = await ready(make_harness)
        await h.seed(VectorKind.NODE, ["complete"], entity_type=7, status="active")
        await h.seed(VectorKind.NODE, ["no-status"], entity_type=7, status=None)
        await h.seed(VectorKind.NODE, ["no-type"], status="active")
        await h.store.upsert(
            VectorKind.NODE,
            [
                node_item("complete", "a", entity_type=7, status="active"),
                node_item("no-status", "a", entity_type=7),
                node_item("no-type", "a", status="active"),
            ],
        )

        by_status = await h.store.search(
            VectorKind.NODE,
            A,
            k=10,
            filters=SearchFilters(statuses=frozenset({"active"})),
        )
        by_type = await h.store.search(
            VectorKind.NODE,
            A,
            k=10,
            filters=SearchFilters(entity_types=frozenset({7})),
        )

        assert [hit.id for hit in by_status] == ["complete", "no-type"]
        assert [hit.id for hit in by_type] == ["complete", "no-status"]

    async def test_edge_history_and_current_filters_use_fact_visibility(
        self, make_harness
    ):
        """History includes old rows; current view requires active and unexpired."""
        h = await ready(make_harness)
        await seed_edges(h, ["active"], validity=Validity("active", 0, 0))
        await seed_edges(h, ["closed"], validity=Validity("closed", 0, 0))
        await seed_edges(
            h,
            ["expired"],
            validity=Validity("active", 0, 0, expired_at_us=1),
        )
        await h.store.upsert(
            VectorKind.EDGE,
            [
                edge_item("active", "a"),
                edge_item("closed", "a", validity=Validity("closed", 0, 0)),
                edge_item(
                    "expired", "a", validity=Validity("active", 0, 0, expired_at_us=1)
                ),
            ],
        )

        current = await h.store.search(VectorKind.EDGE, A, k=10)
        history = await h.store.search(
            VectorKind.EDGE,
            A,
            k=10,
            filters=SearchFilters(temporal=TemporalFilter(history=True)),
        )

        assert [hit.id for hit in current] == ["active"]
        assert {hit.id for hit in history} == {"active", "closed", "expired"}

    async def test_search_with_zero_k_returns_an_empty_list(self, make_harness):
        """Asking for nothing costs nothing."""
        h = await ready(make_harness)

        assert await h.store.search(VectorKind.NODE, A, k=0) == []

    async def test_same_text_is_not_embedded_again(self, make_harness):
        """An unchanged text, also with other whitespace, keeps its vector."""
        h = await ready(make_harness)
        await h.seed(VectorKind.NODE, ["n1"])
        first = await h.store.upsert(VectorKind.NODE, [node_item("n1", "Ada Lovelace")])

        again = await h.store.upsert(
            VectorKind.NODE, [node_item("n1", "  Ada\n Lovelace ")]
        )

        assert (first.embedded, first.written) == (1, 1)
        assert (again.embedded, again.unchanged, again.written) == (0, 1, 0)

    async def test_changed_text_replaces_the_vector(self, make_harness):
        """New text for the same id moves the vector."""
        h = await ready(make_harness)
        await h.seed(VectorKind.NODE, ["n1"])
        await h.store.upsert(VectorKind.NODE, [node_item("n1", "a")])

        result = await h.store.upsert(VectorKind.NODE, [node_item("n1", "opposite")])
        hits = await h.store.search(VectorKind.NODE, A, k=1)

        assert result.written == 1
        assert hits[0].score == pytest.approx(0.0)

    async def test_last_duplicate_in_a_batch_wins(self, make_harness):
        """When an id repeats, the last item is the one stored."""
        h = await ready(make_harness)
        await h.seed(VectorKind.NODE, ["n1"])

        result = await h.store.upsert(
            VectorKind.NODE,
            [node_item("n1", "opposite"), node_item("n1", "a")],
        )
        hits = await h.store.search(VectorKind.NODE, A, k=1)

        assert result.attempted == 1
        assert hits[0].score == pytest.approx(1.0)

    async def test_empty_text_is_reported_not_raised(self, make_harness):
        """An item with blank text fails by itself; the others are written."""
        h = await ready(make_harness)
        await h.seed(VectorKind.NODE, ["n1", "n2"])

        result = await h.store.upsert(
            VectorKind.NODE, [node_item("n1", "  "), node_item("n2", "a")]
        )

        assert result.written == 1
        assert [(f.id, f.reason) for f in result.failed] == [
            ("n1", FailReason.EMPTY_TEXT)
        ]

    async def test_wrong_length_vector_from_the_embedder_is_never_written(
        self, make_harness
    ):
        """A provider vector of the wrong length is refused before the backend."""
        embedder = MockEmbedder(vectors={"short": [1.0, 0.0]})
        h = await ready(make_harness, embedder)
        await h.seed(VectorKind.NODE, ["n1"])

        result = await h.store.upsert(VectorKind.NODE, [node_item("n1", "short")])

        assert result.written == 0
        assert result.failed[0].reason is FailReason.BAD_DIMENSION
        assert await h.store.search(VectorKind.NODE, A, k=5) == []

    async def test_needs_embedding_lists_missing_and_stale_items(self, make_harness):
        """Items with no vector or other text are returned; current ones are not."""
        h = await ready(make_harness)
        await h.seed(VectorKind.NODE, ["n1", "n2"])
        await h.store.upsert(VectorKind.NODE, [node_item("n1", "a")])
        items = [
            node_item("n1", "a"),
            node_item("n2", "half"),
            node_item("n1x", "orthogonal"),
        ]

        todo = await h.store.needs_embedding(VectorKind.NODE, items)

        assert [i.id for i in todo] == ["n2", "n1x"]

    async def test_delete_removes_vectors_and_ignores_missing_ids(self, make_harness):
        """Delete returns the number removed and is safe to repeat."""
        h = await ready(make_harness)
        await h.seed(VectorKind.NODE, ["n1", "n2"])
        await h.store.upsert(
            VectorKind.NODE, [node_item("n1", "a"), node_item("n2", "half")]
        )

        first = await h.store.delete(VectorKind.NODE, ["n1", "nope"])
        second = await h.store.delete(VectorKind.NODE, ["n1", "nope"])
        hits = await h.store.search(VectorKind.NODE, A, k=5)

        assert (first, second) == (1, 0)
        assert [x.id for x in hits] == ["n2"]

    async def test_kinds_keep_separate_vectors(self, make_harness):
        """Node, episode and edge vectors do not mix."""
        h = await ready(make_harness)
        await h.seed(VectorKind.NODE, ["x"])
        await h.seed(VectorKind.EPISODE, ["x", "y"])
        await seed_edges(h, ["f1"])
        await h.store.upsert(VectorKind.NODE, [node_item("x", "a")])
        await h.store.upsert(
            VectorKind.EPISODE,
            [EmbedItem("x", "opposite"), EmbedItem("y", "a")],
        )
        await h.store.upsert(VectorKind.EDGE, [edge_item("f1", "half")])

        nodes = await h.store.search(VectorKind.NODE, A, k=5)
        episodes = await h.store.search(VectorKind.EPISODE, A, k=5)
        edges = await h.store.search(VectorKind.EDGE, A, k=5)

        assert [(x.id, round(x.score, 3)) for x in nodes] == [("x", 1.0)]
        assert [(x.id, round(x.score, 3)) for x in episodes] == [
            ("y", 1.0),
            ("x", 0.0),
        ]
        assert [(x.id, round(x.score, 3)) for x in edges] == [("f1", 0.75)]

    async def test_query_of_the_wrong_length_or_zero_norm_raises(self, make_harness):
        """An unusable query vector is an error, never an empty result."""
        h = await ready(make_harness)

        with pytest.raises(SpaceMismatchError):
            await h.store.search(VectorKind.NODE, [1.0, 0.0], k=5)
        with pytest.raises(InvalidVectorError):
            await h.store.search(VectorKind.NODE, [0.0, 0.0, 0.0], k=5)

    async def test_store_refuses_io_until_ready(self, make_harness):
        """Reads and writes before `ensure_ready` raise."""
        h = await make_harness(fixed_embedder())

        with pytest.raises(StoreNotReadyError):
            await h.store.search(VectorKind.NODE, A, k=5)
        with pytest.raises(StoreNotReadyError):
            await h.store.upsert(VectorKind.NODE, [node_item("n1", "a")])

    async def test_another_model_on_the_same_tenant_is_refused(self, make_harness):
        """The stored space pins the model; a different one raises on startup."""
        h = await ready(make_harness)

        other = h.reopen(fixed_embedder(model_id="other-model"))

        with pytest.raises(SpaceMismatchError):
            await other.ensure_ready()

    async def test_template_version_drift_is_refused(self, make_harness):
        """A stored template version the code no longer renders raises."""
        h = await make_harness(fixed_embedder())
        await h.store.backend.write_space(
            SpaceSpec(VectorKind.NODE, "mock", 3, "node.name", 99)
        )

        with pytest.raises(SpaceMismatchError):
            await h.store.ensure_ready()

    async def test_probe_length_mismatch_is_refused(self, make_harness):
        """An embedder whose probe has another length never serves traffic."""
        h = await make_harness(ProbeMismatchEmbedder())

        with pytest.raises(SpaceMismatchError):
            await h.store.ensure_ready()

    async def test_same_model_reopens_cleanly(self, make_harness):
        """A second store with the same configuration starts."""
        h = await ready(make_harness)

        await h.reopen(fixed_embedder()).ensure_ready()

    async def test_tenants_do_not_share_vectors(self, make_harness):
        """The same id in two tenants has two vectors, and a tenant sees its own."""
        one = await ready(make_harness)
        two = await ready(make_harness, fixed_embedder(model_id="other-model"))
        await one.seed(VectorKind.NODE, [f"n{i}" for i in range(20)])
        await two.seed(VectorKind.NODE, ["n0"])
        await one.store.upsert(
            VectorKind.NODE, [node_item(f"n{i}", "a") for i in range(20)]
        )
        await two.store.upsert(VectorKind.NODE, [node_item("n0", "opposite")])

        seen_one = await one.store.search(VectorKind.NODE, A, k=10)
        seen_two = await two.store.search(VectorKind.NODE, A, k=10)

        assert len(seen_one) == 10
        assert [(x.id, round(x.score, 3)) for x in seen_two] == [("n0", 0.0)]

    async def test_concurrent_searches_never_repeat_an_id(self, make_harness):
        """Under load every result lists each id once."""
        h = await ready(make_harness)
        ids = [f"n{i}" for i in range(30)]
        await h.seed(VectorKind.NODE, ids)
        await h.store.upsert(VectorKind.NODE, [node_item(i, i) for i in ids])

        results = await asyncio.gather(
            *(h.store.search(VectorKind.NODE, A, k=10) for _ in range(100))
        )

        for hits in results:
            assert len({x.id for x in hits}) == len(hits) == 10

    async def test_score_by_ids_scores_only_stored_ids(self, make_harness):
        """Scores use the same scale as search, and missing ids are absent."""
        h = await ready(make_harness)
        await h.seed(VectorKind.NODE, ["n1", "n2"])
        await h.store.upsert(
            VectorKind.NODE, [node_item("n1", "a"), node_item("n2", "opposite")]
        )

        scores = await h.store.score_by_ids(VectorKind.NODE, A, ["n1", "n2", "missing"])

        assert scores["n1"] == pytest.approx(1.0)
        assert scores["n2"] == pytest.approx(0.0)
        assert "missing" not in scores

    async def test_iter_ids_returns_keyset_pages(self, make_harness):
        """Ids come back ascending, and `after` resumes the page."""
        h = await ready(make_harness)
        ids = ["c", "a", "d", "b"]
        await h.seed(VectorKind.NODE, ids)
        await h.store.upsert(VectorKind.NODE, [node_item(i, "a") for i in ids])

        first = await h.store.iter_ids(VectorKind.NODE, None, 2)
        second = await h.store.iter_ids(VectorKind.NODE, first[-1], 2)

        assert first == ["a", "b"]
        assert second == ["c", "d"]

    async def test_fetch_vectors_returns_unit_vectors(self, make_harness):
        """Stored vectors have unit length and match the embedded text."""
        h = await ready(make_harness)
        await h.seed(VectorKind.NODE, ["n1"])
        await h.store.upsert(VectorKind.NODE, [node_item("n1", "a")])

        vectors = await h.store.fetch_vectors(VectorKind.NODE, ["n1", "missing"])

        assert list(vectors) == ["n1"]
        assert vectors["n1"] == pytest.approx(A, abs=1e-6)

    async def test_exists_reports_stored_ids(self, make_harness):
        """Exists ignores ids without a vector."""
        h = await ready(make_harness)
        await h.seed(VectorKind.NODE, ["n1", "n2"])
        await h.store.upsert(VectorKind.NODE, [node_item("n1", "a")])

        assert await h.store.exists(VectorKind.NODE, ["n1", "n2"]) == {"n1"}

    async def test_hydrate_splits_present_and_missing(self, make_harness):
        """Hydrate returns the hits that still exist and the ids that do not."""
        h = await ready(make_harness)
        await h.seed(VectorKind.NODE, ["n1"])
        await h.store.upsert(VectorKind.NODE, [node_item("n1", "a")])
        hits = [Hit("n1", 1.0, 0), Hit("gone", 0.5, 1)]

        kept, missing = await h.store.hydrate(VectorKind.NODE, hits)

        assert kept == [hits[0]]
        assert missing == ["gone"]

    async def test_node_filters_select_entity_type_and_status(self, make_harness):
        """Node filters keep only matching entities."""
        h = await ready(make_harness)
        await h.seed(VectorKind.NODE, ["n1"], entity_type=1, status="active")
        await h.seed(VectorKind.NODE, ["n2"], entity_type=2, status="closed")
        await h.seed(VectorKind.NODE, ["n3"], entity_type=1, status="active")
        await h.store.upsert(
            VectorKind.NODE,
            [
                node_item("n1", "a", entity_type=1, status="active"),
                node_item("n2", "a", entity_type=2, status="closed"),
                node_item("n3", "a", entity_type=1, status="active"),
            ],
        )

        by_type = await h.store.search(
            VectorKind.NODE,
            A,
            k=10,
            filters=SearchFilters(entity_types=frozenset({1})),
        )
        by_status = await h.store.search(
            VectorKind.NODE,
            A,
            k=10,
            filters=SearchFilters(statuses=frozenset({"active"})),
        )
        both = await h.store.search(
            VectorKind.NODE,
            A,
            k=10,
            filters=SearchFilters(
                entity_types=frozenset({1}), statuses=frozenset({"active"})
            ),
        )

        assert [x.id for x in by_type] == ["n1", "n3"]
        assert [x.id for x in by_status] == ["n1", "n3"]
        assert [x.id for x in both] == ["n1", "n3"]

    async def test_filter_for_the_wrong_kind_is_refused(self, make_harness):
        """A filter that names a property the kind lacks is a caller error."""
        h = await ready(make_harness)

        with pytest.raises(ValueError):
            await h.store.search(
                VectorKind.EDGE,
                A,
                k=5,
                filters=SearchFilters(entity_types=frozenset({1})),
            )
        with pytest.raises(ValueError):
            await h.store.search(
                VectorKind.NODE,
                A,
                k=5,
                filters=SearchFilters(temporal=TemporalFilter()),
            )

    async def test_edge_item_requires_edge_extras(self, make_harness):
        """An edge item without endpoints and validity is refused."""
        h = await ready(make_harness)
        await seed_edges(h, ["f1"])

        with pytest.raises(ValueError):
            await h.store.upsert(VectorKind.EDGE, [EmbedItem("f1", "a")])

    async def test_edge_search_defaults_to_current_belief(self, make_harness):
        """Closed and expired facts are hidden unless history is asked for."""
        h = await ready(make_harness)
        current = Validity("active", 0, 0)
        closed = Validity("closed", 0, 0)
        expired = Validity("active", 0, 0, expired_at_us=10)
        await seed_edges(h, ["f-current"], validity=current)
        await seed_edges(h, ["f-closed"], validity=closed)
        await seed_edges(h, ["f-expired"], validity=expired)
        await h.store.upsert(
            VectorKind.EDGE,
            [
                edge_item("f-current", "a", validity=current),
                edge_item("f-closed", "a", validity=closed),
                edge_item("f-expired", "a", validity=expired),
            ],
        )

        visible = await h.store.search(VectorKind.EDGE, A, k=10)
        history = await h.store.search(
            VectorKind.EDGE,
            A,
            k=10,
            filters=SearchFilters(temporal=TemporalFilter(history=True)),
        )

        assert [x.id for x in visible] == ["f-current"]
        assert sorted(x.id for x in history) == [
            "f-closed",
            "f-current",
            "f-expired",
        ]

    async def test_edge_search_respects_valid_at(self, make_harness):
        """A fact is visible at an instant only inside its validity window."""
        h = await ready(make_harness)
        early = Validity("active", 0, 0, valid_to_us=100)
        late = Validity("active", 100, 0, valid_to_us=200)
        await seed_edges(h, ["f-early"], validity=early)
        await seed_edges(h, ["f-late"], validity=late)
        await h.store.upsert(
            VectorKind.EDGE,
            [
                edge_item("f-early", "a", validity=early),
                edge_item("f-late", "a", validity=late),
            ],
        )

        hits = await h.store.search(
            VectorKind.EDGE,
            A,
            k=10,
            filters=SearchFilters(temporal=TemporalFilter(valid_at_us=150)),
        )

        assert [x.id for x in hits] == ["f-late"]

    async def test_edge_endpoint_and_relation_filters(self, make_harness):
        """Endpoint and relation filters keep only matching facts."""
        h = await ready(make_harness)
        await seed_edges(h, ["f1"], subject="s1", target="o1", relation=7)
        await seed_edges(h, ["f2"], subject="s1", target="o2", relation=8)
        await seed_edges(h, ["f3"], subject="s2", target="o1", relation=7)
        await h.store.upsert(
            VectorKind.EDGE,
            [
                edge_item("f1", "a", subject="s1", target="o1", relation=7),
                edge_item("f2", "a", subject="s1", target="o2", relation=8),
                edge_item("f3", "a", subject="s2", target="o1", relation=7),
            ],
        )

        by_subject = await h.store.search(
            VectorKind.EDGE,
            A,
            k=10,
            filters=SearchFilters(subject_ids=frozenset({"s1"})),
        )
        by_object = await h.store.search(
            VectorKind.EDGE,
            A,
            k=10,
            filters=SearchFilters(object_ids=frozenset({"o1"})),
        )
        by_relation = await h.store.search(
            VectorKind.EDGE,
            A,
            k=10,
            filters=SearchFilters(relation_type_ids=frozenset({8})),
        )

        assert [x.id for x in by_subject] == ["f1", "f2"]
        assert [x.id for x in by_object] == ["f1", "f3"]
        assert [x.id for x in by_relation] == ["f2"]

    async def test_unchanged_edge_text_keeps_the_row(self, make_harness):
        """Re-upserting the same fact text embeds nothing and changes nothing."""
        h = await ready(make_harness)
        validity = Validity("active", 0, 0)
        await seed_edges(h, ["f1"], validity=validity)
        first = await h.store.upsert(
            VectorKind.EDGE, [edge_item("f1", "a", validity=validity)]
        )

        again = await h.store.upsert(
            VectorKind.EDGE, [edge_item("f1", " a ", validity=validity)]
        )

        assert (first.embedded, first.written) == (1, 1)
        assert (again.embedded, again.unchanged, again.written) == (0, 1, 0)

    async def test_known_at_search_raises_capability_error(self, make_harness):
        """Past belief needs the graph re-check, which the store does not do."""
        h = await ready(make_harness)

        with pytest.raises(CapabilityError):
            await h.store.search(
                VectorKind.EDGE,
                A,
                k=5,
                filters=SearchFilters(temporal=TemporalFilter(known_at_us=5)),
            )

    async def test_a_backend_refuses_a_past_belief_filter(self, make_harness):
        """A backend is asked for rows, never for the belief behind them."""
        h = await ready(make_harness)

        with pytest.raises(CapabilityError):
            await h.store.backend.search(
                VectorKind.EDGE,
                A,
                5,
                SearchFilters(temporal=TemporalFilter(known_at_us=5)),
                None,
            )

    async def test_update_validity_needs_a_mirror(self, make_harness):
        """A backend with a mirror records the change; one without refuses."""
        h = await ready(make_harness)
        validity = Validity("active", 0, 0)
        await seed_edges(h, ["f1"], validity=validity)
        await h.store.upsert(VectorKind.EDGE, [edge_item("f1", "a", validity=validity)])
        update = ValidityUpdate("f1", Validity("closed", 0, 0))

        if not h.store.capabilities.validity_mirror:
            with pytest.raises(CapabilityError):
                await h.store.update_validity([update])
            return

        assert await h.store.update_validity([update]) == 1
        visible = await h.store.search(VectorKind.EDGE, A, k=5)
        history = await h.store.search(
            VectorKind.EDGE,
            A,
            k=5,
            filters=SearchFilters(temporal=TemporalFilter(history=True)),
        )

        assert visible == []
        assert [x.id for x in history] == ["f1"]


class GraphVectorStoreContract(VectorStoreContract):
    """Extra checks for backends that keep vectors on graph objects."""

    async def test_write_for_a_missing_node_is_reported_not_raised(self, make_harness):
        """A vector for a node that does not exist fails by itself."""
        h = await ready(make_harness)
        await h.seed(VectorKind.NODE, ["present"])

        result = await h.store.upsert(
            VectorKind.NODE,
            [node_item("present", "a"), node_item("absent", "half")],
        )

        assert result.written == 1
        assert [(f.id, f.reason) for f in result.failed] == [
            ("absent", FailReason.MISSING_NODE)
        ]

    async def test_write_for_a_missing_edge_is_reported_not_raised(self, make_harness):
        """A vector for an edge that does not exist fails by itself."""
        h = await ready(make_harness)
        await seed_edges(h, ["f1"])

        result = await h.store.upsert(
            VectorKind.EDGE,
            [edge_item("f1", "a"), edge_item("f2", "half")],
        )

        assert result.written == 1
        assert [(f.id, f.reason) for f in result.failed] == [
            ("f2", FailReason.MISSING_NODE)
        ]

    async def test_search_on_a_missing_index_raises(self, make_harness):
        """A dropped index is an error, never an empty list."""
        h = await ready(make_harness)
        assert h.drop_index is not None
        await h.seed(VectorKind.NODE, ["n1"])
        await h.store.upsert(VectorKind.NODE, [node_item("n1", "a")])
        await h.drop_index(VectorKind.NODE)

        try:
            with pytest.raises(IndexDriftError):
                await h.store.search(VectorKind.NODE, A, k=5)
        finally:
            await h.store.ensure_ready()

    async def test_node_without_a_vector_is_not_a_hit(self, make_harness):
        """A seeded node that was never embedded is not returned."""
        h = await ready(make_harness)
        await h.seed(VectorKind.NODE, ["with", "without"])
        await h.store.upsert(VectorKind.NODE, [node_item("with", "a")])

        hits = await h.store.search(VectorKind.NODE, A, k=5)

        assert [x.id for x in hits] == ["with"]
