"""Tests for the fact-to-observation adapter."""

import pytest

from ontogr_mem.data_models.extraction import Method
from ontogr_mem.data_models.facts import (
    ExtractedFact,
    FactKind,
    Modality,
    Polarity,
    Precision,
    TimeBasis,
    TypedFact,
)
from ontogr_mem.data_models.observation import NO_STAMP, Observation, ProvenanceStamp
from ontogr_mem.data_models.type_catalog import TypeView
from ontogr_mem.temporal.adapter import (
    EntityRefResolver,
    Skipped,
    SkipReason,
    observations_from_facts,
)
from ontogr_mem.temporal.cardinality import (
    Cardinality,
    DictRelationPolicy,
    RelationRules,
)
from ontogr_mem.temporal.errors import EpisodeMismatchError, UnknownRelationError
from ontogr_mem.temporal.policy import from_type_view
from tests.support.facts import fact_catalog, make_episode
from tests.unit.temporal.helpers import LocalRefResolver, at


EP = make_episode()
ONE = DictRelationPolicy({1: RelationRules(Cardinality.ONE)})
MANY = DictRelationPolicy()
INVERSE = DictRelationPolicy({1: RelationRules(Cardinality.ONE_PER_OBJECT)})


def extracted(**kw: object) -> ExtractedFact:
    """Build an extracted fact with a dense relation id."""
    base: dict[str, object] = {
        "fact_id": "f1",
        "text": "Ana lives in Paris",
        "entity_refs": ("ana", "paris"),
        "relation_type_id": 99,
        "relation_label": None,
        "fact_kind": FactKind.STATE,
        "polarity": Polarity.AFFIRMED,
        "modality": Modality.ACTUAL,
        "asserted_by": None,
        "valid_from": at(2020),
        "valid_to": None,
        "precision": Precision.YEAR,
        "time_expression": "in 2020",
        "time_basis": TimeBasis.STATED,
        "evidence_quote": "Ana lives in Paris",
        "evidence_span": None,
        "confidence": 1.0,
        "method": Method.LLM,
        "episode_id": "ep1",
    }
    base.update(kw)
    return ExtractedFact(**base)  # type: ignore[arg-type]


def typed(*, relation_type: int | None = 1, **kw: object) -> TypedFact:
    """Wrap an extracted fact with a stable relation type id."""
    return TypedFact(extracted(**kw), relation_type=relation_type, catalog_version="v1")


async def convert(
    *facts: TypedFact,
    resolver: EntityRefResolver | None = None,
    policy: DictRelationPolicy = ONE,
) -> tuple[list[Observation], list[Skipped]]:
    """Convert facts with the default resolver and policy."""
    return await observations_from_facts(
        facts,
        EP,
        resolver or LocalRefResolver(),
        policy,
    )


class TestAdapter:
    """Behavior of the fact-to-observation translation."""

    async def test_affirmed_state_becomes_an_assertion(self):
        """A dated state becomes an assertion with subject and object."""
        (observation,), skipped = await convert(typed())
        assert not skipped
        assert (
            observation.subject_id,
            observation.object_id,
            observation.kind.value,
        ) == ("ana", "paris", "assertion")
        assert observation.valid_from_precision is Precision.YEAR
        assert observation.valid_to is None

    async def test_stable_relation_id_replaces_the_dense_one(self):
        """The adapter reads ``TypedFact.relation_type``, not the dense id."""
        fact = typed()
        assert fact.fact.relation_type_id == 99
        (observation,), _ = await convert(fact)
        assert observation.relation_type_id == 1

    async def test_stated_valid_to_covers_its_whole_period(self):
        """A stated end is the first instant after its period."""
        (observation,), _ = await convert(typed(valid_to=at(2022)))
        assert observation.valid_to == at(2023)

    async def test_negated_state_is_a_stated_end_at_valid_from(self):
        """A negated state closes at the start it names."""
        (observation,), _ = await convert(typed(polarity=Polarity.NEGATED))
        assert observation.kind.value == "stated_end"
        assert observation.valid_from == at(2020)

    async def test_negated_without_a_date_is_skipped(self):
        """A stated end needs an instant."""
        observations, skipped = await convert(
            typed(
                polarity=Polarity.NEGATED,
                valid_from=None,
                time_basis=TimeBasis.NONE,
            )
        )
        assert observations == []
        assert skipped[0].reason is SkipReason.STATED_END_UNDATED

    async def test_affirmed_end_only_state_is_a_stated_end_after_its_period(self):
        """An affirmed "until 2022" ends the value after 2022."""
        (observation,), skipped = await convert(
            typed(
                valid_from=None,
                valid_to=at(2022),
                precision=Precision.YEAR,
                time_basis=TimeBasis.STATED,
            )
        )
        assert not skipped
        assert observation.kind.value == "stated_end"
        assert observation.valid_from == at(2023)
        assert observation.valid_to is None

    async def test_negated_end_only_state_is_not_turned_into_an_end(self):
        """A negated end-only claim says the state began later."""
        observations, skipped = await convert(
            typed(
                polarity=Polarity.NEGATED,
                valid_from=None,
                valid_to=at(2022),
                precision=Precision.YEAR,
            )
        )
        assert observations == []
        assert skipped[0].reason is SkipReason.END_ONLY

    async def test_routing_skips_what_cannot_be_reconciled(self):
        """Each unusable fact is returned with its reason."""
        cases = {
            SkipReason.NO_EVIDENCE: typed(evidence_quote=" "),
            SkipReason.NOT_STATE: typed(fact_kind=FactKind.EVENT),
            SkipReason.NOT_ACTUAL: typed(modality=Modality.PLANNED),
            SkipReason.NO_RELATION: typed(relation_type=None),
            SkipReason.REF_COUNT: typed(entity_refs=("ana",)),
        }
        for reason, fact in cases.items():
            observations, skipped = await convert(fact)
            assert observations == []
            assert skipped[0].reason is reason

    async def test_unknown_time_basis_gives_an_undated_assertion(self):
        """A fact with no basis is kept as an undated statement."""
        (observation,), _ = await convert(
            typed(valid_from=None, time_basis=TimeBasis.NONE, precision=None)
        )
        assert not observation.dated

    async def test_unresolved_ref_is_skipped(self):
        """A ref that resolves to nothing cannot become an observation."""

        class Nope:
            async def entity_id_for_ref(
                self, episode_id: str, local_id: str
            ) -> str | None:
                return None

        observations, skipped = await convert(typed(), resolver=Nope())
        assert observations == []
        assert skipped[0].reason is SkipReason.UNRESOLVED_REF

    async def test_reversed_interval_is_bad_interval(self):
        """An end before the start is rejected, not stored."""
        _, skipped = await convert(typed(valid_to=at(2019)))
        assert skipped[0].reason is SkipReason.BAD_INTERVAL

    async def test_slot_key_excludes_the_object_only_for_one_relations(self):
        """A one-valued slot ignores the object; a many-valued one keeps it."""
        (first,), _ = await convert(typed())
        (second,), _ = await convert(typed(entity_refs=("ana", "berlin")))
        assert first.slot_key == second.slot_key
        (first,), _ = await convert(typed(), policy=MANY)
        (second,), _ = await convert(typed(entity_refs=("ana", "berlin")), policy=MANY)
        assert first.slot_key != second.slot_key

    async def test_inverse_slot_is_keyed_on_the_object(self):
        """An inverse slot groups values by their holder, the object."""
        (first,), _ = await convert(typed(), policy=INVERSE)
        (second,), _ = await convert(
            typed(entity_refs=("bob", "paris")), policy=INVERSE
        )
        (third,), _ = await convert(
            typed(entity_refs=("bob", "berlin")), policy=INVERSE
        )
        assert first.slot_key == second.slot_key
        assert first.slot_key != third.slot_key
        (forward,), _ = await convert(typed())
        assert first.slot_key != forward.slot_key

    async def test_a_fact_from_another_episode_is_a_caller_bug(self):
        """A fact must name the episode being ingested."""
        with pytest.raises(EpisodeMismatchError, match="OTHER"):
            await convert(typed(episode_id="OTHER"))

    async def test_an_extracted_fact_is_rejected(self):
        """A dense relation id is only meaningful with its projection."""
        with pytest.raises(TypeError, match="TypedFact"):
            await observations_from_facts(
                [extracted()],  # type: ignore[list-item]
                EP,
                LocalRefResolver(),
                ONE,
            )

    async def test_stamp_reaches_the_observation_and_not_its_id(self):
        """Provenance is copied but excluded from the observation identity."""
        stamp = ProvenanceStamp("llm-extractor", "model-a", "p1")
        other = ProvenanceStamp("llm-extractor", "model-b", "p2")
        (first,), _ = await observations_from_facts(
            [typed()], EP, LocalRefResolver(), ONE, stamp
        )
        (second,), _ = await observations_from_facts(
            [typed()], EP, LocalRefResolver(), ONE, other
        )
        (plain,), _ = await convert(typed())
        assert first.provenance == stamp
        assert first.obs_id == second.obs_id == plain.obs_id
        assert plain.provenance == NO_STAMP


class TestFromTypeView:
    """Building a relation policy from a catalog view."""

    def test_exclusive_relations_become_one_and_others_many(self):
        """Only the ids passed as exclusive are single-valued."""
        policy = from_type_view(TypeView(fact_catalog()), (0,))
        assert policy.rules(0).cardinality is Cardinality.ONE
        assert policy.rules(1).cardinality is Cardinality.MANY

    def test_an_unknown_exclusive_id_is_rejected(self):
        """An id outside the view is a caller error."""
        with pytest.raises(UnknownRelationError):
            from_type_view(TypeView(fact_catalog()), (5,))
