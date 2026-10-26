"""Behavior tests for the optional inline verification tier."""

from ontogr_mem.data_models.entities import DecisionMethod, Outcome, VerdictKind
from ontogr_mem.graphstore.memory.resolver_store import MemoryResolverStore
from ontogr_mem.resolution.decision import collect_matches
from ontogr_mem.resolution.grouping import build_groups
from ontogr_mem.resolution.scoring import Candidate, PairSignals
from ontogr_mem.resolution.verdicts import apply_verdicts
from ontogr_mem.resolution.views import DecisionView
from tests.unit.resolution.helpers import FailingLLM, OracleLLM, entity
from tests.unit.resolution.resolver_helpers import (
    count_active,
    ex,
    latest_mention,
    make,
)


async def test_flag_off_by_default_means_no_llm_call_in_resolve():
    """The inline tier stays off unless the settings turn it on."""
    llm = OracleLLM(lambda a, b: True)
    store = MemoryResolverStore()
    resolver = await make(store, llm=llm)

    await resolver.resolve([ex("1", "John Smith", "Person")])
    await resolver.resolve([ex("1", "John Smith", "Person", episode_id="e2")])

    mention = await latest_mention(store, "e2")
    assert llm.requests == []
    assert mention.decision_outcome is Outcome.CREATED_PROVISIONAL


async def test_inline_yes_links_the_mention_to_the_existing_person():
    """A YES from the verifier links the second mention to the first entity."""
    llm = OracleLLM(lambda a, b: True)
    store = MemoryResolverStore()
    resolver = await make(store, llm=llm, inline=True)

    await resolver.resolve([ex("1", "John Smith", "Person")])
    await resolver.resolve([ex("1", "John Smith", "Person", episode_id="e2")])

    first = await latest_mention(store, "ep1")
    second = await latest_mention(store, "e2")
    assert second.entity_id == first.entity_id
    assert await count_active(store) == 1
    assert second.decision_method is DecisionMethod.LLM
    assert second.decision_reason == "llm_same"


async def test_inline_no_creates_a_resolved_entity_with_no_candidates_left():
    """A NO settles the pair, so the new entity is not provisional."""
    llm = OracleLLM(lambda a, b: False)
    store = MemoryResolverStore()
    resolver = await make(store, llm=llm, inline=True)

    await resolver.resolve([ex("1", "John Smith", "Person")])
    await resolver.resolve([ex("1", "John Smith", "Person", episode_id="e2")])

    mention = await latest_mention(store, "e2")
    assert mention.decision_outcome is Outcome.CREATED
    assert mention.decision_reason == "verified_distinct"
    assert await count_active(store) == 2


async def test_inline_llm_outage_never_fails_the_job_it_falls_back_to_provisional():
    """A failed verification leaves the pair pending, not merged."""
    store = MemoryResolverStore()
    resolver = await make(store, llm=FailingLLM(), inline=True)

    await resolver.resolve([ex("1", "John Smith", "Person")])
    await resolver.resolve([ex("1", "John Smith", "Person", episode_id="e2")])

    mention = await latest_mention(store, "e2")
    assert mention.decision_outcome is Outcome.CREATED_PROVISIONAL
    assert await count_active(store) == 2


async def test_inline_cross_type_yes_needs_two_votes():
    """A company and an organization need a swapped second YES."""
    llm = OracleLLM(lambda a, b: True)
    store = MemoryResolverStore()
    resolver = await make(store, llm=llm, inline=True)

    await resolver.resolve([ex("1", "Acme", "Company")])
    await resolver.resolve([ex("1", "Acme", "Organization", episode_id="e2")])

    mention = await latest_mention(store, "e2")
    assert mention.decision_method is DecisionMethod.LLM
    assert len(llm.requests) == 2


async def test_inline_one_call_per_resolve_for_many_ambiguous_groups():
    """Two ambiguous groups share one batched verification request."""
    llm = OracleLLM(lambda a, b: False)
    store = MemoryResolverStore()
    resolver = await make(store, llm=llm, inline=True)

    await resolver.resolve(
        [ex("1", "John Smith", "Person"), ex("2", "Mary Jones", "Person")]
    )
    await resolver.resolve(
        [
            ex("1", "John Smith", "Person", episode_id="e2"),
            ex("2", "Mary Jones", "Person", episode_id="e2"),
        ]
    )

    assert len(llm.requests) == 1


async def test_a_candidate_the_verifier_never_answered_stays_pending():
    """A pair whose call failed keeps the entity on the pending list."""
    store = MemoryResolverStore()
    resolver = await make(store)
    [group] = build_groups(
        [ex("1", "John Smith", "Person")], resolver.policy, resolver.relations
    )
    candidates = [
        Candidate(entity_id="e1", signals=PairSignals(jaccard=0.5)),
        Candidate(entity_id="e2", signals=PairSignals(jaccard=0.5)),
    ]
    view = DecisionView(
        candidates={
            "e1": entity("e1", name="John Smith", type_label="Person"),
            "e2": entity("e2", name="John Smith", type_label="Person"),
        }
    )

    collected = collect_matches(
        group,
        candidates,
        view,
        policy=resolver.policy,
        settings=resolver.settings,
        relations=resolver.relations,
    )
    plan = apply_verdicts(collected, {"e1": VerdictKind.DISTINCT}, False)

    assert plan.outcome is Outcome.CREATED_PROVISIONAL
    assert plan.candidates == ("e2",)
    assert plan.verify == ("e2",)
