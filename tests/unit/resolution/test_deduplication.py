"""Deduplication behavior on the memory twin."""

from ontogr_mem.data_models.entities import EntityStatus
from ontogr_mem.graphstore.memory.resolver_store import MemoryResolverStore
from ontogr_mem.resolution.config import Budget
from tests.contracts.resolver_store import entity
from tests.unit.resolution.helpers import OracleLLM
from tests.unit.resolution.resolver_helpers import make


async def _duplicate_store() -> MemoryResolverStore:
    store = MemoryResolverStore()
    await store.write_entity_batch(
        [
            entity("e1", name="Acme"),
            entity("e2", name="acme").model_copy(update={"strict_key": "acme"}),
        ],
        [],
    )
    return store


async def test_rule_same_pairs_are_merged_without_a_model():
    """Two active entities with one strict key merge on the rule alone."""
    store = await _duplicate_store()
    resolver = await make(store)

    report = await resolver.deduplicate()

    assert report.pairs_rule_same == 1
    assert report.pairs_merged == 1
    statuses = {e.entity_id: e.status for e in await store.get_entities(["e1", "e2"])}
    assert sorted(statuses.values()) == [EntityStatus.ACTIVE, EntityStatus.MERGED]


async def test_plan_only_reports_the_plan_without_writing():
    """A plan-only run leaves both entities active."""
    store = await _duplicate_store()
    resolver = await make(store)

    report = await resolver.deduplicate(plan_only=True)

    assert report.plan
    statuses = {e.entity_id: e.status for e in await store.get_entities(["e1", "e2"])}
    assert set(statuses.values()) == {EntityStatus.ACTIVE}


async def test_selected_entities_are_marked_checked():
    """A completed run records the revision it checked."""
    store = await _duplicate_store()
    resolver = await make(store)

    await resolver.deduplicate(budget=Budget(sweep=10))

    survivor = (await store.get_entities(["e1"]))[0]
    assert survivor.last_checked_version == survivor.revision


async def test_a_blocked_pair_is_judged_on_its_jaccard_and_not_on_a_cosine():
    """A pair found by blocking clears the Jaccard bar on one vote."""
    store = MemoryResolverStore()
    # The two names share a Jaccard of 0.94: above the Jaccard bar of 0.92 but
    # below the cosine bar of 0.96, so only the Jaccard floor can settle them.
    await store.write_entity_batch(
        [
            entity("e1", name="Beta Solutions Group"),
            entity("e2", name="Beta Solutions Groups"),
        ],
        [],
    )
    llm = OracleLLM(lambda a, b: True)
    resolver = await make(store, llm=llm)

    report = await resolver.deduplicate()

    assert len(llm.requests) == 1
    assert "jaccard=" in llm.requests[0].prompt
    assert "cosine=" not in llm.requests[0].prompt
    assert report.pairs_merged == 1
