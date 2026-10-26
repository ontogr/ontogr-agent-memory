"""Stale episode detection on a live graph tenant stack."""

from ontogr_mem.ontology.manager import OntologyManager
from ontogr_mem.ontology.models import AddEntity
from ontogr_mem.ontology.settings import OntologySettings
from ontogr_mem.ontology.validate import Context, EntityItem, ExtractedItems
from tests.support.clock import FixedClock
from tests.support.graph_backend import GraphBackend
from tests.support.llm import ScriptedLLM
from tests.support.ontology import base, cs, ingest_episode


async def test_stale_episodes_reports_an_ingested_episode_after_a_catalog_change(
    graph_backend: GraphBackend,
):
    """A catalog change that covers an unmatched label marks its episode stale."""
    stores = await graph_backend.open_stores()
    await ingest_episode(stores.episodes, "ep1")
    manager = OntologyManager(
        stores.ontology_store,
        ScriptedLLM(),
        OntologySettings(min_support=2),
        FixedClock(),
    )
    await manager.load_catalog(base())
    await manager.process_extraction(
        ExtractedItems(entities=(EntityItem("a", raw_type="Gadget"),)),
        Context("ep1"),
    )
    before = await manager.catalog()
    after = await manager.apply(cs(before, AddEntity(label="Gadget")))

    stale = await manager.stale_episodes(before.version, after.version)

    assert stale == {"ep1"}
