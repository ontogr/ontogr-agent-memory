"""Episode ontology version stamps use the extraction projection."""

import pytest

from ontogr_mem.ontology.declaration import EntityDeclaration
from ontogr_mem.ontology.default import default_ontology
from ontogr_mem.service.contracts import AddMemory, MemoryKind
from ontogr_mem.temporal.engine import TemporalEngine
from tests.integration.test_memory_service import _memory_models
from tests.support.graph_backend import GraphBackend
from tests.support.jobs import wait_for_job
from tests.support.memory_service import memory_service_harness


class TestEpisodeOntologyVersion:
    """Episode state records the ontology version used for extraction."""

    async def test_text_write_stamps_extraction_ontology_version(
        self, graph_backend: GraphBackend
    ) -> None:
        """Stamp the ontology version used by extraction."""
        harness = await memory_service_harness(graph_backend, models=_memory_models())
        receipt = await harness.service.add_memory(
            harness.principal,
            AddMemory(
                tenant="t1",
                kind=MemoryKind.TEXT,
                content="A violet comet note exists.",
                wait_seconds=5,
            ),
        )
        job = await wait_for_job(harness.service, harness.principal, receipt.job_id)
        assert job.result is not None
        tenant = await harness.service._jobs._deps.tenants.get_or_create("t1")
        episode_state = await tenant.stores.temporal_store.episode_state(
            job.result["episode_id"]
        )

        assert episode_state is not None
        assert (
            episode_state.ontology_version == (await tenant.ontology.catalog()).version
        )

    async def test_text_write_keeps_projection_version_after_ontology_change(
        self, graph_backend: GraphBackend, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Keep the extraction version when the ontology changes before ingest."""
        harness = await memory_service_harness(graph_backend, models=_memory_models())
        tenant = await harness.service._jobs._deps.tenants.get_or_create("t1")
        extraction_version = (await tenant.ontology.catalog()).version
        declaration = default_ontology()
        declaration.entities.append(EntityDeclaration(label="TestEntity"))
        original_ingest = TemporalEngine.ingest

        async def apply_change_then_ingest(engine, *args, **kwargs):
            await tenant.ontology.declare(declaration)
            return await original_ingest(engine, *args, **kwargs)

        monkeypatch.setattr(TemporalEngine, "ingest", apply_change_then_ingest)
        receipt = await harness.service.add_memory(
            harness.principal,
            AddMemory(
                tenant="t1",
                kind=MemoryKind.TEXT,
                content="A violet comet note exists.",
                wait_seconds=5,
            ),
        )
        job = await wait_for_job(harness.service, harness.principal, receipt.job_id)
        assert job.result is not None
        episode_state = await tenant.stores.temporal_store.episode_state(
            job.result["episode_id"]
        )

        assert episode_state is not None
        assert episode_state.ontology_version == extraction_version
        assert (await tenant.ontology.catalog()).version > extraction_version
