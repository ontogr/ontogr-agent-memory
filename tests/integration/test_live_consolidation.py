"""The model stages against a real model on Ollama Cloud. Skipped without a key.

The run prints what each stage wrote so the summaries can be read by eye. Run it
with ``OLLAMA_API_KEY`` set for one command.
"""

import pytest

from ontogr_mem.consolidation import (
    ConsolidationRunner,
    RunStatus,
    StageStatus,
    default_stages,
)
from ontogr_mem.consolidation.models import Community, Proposal
from ontogr_mem.consolidation.signals import record_session_entry
from tests.support.consolidation import new_world
from tests.support.ollama import OllamaLLMClient, available
from tests.unit.temporal.helpers import at, obs


@pytest.mark.skipif(not available(), reason="OLLAMA_API_KEY is not set")
async def test_the_model_stages_run_against_a_real_model(capsys) -> None:  # type: ignore[no-untyped-def]
    """Communities and reflection finish without a failed stage."""
    world = await new_world()
    for entity_id, name in [
        ("ana", "Ana"),
        ("acme", "Acme"),
        ("kreuzberg", "Kreuzberg"),
    ]:
        await world.add_entity(entity_id, name)
    await world.ensure_entities(["ana", "acme", "kreuzberg"])
    await world.temporal.ingest_observations(
        [
            obs("o1", "acme", at(2024), subject="ana", slot="ana-employer"),
            obs("o2", "kreuzberg", at(2024), subject="acme", slot="acme-office"),
        ]
    )
    await record_session_entry(
        world.store,
        "s1",
        "Ana asked for short answers twice and thanked the agent for them.",
        world.clock(),
        helpfulness=1,
        confidence=0.9,
    )
    runner = ConsolidationRunner(
        default_stages(),
        world.store,
        world.clock,
        world.settings,
        resolution=world.resolution,
        temporal=world.temporal,
        llm=OllamaLLMClient(),
    )

    result = await runner.run()

    communities = await world.store.list(Community)
    proposals = await world.store.list(Proposal)
    with capsys.disabled():
        print("\n" + "\n".join(f"{s.stage}: {s.status.value}" for s in result.stages))
        for community in communities:
            print(f"community {community.community_id}: {community.summary}")
        for proposal in proposals:
            print(f"proposal {proposal.status.value}: {proposal.text}")
    assert result.status is RunStatus.OK
    assert all(s.status is not StageStatus.FAILED for s in result.stages)
    assert communities
