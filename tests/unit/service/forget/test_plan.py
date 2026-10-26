"""Forget planning resolves evidence and collateral without writing state."""

from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from ontogr_mem.data_models.fact_versions import FactStatus
from ontogr_mem.service.errors import ObjectNotFoundError
from ontogr_mem.service.forget.plan import build_plan
from ontogr_mem.service.forget.selector import ForgetSelector
from ontogr_mem.temporal.temporal_store import SlotState, SourceSelector


@dataclass(frozen=True)
class ObservationRow:
    """Observation fields consumed by the planner."""

    obs_id: str
    episode_id: str
    slot_key: str


class PlanningEpisodes:
    """Episode read port with a small metadata-backed fixture."""

    async def get_episodes(self, ids):
        """Return episode stubs for the requested ids that the fixture holds."""
        return [
            SimpleNamespace(id=value) for value in sorted(set(ids)) if value == "e2"
        ]

    async def episode_ids_by_subject(self, subject_ref):
        """Return the fixture episode ids that mention the subject."""
        return ["e2"] if subject_ref == "ana" else []


class PlanningTemporal:
    """Temporal read port with one updated slot."""

    def __init__(self):
        """Build a fixture slot with a closed prior fact, a target, and a neighbor."""
        self.observation = ObservationRow("o2", "e2", "slot")
        self.prior = SimpleNamespace(
            fact_id="f1",
            slot_key="slot",
            evidence=("o1",),
            current=False,
            status=FactStatus.CLOSED,
        )
        self.target = SimpleNamespace(
            fact_id="f2",
            slot_key="slot",
            evidence=("o2",),
            current=True,
            status=FactStatus.ACTIVE,
            subject_id="person",
            object_id="other",
        )
        self.neighbor = SimpleNamespace(
            fact_id="f3",
            slot_key="slot",
            evidence=("o3",),
            current=True,
            status=FactStatus.ACTIVE,
            subject_id="person",
            object_id="other",
        )

    async def facts_by_ids(self, ids):
        """Return the target fact when its id is requested."""
        return [self.target] if "f2" in ids else []

    async def facts_touching(self, ids):
        """Return no facts, since the fixture has no touching facts."""
        return []

    async def observations_by_ids(self, ids):
        """Return the target observation when its id is requested."""
        return [self.observation] if "o2" in ids else []

    async def observations_of_episode(self, episode_id):
        """Return the target observation for episode e2."""
        return [self.observation] if episode_id == "e2" else []

    async def load_slot(self, slot_key):
        """Return the fixture slot with its prior, target, and neighbor facts."""
        return SlotState(slot_key, facts=(self.prior, self.target, self.neighbor))

    async def supersessions_of_slot(self, slot_key):
        """Return one supersession that replaces f1 from episode e2."""
        return [SimpleNamespace(source_episode_id="e2", old_fact_id="f1")]

    async def episodes_matching(self, selector: SourceSelector):
        """Return episode e2 for mail selectors."""
        return ["e2"] if selector.system == "mail" else []


class PlanningResolver:
    """Resolver read port with one entity mentioned only in the target episode."""

    def __init__(self, merge_protected: set[str] | None = None):
        """Set entities named by merge history."""
        self.merge_protected = merge_protected or set()

    async def mentions_of_episode(self, episode_id):
        """Return the entity mention recorded in episode e2."""
        return [SimpleNamespace(entity_id="person")] if episode_id == "e2" else []

    async def mentions_of(self, entity_id):
        """Return episode e2 when asked about the fixture entity."""
        return [SimpleNamespace(episode_id="e2")] if entity_id == "person" else []

    async def has_merge_record(self, entity_id):
        """Return whether the entity is protected by a merge record."""
        return entity_id in self.merge_protected


class TestForgetPlan:
    """Resolved plans include only target evidence and current collateral."""

    async def test_update_plan_lists_collateral_and_reopened_facts(self):
        """Removing the update exposes its prior Fact and neighbor Fact."""
        plan = await build_plan(
            ForgetSelector(episode_ids=("e2", "unknown")),
            episodes=PlanningEpisodes(),
            temporal=PlanningTemporal(),
            resolver=PlanningResolver(),
        )

        assert plan.episode_ids == ("e2",)
        assert plan.observation_ids == ("o2",)
        assert plan.fact_ids == ("f2",)
        assert plan.entity_ids == ("person",)
        assert plan.collateral_fact_ids == ("f3",)
        assert plan.reopened_fact_ids == ("f1",)

    async def test_fact_selector_resolves_episodes_from_fact_evidence(self):
        """A Fact selector resolves through its stored evidence observations."""
        plan = await build_plan(
            ForgetSelector(fact_id="f2"),
            episodes=PlanningEpisodes(),
            temporal=PlanningTemporal(),
            resolver=PlanningResolver(),
        )

        assert plan.episode_ids == ("e2",)
        assert plan.fact_ids == ("f2",)

    async def test_unknown_fact_selector_is_not_found(self):
        """A Fact selector cannot resolve when no Fact version exists."""
        with pytest.raises(ObjectNotFoundError):
            await build_plan(
                ForgetSelector(fact_id="missing"),
                episodes=PlanningEpisodes(),
                temporal=PlanningTemporal(),
                resolver=PlanningResolver(),
            )

    async def test_subject_selector_uses_episode_metadata_read_port(self):
        """Subject selectors resolve through the episode store read port."""
        plan = await build_plan(
            ForgetSelector(subject_ref="ana"),
            episodes=PlanningEpisodes(),
            temporal=PlanningTemporal(),
            resolver=PlanningResolver(),
        )

        assert plan.episode_ids == ("e2",)

    async def test_source_selector_uses_temporal_source_read(self):
        """Source selectors resolve through the temporal source index."""
        plan = await build_plan(
            ForgetSelector(source_system="mail"),
            episodes=PlanningEpisodes(),
            temporal=PlanningTemporal(),
            resolver=PlanningResolver(),
        )

        assert plan.episode_ids == ("e2",)

    async def test_merge_history_protects_entity_from_orphan_candidates(self):
        """Entities named by merge history cannot appear as removable orphans."""
        plan = await build_plan(
            ForgetSelector(episode_ids=("e2",)),
            episodes=PlanningEpisodes(),
            temporal=PlanningTemporal(),
            resolver=PlanningResolver(merge_protected={"person"}),
        )

        assert plan.entity_ids == ()
