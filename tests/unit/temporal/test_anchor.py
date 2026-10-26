"""Anchor selection shared by temporal fingerprints and judge inputs."""

from ontogr_mem.temporal.anchor import anchor_observation_id
from tests.unit.temporal.helpers import at, obs


class TestAnchorObservation:
    """The stable observation that identifies evidence for a fact."""

    def test_earliest_dated_observation_wins_with_id_as_tie_breaker(self):
        """Dated evidence precedes undated evidence and ties use the id."""
        observations = [
            obs("b", "acme", at(2010)),
            obs("a", "acme", at(2010)),
            obs("z", "acme", None),
        ]
        by_id = {observation.obs_id: observation for observation in observations}

        assert (
            anchor_observation_id(
                [observation.obs_id for observation in observations], by_id
            )
            == "a"
        )

    def test_undated_observations_use_the_smallest_id(self):
        """When no evidence is dated, selection is stable by observation id."""
        observations = [obs("b", "acme", None), obs("a", "acme", None)]
        by_id = {observation.obs_id: observation for observation in observations}

        assert (
            anchor_observation_id(
                [observation.obs_id for observation in observations], by_id
            )
            == "a"
        )
