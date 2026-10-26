"""Stable anchor selection for a fact's observations."""

from collections.abc import Iterable, Mapping, Sequence

from ontogr_mem.data_models.observation import Observation


def anchor_observation(observations: Sequence[Observation]) -> Observation | None:
    """Return the observation that identifies a fact: its earliest dated one.

    Undated evidence is used only when nothing is dated. A tie on the date is
    broken by the observation id, so the anchor of a fact never moves.

    Args:
        observations: The observations that support the fact.

    Returns:
        The anchor, or ``None`` when there is no observation.
    """
    if not observations:
        return None
    dated = [
        observation
        for observation in observations
        if observation.valid_from is not None
    ]
    if dated:
        return min(
            dated, key=lambda observation: (observation.valid_from, observation.obs_id)
        )
    return min(observations, key=lambda observation: observation.obs_id)


def anchor_observation_id(
    obs_ids: Iterable[str], observations: Mapping[str, Observation]
) -> str:
    """Return the earliest dated known observation, with stable fallbacks."""
    ids = tuple(obs_ids)
    known = [observations[obs_id] for obs_id in ids if obs_id in observations]
    anchor = anchor_observation(known)
    if anchor is not None:
        return anchor.obs_id
    return min(ids, default="")
