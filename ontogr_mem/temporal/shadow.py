"""Shadow runs for the observations a directive excludes.

An excluded fact takes no part in the walk, but it still gets a version so the
history shows that it was corrected or overridden.
"""

from collections.abc import Sequence

from ontogr_mem.data_models.fact_versions import FactStatus
from ontogr_mem.data_models.observation import Observation, ObservationKind
from ontogr_mem.temporal.anchor import anchor_observation
from ontogr_mem.temporal.timeline import Run, SlotDirectives


def shadow_runs(
    observations: Sequence[Observation], directives: SlotDirectives
) -> tuple[tuple[Run, FactStatus], ...]:
    """Return one run per value whose dated assertions the directives exclude.

    Args:
        observations: Every observation of the slot.
        directives: The directives the contradiction handler supplied.

    Returns:
        One run per excluded value, with the status that value takes.
    """
    by_value: dict[str, list[Observation]] = {}
    for observation in observations:
        if (
            observation.obs_id in directives.excluded
            and observation.dated
            and observation.kind is not ObservationKind.STATED_END
        ):
            by_value.setdefault(observation.value_id, []).append(observation)
    out: list[tuple[Run, FactStatus]] = []
    for value, items in sorted(by_value.items()):
        first = anchor_observation(items)
        if first is None or first.valid_from is None:
            continue
        last = max(
            items, key=lambda observation: (observation.valid_from, observation.obs_id)
        )
        statuses = {directives.excluded[o.obs_id] for o in items}
        status = (
            FactStatus.CORRECTED
            if FactStatus.CORRECTED in statuses
            else FactStatus.OVERRIDDEN
        )
        out.append(
            (
                Run(
                    value_id=value,
                    first_obs_id=first.obs_id,
                    obs_ids=tuple(sorted(o.obs_id for o in items)),
                    valid_from=first.valid_from,
                    valid_from_precision=first.valid_from_precision,
                    valid_to=last.valid_to,
                    valid_to_precision=last.valid_to_precision,
                    close_kind=None,
                    closed_by=None,
                    last_confirmed_at=max(o.occurred_at for o in items),
                ),
                status,
            )
        )
    return tuple(out)
