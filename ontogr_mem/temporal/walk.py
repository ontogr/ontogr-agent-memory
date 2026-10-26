"""The derive walk: one slot's observations in, one timeline out.

The walk reads the slot's events in order and keeps one open run per value. An
assertion of a value that has an open run joins that run; otherwise it opens
one, and in a single-valued slot it cuts every other open run that is not tied
with it. A run whose last member stated an end is closed there before a later
event, so a later repeat opens a new run.

The walk is a function of the whole observation set, so the valid-time result
never depends on arrival order.
"""

from collections.abc import Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime

from ontogr_mem.data_models.fact_versions import SupersessionKind
from ontogr_mem.data_models.facts import Precision
from ontogr_mem.data_models.ids import stable_id
from ontogr_mem.data_models.observation import Observation, ObservationKind
from ontogr_mem.temporal.anchor import anchor_observation
from ontogr_mem.temporal.cardinality import Cardinality
from ontogr_mem.temporal.shadow import shadow_runs
from ontogr_mem.temporal.ties import tie_components
from ontogr_mem.temporal.timeline import (
    Close,
    Conflict,
    ConflictKind,
    Derived,
    Run,
    SlotDirectives,
    TimelineEvent,
    build_timeline,
)


_FAR_FUTURE = datetime(9999, 12, 31, tzinfo=UTC)


@dataclass(frozen=True, slots=True)
class _OpenRun:
    """One value's run while the walk reads the timeline."""

    value_id: str
    observations: tuple[Observation, ...]
    start: datetime
    start_precision: Precision | None
    undated: tuple[Observation, ...] = ()
    end: datetime | None = None
    end_precision: Precision | None = None
    close_kind: SupersessionKind | None = None
    closed_by: str | None = None


def _conflict(
    slot_key: str, kind: ConflictKind, members: Sequence[Observation], reason: str
) -> Conflict:
    """Build one conflict from the observations that take part in it."""
    ids = tuple(sorted({o.obs_id for o in members}))
    return Conflict(
        conflict_id=stable_id("conflict", kind.value, slot_key, *ids),
        slot_key=slot_key,
        kind=kind,
        obs_ids=ids,
        episode_ids=tuple(sorted({o.episode_id for o in members})),
        reason=reason,
    )


def _tied_with(
    component: int | None, run: _OpenRun, component_of: dict[str, int]
) -> bool:
    """True when the run holds an observation in the newcomer's tie component."""
    if component is None:
        return False
    return any(component_of.get(o.obs_id) == component for o in run.observations)


def _anchor_id(observations: Sequence[Observation]) -> str:
    """Return the id of the observation that identifies the run."""
    anchor = anchor_observation(observations)
    return anchor.obs_id if anchor is not None else ""


@dataclass(frozen=True, slots=True)
class _WalkState:
    """Runs in opening order and events that could not close a run."""

    runs: tuple[_OpenRun, ...] = ()
    unmatched: tuple[str, ...] = ()
    unmatched_closes: tuple[str, ...] = ()


def _close(
    run: _OpenRun,
    at: datetime,
    precision: Precision | None,
    kind: SupersessionKind,
    by: str,
) -> _OpenRun:
    """Return the run closed with the supplied cause."""
    return replace(run, end=at, end_precision=precision, close_kind=kind, closed_by=by)


def _advance(state: _WalkState, until: datetime) -> _WalkState:
    """Close runs whose latest assertion stated an end at or before until."""
    runs = []
    for run in state.runs:
        updated = run
        last = run.observations[-1]
        if run.end is None and last.valid_to is not None and last.valid_to <= until:
            updated = _close(
                run,
                last.valid_to,
                last.valid_to_precision,
                SupersessionKind.VALID_TO_STATED,
                last.obs_id,
            )
        runs.append(updated)
    return replace(state, runs=tuple(runs))


def _apply_close(state: _WalkState, event: Close) -> _WalkState:
    """Apply a judged close to its open anchor, or record it as unmatched."""
    home = next(
        (
            run
            for run in sorted(state.runs, key=lambda run: run.value_id)
            if run.end is None
            and any(o.obs_id == event.anchor_obs_id for o in run.observations)
        ),
        None,
    )
    if home is None or event.at < home.start:
        return replace(
            state, unmatched_closes=(*state.unmatched_closes, event.cause_id)
        )
    return replace(
        state,
        runs=tuple(
            _close(
                run,
                event.at,
                event.at_precision,
                SupersessionKind.VERDICT,
                event.cause_id,
            )
            if run is home
            else run
            for run in state.runs
        ),
    )


def _apply_end(state: _WalkState, event: Observation, at: datetime) -> _WalkState:
    """Close the stated value, or record the end as unmatched."""
    if not any(
        run.end is None and run.value_id == event.value_id for run in state.runs
    ):
        return replace(state, unmatched=(*state.unmatched, event.obs_id))
    return replace(
        state,
        runs=tuple(
            _close(
                run,
                at,
                event.valid_from_precision,
                SupersessionKind.STATED_END,
                event.obs_id,
            )
            if run.end is None and run.value_id == event.value_id
            else run
            for run in state.runs
        ),
    )


def _apply_assertion(
    state: _WalkState,
    event: Observation,
    at: datetime,
    single: bool,
    component_of: dict[str, int],
    *,
    directives: SlotDirectives,
) -> _WalkState:
    """Join an open run or open a value and replace unprotected runs."""
    if any(run.end is None and run.value_id == event.value_id for run in state.runs):
        return replace(
            state,
            runs=tuple(
                replace(run, observations=(*run.observations, event))
                if run.end is None and run.value_id == event.value_id
                else run
                for run in state.runs
            ),
        )
    runs = []
    for run in state.runs:
        updated = run
        spared = any(
            event.obs_id in group and any(o.obs_id in group for o in run.observations)
            for group in directives.no_cut
        )
        if (
            single
            and run.end is None
            and not spared
            and not _tied_with(component_of.get(event.obs_id), run, component_of)
        ):
            updated = _close(
                run,
                at,
                event.valid_from_precision,
                SupersessionKind.REPLACEMENT,
                event.obs_id,
            )
        runs.append(updated)
    runs.append(_OpenRun(event.value_id, (event,), at, event.valid_from_precision))
    return replace(state, runs=tuple(runs))


def _step(
    state: _WalkState,
    item: TimelineEvent,
    single: bool,
    component_of: dict[str, int],
    directives: SlotDirectives,
) -> _WalkState:
    """Advance stated ends before applying the next ordered event."""
    state = _advance(state, item.at)
    event = item.event
    if isinstance(event, Close):
        return _apply_close(state, event)
    if event.kind is ObservationKind.STATED_END:
        return _apply_end(state, event, item.at)
    return _apply_assertion(
        state, event, item.at, single, component_of, directives=directives
    )


def _attach_undated(
    slot_key: str,
    runs: tuple[_OpenRun, ...],
    undated: Sequence[Observation],
    single: bool,
) -> tuple[tuple[_OpenRun, ...], tuple[Conflict, ...], tuple[str, ...]]:
    """Attach undated evidence and return conflicts or orphans for the rest."""
    conflicts = []
    orphans = []
    for observation in undated:
        home = next(
            (
                run
                for run in runs
                if run.value_id == observation.value_id and run.end is None
            ),
            None,
        )
        if home is not None:
            runs = tuple(
                replace(run, undated=(*run.undated, observation))
                if run is home
                else run
                for run in runs
            )
            continue
        open_runs = [run for run in runs if run.end is None]
        if single and open_runs:
            conflicts.append(
                _conflict(
                    slot_key,
                    ConflictKind.NO_DATES,
                    [observation, *(run.observations[0] for run in open_runs)],
                    "undated assertion names a different value than the open fact",
                )
            )
        else:
            orphans.append(observation.obs_id)
    return runs, tuple(conflicts), tuple(orphans)


def _derived_run(run: _OpenRun) -> Run:
    """Return the public interval and its sorted evidence."""
    return Run(
        value_id=run.value_id,
        first_obs_id=_anchor_id(run.observations),
        obs_ids=tuple(sorted(o.obs_id for o in (*run.observations, *run.undated))),
        valid_from=run.start,
        valid_from_precision=run.start_precision,
        valid_to=run.end,
        valid_to_precision=run.end_precision,
        close_kind=run.close_kind,
        closed_by=run.closed_by,
        last_confirmed_at=max(o.occurred_at for o in (*run.observations, *run.undated)),
    )


def derive(
    slot_key: str,
    observations: Sequence[Observation],
    cardinality: Cardinality,
    directives: SlotDirectives | None = None,
) -> Derived:
    """Derive the slot's runs, conflicts and leftovers from all its observations.

    Args:
        slot_key: Slot the observations belong to.
        observations: Every observation of the slot, in any order.
        cardinality: Whether a new value replaces the active one.
        directives: Excluded observations, non-cutting groups and judged closes.
            ``None`` means no directives.

    Returns:
        The derived timeline with any ties, unmatched ends and orphans.
    """
    directives = directives or SlotDirectives()
    single = cardinality is not Cardinality.MANY
    kept = [o for o in observations if o.obs_id not in directives.excluded]
    dated = [o for o in kept if o.dated]
    undated = sorted((o for o in kept if not o.dated), key=lambda o: o.obs_id)

    timeline = build_timeline(dated, directives.closes)
    assertions = [
        item.event
        for item in timeline
        if isinstance(item.event, Observation)
        and item.event.kind is not ObservationKind.STATED_END
    ]
    component_of = tie_components(assertions) if single else {}

    state = _WalkState()
    for item in timeline:
        state = _step(state, item, single, component_of, directives)
    state = _advance(state, _FAR_FUTURE)

    conflicts: list[Conflict] = []
    for root in sorted(set(component_of.values())):
        members = [o for o in assertions if component_of.get(o.obs_id) == root]
        conflicts.append(
            _conflict(
                slot_key,
                ConflictKind.EQUAL_START,
                members,
                "starts fall in one period at the coarser precision",
            )
        )

    runs, undated_conflicts, orphans = _attach_undated(
        slot_key, state.runs, undated, single
    )
    conflicts.extend(undated_conflicts)
    return Derived(
        runs=tuple(_derived_run(run) for run in runs),
        conflicts=tuple(sorted(conflicts, key=lambda item: item.conflict_id)),
        unmatched_ends=tuple(sorted(state.unmatched)),
        orphans=tuple(orphans),
        unmatched_closes=tuple(sorted(state.unmatched_closes)),
        shadow=shadow_runs(observations, directives),
    )
