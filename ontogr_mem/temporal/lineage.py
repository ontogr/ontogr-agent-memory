"""Lineage: why a fact is believed, what an episode changed, how belief moved.

Every function here reads through the temporal store port and writes nothing, so
each store gives the same answers.

Roles of an observation for a fact: ``CREATOR`` is the earliest evidence,
``CONFIRMER`` is any later evidence for the same value, and ``CLOSER`` is the
observation that closed the fact (named by ``closed_by``).
"""

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum

from ontogr_mem.data_models.fact_versions import (
    Fact,
    FactStatus,
    Supersession,
    SupersessionKind,
)
from ontogr_mem.data_models.observation import Observation
from ontogr_mem.temporal.query import FactView, TemporalQuery, matches
from ontogr_mem.temporal.reader import FactReader, ObservationText
from ontogr_mem.temporal.temporal_store import TemporalStore


class Role(StrEnum):
    """What one observation did for one fact."""

    CREATOR = "creator"
    CONFIRMER = "confirmer"
    CLOSER = "closer"


@dataclass(frozen=True, slots=True)
class EvidenceRow:
    """One observation behind a fact, with who produced it.

    Attributes:
        obs_id: Observation id.
        role: What the observation did for the fact.
        episode_id: Episode the observation came from.
        occurred_at: When the episode says the statement happened.
        quote: Episode text that supports the statement.
        statement: The claim in plain words.
        source_system: Source system of the episode.
        connector: Connector that brought the statement in.
        actor_role: Role of the person or system behind the statement.
        extractor: Extractor that produced the statement.
        model: Model that read the text.
        prompt_version: Prompt version the model received.
    """

    obs_id: str
    role: Role
    episode_id: str
    occurred_at: datetime
    quote: str
    statement: str
    source_system: str
    connector: str
    actor_role: str
    extractor: str
    model: str
    prompt_version: str


@dataclass(frozen=True, slots=True)
class Why:
    """Everything that supports one current fact, oldest evidence first.

    Attributes:
        fact_id: Fact the evidence supports.
        version: Version of the fact that was read.
        status: Lifecycle state of that version.
        closed_by: Observation that closed the fact, or ``None``.
        evidence: Evidence rows, creator first.
    """

    fact_id: str
    version: int
    status: FactStatus
    closed_by: str | None
    evidence: tuple[EvidenceRow, ...]


@dataclass(frozen=True, slots=True)
class Effect:
    """What one episode did to one fact.

    Attributes:
        fact_id: Fact the episode touched.
        slot_key: Slot of the fact.
        role: Whether the episode created, confirmed or closed the fact.
        status: Lifecycle state of the current version.
    """

    fact_id: str
    slot_key: str
    role: Role
    status: FactStatus


@dataclass(frozen=True, slots=True)
class ChainLink:
    """One version of a fact and the audit row that closed it.

    Attributes:
        view: The fact version.
        closed_by: Audit row that ended the version, or ``None`` if it is open.
    """

    view: FactView
    closed_by: Supersession | None


@dataclass(frozen=True, slots=True)
class WorldEnd:
    """How a fact version ended in valid time.

    Attributes:
        kind: Reason the valid-time interval closed.
        cause_id: Observation or verdict that caused the close.
        episode_id: Episode of the closing observation, if available.
    """

    kind: SupersessionKind
    cause_id: str
    episode_id: str | None


@dataclass(frozen=True, slots=True)
class HistoryVersion:
    """One stored version with separate world-time and system-time ends.

    Attributes:
        fact_id: Stable id shared by each version of the fact.
        version: Stored version number.
        status: Lifecycle state of this version.
        statement: Annotated claim text.
        source: Annotated source system.
        valid_from: Start of the world-time interval.
        valid_to: End of the world-time interval, or ``None``.
        recorded_at: When this version entered the record.
        superseded_at: When a later version replaced this one, or ``None``.
        superseded_by_version: Immediate successor version, or ``None``.
        world_end: Cause and episode of a valid-time close, if present.
        evidence: Observation ids supporting this version.
        episode_ids: Episodes supporting or closing this version.
    """

    fact_id: str
    version: int
    status: FactStatus
    statement: str
    source: str
    valid_from: datetime
    valid_to: datetime | None
    recorded_at: datetime
    superseded_at: datetime | None
    superseded_by_version: int | None
    world_end: WorldEnd | None
    evidence: tuple[str, ...]
    episode_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class FactHistory:
    """The visible versions of one fact slot, in valid and record order.

    Attributes:
        fact_id: Fact id requested by the caller.
        slot_key: Slot containing every version of this fact.
        versions: Matching versions in ``(valid_from, recorded_at, version)``
            order.
    """

    fact_id: str
    slot_key: str
    versions: tuple[HistoryVersion, ...]


def _row(observation: Observation, role: Role) -> EvidenceRow:
    """Copy the provenance fields of an observation into an evidence row."""
    return EvidenceRow(
        obs_id=observation.obs_id,
        role=role,
        episode_id=observation.episode_id,
        occurred_at=observation.occurred_at,
        quote=observation.quote,
        statement=observation.statement,
        source_system=observation.source_system,
        connector=observation.connector,
        actor_role=observation.actor_role,
        extractor=observation.provenance.extractor,
        model=observation.provenance.model,
        prompt_version=observation.provenance.prompt_version,
    )


def _ordered(observations: list[Observation]) -> list[Observation]:
    """Return observations oldest first, with the id as a stable tie-break."""
    return sorted(observations, key=lambda o: (o.occurred_at, o.obs_id))


def _fact_observation_ids(fact: Fact) -> tuple[str, ...]:
    """Return evidence ids and the optional observation that closed a fact."""
    ids = (*fact.evidence, fact.closed_by) if fact.closed_by else fact.evidence
    return tuple(dict.fromkeys(ids))


def _world_end(fact: Fact, texts: dict[str, ObservationText]) -> WorldEnd | None:
    """Build the valid-time end and resolve its observation episode, if any."""
    if fact.close_kind is None or fact.closed_by is None:
        return None
    text = texts.get(fact.closed_by)
    episode_id = text.episode_id if text is not None else None
    return WorldEnd(fact.close_kind, fact.closed_by, episode_id)


async def fact_history(
    store: TemporalStore,
    reader: FactReader,
    fact_id: str,
    *,
    now: datetime,
    valid_at: datetime | None = None,
    known_at: datetime | None = None,
) -> FactHistory | None:
    """Return a fact's visible slot history at the requested times.

    Args:
        store: Tenant-bound temporal store.
        reader: Reader bound to the same store for annotations and evidence.
        fact_id: Stored fact whose slot history is requested.
        now: Present time used by the temporal query.
        valid_at: Optional world-time instant.
        known_at: Optional system-time instant.

    Returns:
        Matching versions, or ``None`` when the fact id is not stored.
    """
    matches_by_id = await store.facts_by_ids([fact_id])
    if not matches_by_id:
        return None

    slot_key = matches_by_id[0].slot_key
    facts = (await store.load_slot(slot_key)).facts
    query = TemporalQuery(
        now=now,
        valid_at=valid_at,
        known_at=known_at,
        include_history=True,
    )
    selected = [fact for fact in facts if matches(query, fact)]
    views = await reader.annotate(selected)
    views_by_key = {(view.fact_id, view.version): view for view in views}
    observation_ids = sorted(
        {obs_id for fact in selected for obs_id in _fact_observation_ids(fact)}
    )
    texts = await reader.observation_texts(observation_ids)
    versions = []
    version_numbers = {fact.version for fact in facts if fact.fact_id == fact_id}
    for fact in selected:
        view = views_by_key[(fact.fact_id, fact.version)]
        superseded_by = (
            fact.version + 1
            if fact.expired_at is not None and fact.version + 1 in version_numbers
            else None
        )
        episode_ids = tuple(
            dict.fromkeys(
                texts[obs_id].episode_id
                for obs_id in _fact_observation_ids(fact)
                if obs_id in texts
            )
        )
        versions.append(
            HistoryVersion(
                fact_id=fact.fact_id,
                version=fact.version,
                status=fact.status,
                statement=view.statement,
                source=view.source,
                valid_from=fact.valid_from,
                valid_to=fact.valid_to,
                recorded_at=fact.recorded_at,
                superseded_at=fact.expired_at,
                superseded_by_version=superseded_by,
                world_end=_world_end(fact, texts),
                evidence=fact.evidence,
                episode_ids=episode_ids,
            )
        )
    versions.sort(key=lambda item: (item.valid_from, item.recorded_at, item.version))
    return FactHistory(fact_id, slot_key, tuple(versions))


def _role(fact: Fact, obs_id: str, ordered_evidence: list[Observation]) -> Role:
    """Return the role one observation plays for one fact version."""
    if fact.closed_by == obs_id:
        return Role.CLOSER
    if ordered_evidence and ordered_evidence[0].obs_id == obs_id:
        return Role.CREATOR
    return Role.CONFIRMER


async def why(store: TemporalStore, fact_id: str) -> Why | None:
    """Return the evidence behind the current version of a fact.

    Returns ``None`` when the fact has no live current version.
    """
    current = [f for f in await store.current_facts() if f.fact_id == fact_id]
    if not current:
        return None
    fact = current[0]
    by_id = {o.obs_id: o for o in await store.observations_of_slot(fact.slot_key)}
    wanted = [i for i in fact.evidence if i in by_id]
    if fact.closed_by is not None and fact.closed_by in by_id:
        wanted.append(fact.closed_by)
    ordered = _ordered([by_id[i] for i in dict.fromkeys(wanted)])
    evidence = tuple(_row(o, _role(fact, o.obs_id, ordered)) for o in ordered)
    return Why(fact.fact_id, fact.version, fact.status, fact.closed_by, evidence)


async def episode_effects(store: TemporalStore, episode_id: str) -> tuple[Effect, ...]:
    """Return the facts an episode created, confirmed or closed.

    Only the slots where the episode has observations are read, so a close made
    on another slot by a verdict is not reported here.
    """
    mine = await store.observations_of_episode(episode_id)
    effects: dict[tuple[str, str], Effect] = {}
    for slot_key in sorted({o.slot_key for o in mine}):
        state = await store.load_slot(slot_key)
        by_id = {o.obs_id: o for o in await store.observations_of_slot(slot_key)}
        episode_ids = {o.obs_id for o in mine if o.slot_key == slot_key}
        for fact in state.facts:
            if fact.expired_at is not None:
                continue
            ordered = _ordered([by_id[i] for i in fact.evidence if i in by_id])
            for obs_id in sorted(episode_ids):
                if obs_id not in fact.evidence and fact.closed_by != obs_id:
                    continue
                role = _role(fact, obs_id, ordered)
                effects[(fact.fact_id, role.value)] = Effect(
                    fact.fact_id, slot_key, role, fact.status
                )
    return tuple(effects[key] for key in sorted(effects))


async def belief_chain(
    store: TemporalStore,
    *,
    subject_id: str,
    relation_type_id: int,
    now: datetime,
) -> tuple[ChainLink, ...]:
    """Return every version of the facts for a subject and relation, oldest first.

    Each link carries the audit row that ended its version, if there is one.
    Read top to bottom, it shows how belief about one subject and relation moved.
    """
    views = await store.query(
        TemporalQuery(
            now=now,
            include_history=True,
            subject_id=subject_id,
            relation_type_id=relation_type_id,
        )
    )
    closers: dict[str, list[Supersession]] = {}
    for slot_key in {v.slot_key for v in views}:
        for row in await store.supersessions_of_slot(slot_key):
            closers.setdefault(row.old_fact_id, []).append(row)
    links: list[ChainLink] = []
    for view in sorted(views, key=lambda v: (v.valid_from, v.fact_id, v.version)):
        ends = [
            row
            for row in closers.get(view.fact_id, [])
            if row.applied_valid_to == view.valid_to
        ]
        closer = max(ends, key=lambda row: row.created_at) if ends else None
        links.append(ChainLink(view, closer))
    return tuple(links)
