"""Builders shared by the temporal unit tests."""

from datetime import UTC, datetime

from ontogr_mem.data_models.fact_versions import Fact, FactStatus
from ontogr_mem.data_models.facts import Precision, TimeBasis
from ontogr_mem.data_models.observation import Observation, ObservationKind


class LocalRefResolver:
    """Resolve a local ref to itself."""

    async def entity_id_for_ref(self, episode_id: str, local_id: str) -> str | None:
        """Return the local ref unchanged."""
        return local_id


class EntityLookup:
    """Expose a fixed entity set to temporal unit tests."""

    def __init__(self, entity_ids: tuple[str, ...]) -> None:
        """Create a lookup containing ``entity_ids``."""
        self._entity_ids = frozenset(entity_ids)

    async def has_entity(self, entity_id: str) -> bool:
        """Return whether the fixed entity set contains ``entity_id``."""
        return entity_id in self._entity_ids


def at(year: int, month: int = 1, day: int = 1) -> datetime:
    """Build a UTC instant."""
    return datetime(year, month, day, tzinfo=UTC)


INGESTED = datetime(2030, 1, 1, tzinfo=UTC)


def obs(
    obs_id: str,
    obj: str,
    start: datetime | None,
    *,
    prec: Precision | None = Precision.YEAR,
    end: datetime | None = None,
    end_prec: Precision | None = None,
    kind: ObservationKind = ObservationKind.ASSERTION,
    episode: str | None = None,
    subject: str = "ana",
    rel: int = 1,
    one: bool = True,
    inverse: bool = False,
    holder: str = "acme",
    slot: str = "slot",
) -> Observation:
    """Build an observation for one slot.

    ``obj`` is the value that varies in the slot. Unless ``inverse`` is set it
    is the object; with ``inverse`` it is the subject and ``holder`` is the
    object.
    """
    if inverse:
        subject, object_id = obj, holder
    else:
        object_id = obj
    return Observation(
        obs_id=obs_id,
        slot_key=slot,
        subject_id=subject,
        object_id=object_id,
        relation_type_id=rel,
        kind=kind,
        episode_id=episode or f"ep-{obs_id}",
        occurred_at=start or INGESTED,
        ingested_at=INGESTED,
        valid_from=start,
        valid_from_precision=prec if start else None,
        valid_to=end,
        valid_to_precision=end_prec,
        time_basis=TimeBasis.STATED if start else TimeBasis.NONE,
        inverse=inverse,
    )


def fact_of(
    fact_id: str,
    observations: list[Observation],
    *,
    slot: str = "slot",
    status: FactStatus = FactStatus.ACTIVE,
) -> Fact:
    """Build the current version of a fact that holds the given observations.

    The value and interval come from the first observation's run: the fact takes
    the first observation's object, start and end, and evidence is every id.
    """
    first = observations[0]
    return Fact(
        fact_id=fact_id,
        version=1,
        slot_key=slot,
        subject_id=first.subject_id,
        object_id=first.object_id,
        relation_type_id=first.relation_type_id,
        valid_from=first.valid_from,
        valid_to=None,
        valid_from_precision=first.valid_from_precision,
        valid_to_precision=None,
        recorded_at=INGESTED,
        expired_at=None,
        status=status,
        evidence=tuple(sorted(o.obs_id for o in observations)),
        last_confirmed_at=INGESTED,
        inverse=first.inverse,
    )


def stated_end(obs_id: str, obj: str, when: datetime, **kw: object) -> Observation:
    """Build a stated end observation."""
    return obs(obs_id, obj, when, kind=ObservationKind.STATED_END, **kw)
