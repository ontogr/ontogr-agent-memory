"""Write a vector for every new fact, after a slot commit.

The observer reads new fact versions from the temporal slot commit.
"""

from collections.abc import Mapping, Sequence

from ontogr_mem.data_models import Fact, Observation, to_micros
from ontogr_mem.temporal.temporal_store import SlotCommit
from ontogr_mem.vectorstore.models import (
    EdgeExtras,
    EmbedItem,
    Validity,
    VectorKind,
)
from ontogr_mem.vectorstore.text import edge_text
from ontogr_mem.vectorstore.writer import VectorWriter


def _distinct_facts(facts: Sequence[Fact]) -> list[Fact]:
    """Return the first version of each fact id, in the order they appear."""
    seen: set[str] = set()
    out: list[Fact] = []
    for fact in facts:
        if fact.fact_id in seen:
            continue
        seen.add(fact.fact_id)
        out.append(fact)
    return out


def _statement(fact: Fact, observations: Mapping[str, Observation]) -> str:
    """Return the statement of the first evidence observation that has one."""
    for obs_id in fact.evidence:
        observation = observations.get(obs_id)
        if observation is not None and observation.statement:
            return observation.statement
    return ""


def validity_of(fact: Fact) -> Validity:
    """Return the validity mirror of a fact version.

    An open end is `OPEN_MICROS`.
    """
    return Validity(
        status=fact.status,
        valid_from_us=to_micros(fact.valid_from),
        recorded_at_us=to_micros(fact.recorded_at),
        valid_to_us=to_micros(fact.valid_to),
        expired_at_us=to_micros(fact.expired_at),
    )


def _embed_items(
    facts: Sequence[Fact], observations: Mapping[str, Observation]
) -> list[EmbedItem]:
    """Build one vector item for each fact with a stored statement."""
    items: list[EmbedItem] = []
    for fact in _distinct_facts(facts):
        statement = _statement(fact, observations)
        if not statement:
            continue
        items.append(
            EmbedItem(
                fact.fact_id,
                edge_text(statement, "", "", ""),
                EdgeExtras(
                    subject_id=fact.subject_id,
                    object_id=fact.object_id,
                    validity=validity_of(fact),
                    relation_type_id=fact.relation_type_id,
                ),
            )
        )
    return items


class FactVectorObserver:
    """Writes a vector for every new fact of a slot commit.

    The observer writes its vectors after the commit, so the fact exists in
    the graph before the vector points at it. A fact whose statement is empty
    is skipped, because the fact sentence would name no entity.

    Args:
        writer: The writer that stores the vectors.
    """

    def __init__(self, writer: VectorWriter) -> None:
        """Bind the writer that stores the vectors."""
        self._writer = writer
        self._skipped_empty = 0

    @property
    def skipped_empty(self) -> int:
        """The number of facts that were skipped for want of a statement."""
        return self._skipped_empty

    async def before_slot_commit(self, commit: SlotCommit) -> None:
        """Do nothing; the vectors are written after the commit."""

    async def after_slot_commit(
        self, commit: SlotCommit, observations: Mapping[str, Observation]
    ) -> None:
        """Write one vector per new fact of the commit.

        Args:
            commit: The commit that was written.
            observations: The observations of the slot, keyed by observation id.
        """
        items = _embed_items(commit.new_facts, observations)
        self._skipped_empty += len(_distinct_facts(commit.new_facts)) - len(items)
        if not items:
            return
        await self._writer.write(VectorKind.EDGE, items)

    async def rewrite(
        self, facts: Sequence[Fact], observations: Mapping[str, Observation]
    ) -> int:
        """Rewrite vectors for current facts after their evidence changes.

        Args:
            facts: Current fact versions to write.
            observations: Evidence observations keyed by observation id.

        Returns:
            The number of distinct facts with a non-empty statement.
        """
        items = _embed_items(facts, observations)
        self._skipped_empty += len(_distinct_facts(facts)) - len(items)
        if not items:
            return 0
        await self._writer.write(VectorKind.EDGE, items)
        return len(items)
