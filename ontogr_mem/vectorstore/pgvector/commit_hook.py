"""Turn committed fact changes into outbox work, before the graph commit.

`enqueue_commit` accepts any object with the attributes of `CommittedFact`.
The temporal engine does not need to import this module; this module does
not import `ontogr_mem.temporal`.
"""

from collections.abc import Sequence
from typing import Protocol

from ontogr_mem.vectorstore.models import (
    EdgeExtras,
    EmbedItem,
    Validity,
    VectorKind,
)
from ontogr_mem.vectorstore.pgvector.outbox import (
    Outbox,
    OutboxEntry,
    delete_entry,
    upsert_entry,
    validity_entry,
)
from ontogr_mem.vectorstore.text import edge_text


class CommittedFact(Protocol):
    """One fact touched by a successful slot commit.

    Attributes:
        fact_id: Stable fact id. A rewritten fact gets a new id in the temporal
            engine, so `is_new` also covers a rewritten fact.
        is_new: The fact id was not committed before.
        subject_id: Subject entity id.
        object_id: Object entity id.
        validity: The fact's current version, for the mirror.
        statement: Natural-language fact sentence; may be empty.
        subject_name: Entity name, used when the statement is empty.
        relation_name: Relation label, used when the statement is empty.
        object_name: Entity name, used when the statement is empty.
        relation_type_id: Catalog relation type id.
        episode_id: Episode that caused the change, for the completion gate.
        deleted: The fact was physically removed. Withdrawn, corrected or
            overridden facts are not deleted because as-of queries still need
            their vectors.
    """

    fact_id: str
    is_new: bool
    subject_id: str
    object_id: str
    validity: Validity
    statement: str
    subject_name: str
    relation_name: str
    object_name: str
    relation_type_id: int | None
    episode_id: str | None
    deleted: bool


def entries_for_commit(commits: Sequence[CommittedFact]) -> list[OutboxEntry]:
    """Return outbox work for a commit.

    A deleted fact becomes a delete, a new fact an upsert, and an existing fact
    a validity change.
    """
    out: list[OutboxEntry] = []
    for commit in commits:
        if commit.deleted:
            out.append(delete_entry(VectorKind.EDGE, commit.fact_id, commit.episode_id))
        elif commit.is_new:
            item = EmbedItem(
                commit.fact_id,
                edge_text(
                    commit.statement,
                    commit.subject_name,
                    commit.relation_name,
                    commit.object_name,
                ),
                EdgeExtras(
                    subject_id=commit.subject_id,
                    object_id=commit.object_id,
                    validity=commit.validity,
                    relation_type_id=commit.relation_type_id,
                ),
            )
            out.append(upsert_entry(VectorKind.EDGE, item, commit.episode_id))
        else:
            out.append(
                validity_entry(commit.fact_id, commit.validity, commit.episode_id)
            )
    return out


async def enqueue_commit(outbox: Outbox, commits: Sequence[CommittedFact]) -> int:
    """Write the outbox entries for a commit; call before the graph commit.

    Returns:
        The number of entries written.
    """
    entries = entries_for_commit(commits)
    if entries:
        await outbox.enqueue(entries)
    return len(entries)
