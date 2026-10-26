"""Merge records: what one merge did, enough to undo it.

A merge never deletes: the absorbed entity becomes ``MERGED`` with a pointer to
the survivor, and the record keeps a snapshot plus what moved.
"""

from datetime import datetime
from enum import StrEnum
from typing import Any, TypedDict

from pydantic import Field

from ontogr_mem.data_models.base import _Model


class MergeState(StrEnum):
    """State of a merge record.

    ``APPLYING`` marks a merge whose record is written and whose rows are still
    moving. Recovery finishes or reverses such a record.
    """

    APPLIED = "applied"
    APPLYING = "applying"
    REVERTED = "reverted"


class MergeSnapshot(TypedDict, total=False):
    """The absorbed entity and its aliases, as a merge record keeps them.

    The snapshot is written as plain JSON data, so it survives a round trip
    through the store without a schema of its own.

    Attributes:
        entity: The absorbed entity as JSON data.
        aliases: The aliases of the absorbed entity as JSON data.
        intent: The rows the merge writes, as JSON data.
    """

    entity: dict[str, Any]
    aliases: list[dict[str, Any]]
    intent: dict[str, Any]


class SurvivorDelta(TypedDict, total=False):
    """The survivor fields one merge changed, as a merge record keeps them.

    The delta holds enough to undo the change, so an unmerge can restore the
    survivor without the merge being replayed.

    Attributes:
        type_label: Type label of the survivor before the merge.
        uri: Bound identifier of the survivor before the merge.
        descriptions_added: Descriptions the merge added to the survivor.
        attribute_keys_added: Attribute keys the merge added to the survivor.
        identity_keys_added: Identity attribute keys the merge added.
        mention_count_added: Mentions the merge moved to the survivor.
        summary_stale: Whether the survivor summary was already stale.
        candidate_ids: Candidate ids of the survivor before the merge.
    """

    type_label: str
    uri: str | None
    descriptions_added: list[str]
    attribute_keys_added: list[str]
    identity_keys_added: list[str]
    mention_count_added: int
    summary_stale: bool
    candidate_ids: list[str]


class VerdictPayload(TypedDict, total=False):
    """One verdict row, as a merge record keeps it.

    Attributes:
        verdict_id: Id of the ordered pair.
        a_id: Smaller entity id.
        b_id: Larger entity id.
        verdict: The answer.
        source: Who produced the answer.
        inputs_hash: Hash of the entity data the answer was based on.
        passes: One vote per verification pass.
        superseded_by: Merge that made the answer no longer live.
    """

    verdict_id: str
    a_id: str
    b_id: str
    verdict: str
    source: str
    inputs_hash: str
    passes: list[str]
    superseded_by: str | None


class MergeRecord(_Model):
    """What a merge did, enough to undo it.

    Attributes:
        merge_id: Deterministic id of the merge.
        seq: Store revision of the write.
        survivor_id: Entity that absorbed the other.
        absorbed_id: Entity that was merged away.
        state: Lifecycle of the record.
        actor: Who asked for the merge.
        reason: Free-text reason.
        snapshot: Absorbed entity and aliases before the merge, as JSON data.
            See `MergeSnapshot`.
        snapshot_purged: True when the snapshot was blanked for privacy.
        survivor_delta: Survivor fields the merge changed, enough to undo them.
            See `SurvivorDelta`.
        replaced_verdicts: Earlier verdict rows the merge overwrote. See
            `VerdictPayload`.
        unmerge_moved_mention_ids: Mentions an unmerge moved back.
        moved_mention_ids: Mentions the merge moved to the survivor.
        moved_alias_ids: Aliases the merge moved to the survivor.
        moved_verdict_ids: Verdicts the merge superseded.
        created_verdict_ids: Verdicts the merge wrote for the survivor.
        redirected_ids: Merged entities whose pointer the merge retargeted.
        created_at: When the merge was recorded.
    """

    merge_id: str
    survivor_id: str
    absorbed_id: str
    actor: str
    reason: str
    seq: int = 0
    state: MergeState = MergeState.APPLIED
    snapshot: dict[str, Any] | None = None
    snapshot_purged: bool = False
    survivor_delta: dict[str, Any] = Field(default_factory=dict)
    replaced_verdicts: list[dict[str, Any]] = Field(default_factory=list)
    unmerge_moved_mention_ids: list[str] = Field(default_factory=list)
    moved_mention_ids: list[str] = Field(default_factory=list)
    moved_alias_ids: list[str] = Field(default_factory=list)
    moved_verdict_ids: list[str] = Field(default_factory=list)
    created_verdict_ids: list[str] = Field(default_factory=list)
    redirected_ids: list[str] = Field(default_factory=list)
    created_at: datetime
