"""Resolved entities and the records that track their names and mentions.

The verdict and merge records live in the sibling modules `verdicts` and
`merges`. This module re-exports them so the resolution layer imports all of
its records from one place.
"""

from datetime import datetime
from enum import StrEnum
from typing import Any

from pydantic import Field

from ontogr_mem.data_models.base import _Model
from ontogr_mem.data_models.merges import MergeRecord, MergeState  # noqa: F401
from ontogr_mem.data_models.verdicts import (  # noqa: F401
    EntityVerdict,
    VerdictKind,
    VerdictSource,
    VerifyStatus,
)


class EntityStatus(StrEnum):
    """Lifecycle of a canonical entity.

    A ``MERGED`` entity points at its survivor through ``merged_into``. An
    ``ARCHIVED`` entity stays readable by id and is restored to ``ACTIVE`` when
    a new mention matches it.
    """

    ACTIVE = "active"
    MERGED = "merged"
    ARCHIVED = "archived"


class ResolutionState(StrEnum):
    """Whether an entity's identity has been fully checked."""

    RESOLVED = "resolved"
    PROVISIONAL = "provisional"


class AliasSource(StrEnum):
    """Where an alias came from."""

    EXTRACTED = "extracted"
    MERGED = "merged"
    USER = "user"
    ONTOLOGY = "ontology"


class DecisionMethod(StrEnum):
    """How a mention was linked to an entity."""

    URI = "uri"
    KEY = "key"
    ALIAS = "alias"
    FUZZY = "fuzzy"
    LLM = "llm"
    USER = "user"
    NONE = "none"


class Outcome(StrEnum):
    """Result of resolving one mention."""

    MATCHED = "matched"
    CREATED = "created"
    CREATED_PROVISIONAL = "created_provisional"


class Entity(_Model):
    """A canonical entity of one tenant.

    Attributes:
        entity_id: Deterministic id. A merged id is never reused.
        type_label: Entity type label, such as ``Person``.
        canonical_name: Name chosen when the entity was created.
        strict_key: Normalized lookup key of the canonical name.
        loose_key: Blocking key of the canonical name.
        uri: Bound identifier from the source, if any.
        identity_attrs: Normalized values of the identifying attributes.
        summary: Short description, possibly stale.
        descriptions: Descriptions collected from mentions.
        attributes: Scalar attributes collected from mentions.
        mention_count: Number of mentions linked to the entity.
        status: Lifecycle state.
        merged_into: Survivor id while ``status`` is ``MERGED``.
        resolution_state: Whether the identity has been fully checked.
        candidates_incomplete: True when candidate search missed entities.
        candidate_ids: Candidates that still need a verdict.
        summary_stale: True when the summary should be rewritten.
        revision: Store revision of the last write.
        last_checked_version: Revision at the last consolidation check.
        created_at: When the entity was created.
        updated_at: When the entity was last changed.
    """

    entity_id: str
    type_label: str
    canonical_name: str
    strict_key: str
    loose_key: str
    uri: str | None = None
    identity_attrs: dict[str, str] = Field(default_factory=dict)
    summary: str = ""
    descriptions: list[str] = Field(default_factory=list)
    attributes: dict[str, Any] = Field(default_factory=dict)
    mention_count: int = 0
    status: EntityStatus = EntityStatus.ACTIVE
    merged_into: str | None = None
    resolution_state: ResolutionState = ResolutionState.RESOLVED
    candidates_incomplete: bool = False
    candidate_ids: list[str] = Field(default_factory=list)
    summary_stale: bool = False
    revision: int = 0
    last_checked_version: int = 0
    created_at: datetime
    updated_at: datetime


class Alias(_Model):
    """A name that resolves to an entity. The canonical name is one too.

    Attributes:
        alias_id: Deterministic id built from the entity and the strict key.
        entity_id: Entity the name resolves to.
        text: Surface form as written.
        strict_key: Normalized lookup key of the text.
        loose_key: Blocking key of the text.
        source: Where the alias came from.
        merge_id: Merge that moved the alias, while ``source`` is ``MERGED``.
        first_episode_id: Episode where the name first appeared.
    """

    alias_id: str
    entity_id: str
    text: str
    strict_key: str
    loose_key: str
    source: AliasSource = AliasSource.EXTRACTED
    merge_id: str | None = None
    first_episode_id: str = ""


class Mention(_Model):
    """One occurrence of an entity in an episode.

    The resolution decision is stored on the mention, so its id is also the
    decision id.

    Attributes:
        mention_id: Deterministic id built from the episode, local id and
            resolver version.
        entity_id: Entity the mention is linked to.
        episode_id: Episode the mention came from.
        local_id: Id that is unique inside the episode.
        surface_form: Name as written in the episode.
        type_label_at_extraction: Type label the extractor assigned.
        quote: Text that supports the mention, if any.
        resolver_version: Resolver version that linked the mention.
        seq: Store revision of the write.
        decision_method: How the mention was linked.
        decision_outcome: Whether the entity was matched or created.
        decision_score: Score that decided the link, if any.
        decision_reason: Stable reason code, such as ``key_hit``.
        decision_candidates: Candidate ids the decision considered.
    """

    mention_id: str
    entity_id: str
    episode_id: str
    local_id: str
    surface_form: str
    type_label_at_extraction: str
    resolver_version: str
    decision_method: DecisionMethod
    decision_outcome: Outcome
    quote: str | None = None
    seq: int = 0
    decision_score: float | None = None
    decision_reason: str = ""
    decision_candidates: list[str] = Field(default_factory=list)
