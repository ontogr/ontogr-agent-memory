"""Temporal reconciliation: observations in, versioned facts out."""

from ontogr_mem.data_models.fact_versions import (
    Fact,
    FactStatus,
    Supersession,
    SupersessionKind,
)
from ontogr_mem.data_models.facts import Precision, TimeBasis
from ontogr_mem.data_models.ledger import LedgerEntry, LedgerOp
from ontogr_mem.data_models.observation import (
    NO_STAMP,
    Observation,
    ObservationKind,
    ProvenanceStamp,
)
from ontogr_mem.data_models.timestamps import from_micros, to_micros
from ontogr_mem.temporal.adapter import (
    EntityRefResolver,
    Skipped,
    SkipReason,
    observations_from_facts,
)
from ontogr_mem.temporal.cardinality import (
    Cardinality,
    ConflictPolicy,
    DictRelationPolicy,
    RelationRules,
)
from ontogr_mem.temporal.engine import CommitObserver, TemporalEngine
from ontogr_mem.temporal.errors import (
    EpisodeAlreadyIngestedError,
    EpisodeMismatchError,
    MissingEntityError,
    MissingEpisodeError,
    MissingFactVersionError,
    RevisionConflict,
    TemporalError,
    UnknownRelationError,
)
from ontogr_mem.temporal.ledger import (
    GENESIS,
    checksum,
    link,
    payload_hash,
    verify,
)
from ontogr_mem.temporal.policy import from_type_view
from ontogr_mem.temporal.query import (
    FactView,
    TemporalQuery,
    matches,
    view_of,
)
from ontogr_mem.temporal.reader import FactReader, ObservationText
from ontogr_mem.temporal.reconcile import (
    Conflict,
    ConflictKind,
    Derived,
    Run,
    derive,
)
from ontogr_mem.temporal.render import render_fact_item
from ontogr_mem.temporal.report import Outcome, TemporalReport
from ontogr_mem.temporal.settings import TemporalSettings
from ontogr_mem.temporal.timeutil import (
    coarser,
    same_span,
    span_end,
    truncate,
)


__all__ = [
    "GENESIS",
    "NO_STAMP",
    "Cardinality",
    "CommitObserver",
    "Conflict",
    "ConflictKind",
    "ConflictPolicy",
    "Derived",
    "DictRelationPolicy",
    "EntityRefResolver",
    "EpisodeAlreadyIngestedError",
    "EpisodeMismatchError",
    "Fact",
    "FactReader",
    "FactStatus",
    "FactView",
    "LedgerEntry",
    "LedgerOp",
    "MissingEntityError",
    "MissingEpisodeError",
    "MissingFactVersionError",
    "Observation",
    "ObservationKind",
    "ObservationText",
    "Outcome",
    "Precision",
    "ProvenanceStamp",
    "RelationRules",
    "RevisionConflict",
    "Run",
    "SkipReason",
    "Skipped",
    "Supersession",
    "SupersessionKind",
    "TemporalError",
    "TemporalEngine",
    "TemporalQuery",
    "TemporalReport",
    "TemporalSettings",
    "TimeBasis",
    "UnknownRelationError",
    "checksum",
    "coarser",
    "derive",
    "from_micros",
    "from_type_view",
    "link",
    "matches",
    "observations_from_facts",
    "payload_hash",
    "render_fact_item",
    "same_span",
    "span_end",
    "to_micros",
    "truncate",
    "verify",
    "view_of",
]
