"""Entity resolution: link extracted names to canonical entities."""

from ontogr_mem.resolution.config import Budget, ResolverSettings
from ontogr_mem.resolution.errors import (
    ConcurrentEntityWriteError,
    EntityHasFactsError,
    EpisodeAlreadyResolvedError,
    MergeRefusedError,
    NotFoundError,
    ResolutionError,
    SnapshotPurgedError,
    UnmergeBlockedError,
)
from ontogr_mem.ontology.links import TypeRel
from ontogr_mem.resolution.policy import ResolutionPolicy, TypePolicy
from ontogr_mem.resolution.resolver import Decision, EntityResolver, ResolutionResult
from ontogr_mem.resolution.resolver_store import MergePlan, ResolverStore, UnmergePlan


__all__ = [
    "Budget",
    "ConcurrentEntityWriteError",
    "Decision",
    "EntityHasFactsError",
    "EntityResolver",
    "EpisodeAlreadyResolvedError",
    "MergePlan",
    "MergeRefusedError",
    "NotFoundError",
    "ResolutionError",
    "ResolutionPolicy",
    "ResolutionResult",
    "ResolverSettings",
    "ResolverStore",
    "SnapshotPurgedError",
    "TypePolicy",
    "TypeRel",
    "UnmergeBlockedError",
    "UnmergePlan",
]
