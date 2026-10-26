"""Resolver settings. Every threshold is a design value, not a measurement."""

from dataclasses import dataclass

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class ResolverSettings(BaseSettings):
    """Tunable behaviour of the resolver.

    Environment variables use the prefix `MEMORY_RESOLUTION_`. The cosine and
    Jaccard bars are separate because the two signals are not on one scale.

    Attributes:
        resolver_version: Version stamped on mentions and used in their ids.
        fuzzy_jaccard: Jaccard value that allows an automatic fuzzy match.
        ann_floor: Cosine floor for an embedding candidate.
        ann_top_k: Maximum embedding candidates per entity.
        single_vote_min: Cosine value that needs no second verification vote.
        jaccard_floor: Jaccard floor used to rank and filter candidates.
        jaccard_single_vote_min: Jaccard value that needs no second vote.
        inline_verification: Whether resolve verifies ambiguous pairs.
        cluster_cap: Maximum entities in one deduplication clique.
        summary_stale_after: Number of descriptions that makes a summary stale.
        pairs_per_call: Maximum pairs in one verification request.
        pair_token_budget: Token budget of one verification request.
        verification_votes: Votes a pair needs when it needs more than one.
    """

    model_config = SettingsConfigDict(
        env_prefix="MEMORY_RESOLUTION_",
        env_file=".env",
        extra="ignore",
        frozen=True,
    )

    resolver_version: str = "1"
    fuzzy_jaccard: float = Field(default=0.9, ge=0.0, le=1.0)
    ann_floor: float = Field(default=0.80, ge=0.0, le=1.0)
    ann_top_k: int = Field(default=20, ge=1)
    single_vote_min: float = Field(default=0.96, ge=0.0, le=1.0)
    jaccard_floor: float = Field(default=0.60, ge=0.0, le=1.0)
    jaccard_single_vote_min: float = Field(default=0.92, ge=0.0, le=1.0)
    inline_verification: bool = False
    cluster_cap: int = Field(default=8, ge=2)
    summary_stale_after: int = Field(default=8, ge=1)
    pairs_per_call: int = Field(default=5, ge=1)
    pair_token_budget: int = Field(default=4000, ge=1)
    verification_votes: int = Field(default=2, ge=1)


@dataclass(frozen=True, slots=True)
class Budget:
    """Limits of one deduplication run.

    Attributes:
        max_llm_pairs: Maximum pairs sent to the verifier.
        sweep: Number of old entities to check in addition to changed ones.
    """

    max_llm_pairs: int = 500
    sweep: int = 0
