"""Settings of the retrieval component."""

import math
from typing import Self

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class RetrievalSettings(BaseSettings):
    """Candidate counts, fusion constants and status penalties of retrieval.

    Environment variables use the prefix `MEMORY_RETRIEVAL_`.

    Attributes:
        candidate_factor: Factor by which a lane over-fetches candidates.
        fusion_k_base: Base of the reciprocal rank fusion constant.
        fusion_k_per_limit: Growth of the fusion constant with the limit.
        fusion_k_min: Lowest value the fusion constant takes.
        fusion_k_max: Highest value the fusion constant takes.
        seed_cap: Seeds the graph lane walks from.
        seed_fact_count: Facts whose endpoints become seeds.
        frontier_cap: Endpoints the graph lane visits in one hop.
        disputed_penalty: Multiplier of a fact in an open dispute.
        overridden_penalty: Multiplier of an overridden fact.
        entity_candidates: Entities the entity lane looks for.
        max_cards: Entity cards the result holds.
        rerank_weight: Reranker share of blended scores.
        shortlist_factor: Results per requested item sent to the reranker.
        facts_share: Context budget fraction initially reserved for facts.
        episodes_share: Context budget fraction initially reserved for episodes.
        entities_share: Context budget fraction initially reserved for entities.
        passage_chars: Maximum episode characters in a passage or context item.
    """

    model_config = SettingsConfigDict(
        env_prefix="MEMORY_RETRIEVAL_", env_file=".env", extra="ignore", frozen=True
    )

    candidate_factor: int = Field(default=2, ge=1)
    fusion_k_base: int = Field(default=20, ge=0)
    fusion_k_per_limit: int = Field(default=2, ge=0)
    fusion_k_min: int = Field(default=30, ge=1)
    fusion_k_max: int = Field(default=60, ge=1)
    seed_cap: int = Field(default=12, ge=0)
    seed_fact_count: int = Field(default=5, ge=0)
    frontier_cap: int = Field(default=200, ge=1)
    disputed_penalty: float = Field(default=0.8, ge=0.0, le=1.0)
    overridden_penalty: float = Field(default=0.7, ge=0.0, le=1.0)
    entity_candidates: int = Field(default=10, ge=1)
    max_cards: int = Field(default=10, ge=1)
    rerank_weight: float = Field(default=0.5, ge=0.0, le=1.0)
    shortlist_factor: int = Field(default=2, ge=1)
    facts_share: float = Field(default=0.5, ge=0.0, le=1.0)
    episodes_share: float = Field(default=0.3, ge=0.0, le=1.0)
    entities_share: float = Field(default=0.2, ge=0.0, le=1.0)
    passage_chars: int = Field(default=400, ge=1)

    @model_validator(mode="after")
    def validate_context_shares(self) -> Self:
        """Require the three initial context shares to sum to one."""
        if not math.isclose(
            self.facts_share + self.episodes_share + self.entities_share, 1.0
        ):
            raise ValueError("context budget shares must sum to 1.0")
        return self


class KeywordSettings(BaseSettings):
    """Limits for tenant-scoped fulltext search.

    Attributes:
        tenant_overfetch: Shared-index candidate multiplier.
            Env: `MEMORY_KEYWORD_TENANT_OVERFETCH`.
        max_terms: Maximum query terms, at most 32.
            Env: `MEMORY_KEYWORD_MAX_TERMS`.
        timeout_ms: Server timeout; None uses the runner's resilience default.
            Env: `MEMORY_KEYWORD_TIMEOUT_MS`.
    """

    model_config = SettingsConfigDict(
        env_prefix="MEMORY_KEYWORD_", env_file=".env", extra="ignore", frozen=True
    )

    tenant_overfetch: int = Field(default=4, ge=1)
    max_terms: int = Field(default=32, ge=1, le=32)
    timeout_ms: int | None = Field(default=None, ge=1)
