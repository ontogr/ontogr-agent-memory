"""Check optional package and model requirements before opening resources."""

from dataclasses import dataclass
from importlib.util import find_spec
from typing import Literal, Mapping

from ontogr_mem.errors import MissingExtraError
from ontogr_mem.llm.failures import ModelNotConfiguredError
from ontogr_mem.llm.models import RoleModels
from ontogr_mem.llm.roles import ModelRole
from ontogr_mem.store_urls import GraphTarget, StateTarget
from ontogr_mem.vectorstore.builtin.select import choose_provider
from ontogr_mem.vectorstore.builtin.settings import EmbeddingSettings


@dataclass(frozen=True, slots=True)
class Requirement:
    """Modules and features provided by one optional extra."""

    modules: tuple[str, ...]
    features: tuple[str, ...]


REQUIREMENTS: Mapping[str, Requirement] = {
    "falkordb": Requirement(("falkordb", "redis"), ("FalkorDB graph",)),
    "neo4j": Requirement(("neo4j",), ("Neo4j graph",)),
    "postgres": Requirement(("asyncpg",), ("PostgreSQL state store",)),
    "fastembed": Requirement(("fastembed",), ("local FastEmbed embedder",)),
    "mcp": Requirement(("fastmcp",), ("MCP server",)),
    "otel": Requirement(("opentelemetry",), ("OpenTelemetry",)),
    "gliner": Requirement(("gliner",), ("GLiNER extraction",)),
    "spacy": Requirement(("spacy",), ("spaCy extraction",)),
    "local": Requirement(
        ("falkordb", "redis", "fastembed", "fastmcp"), ("local development",)
    ),
}
TEXT_WRITE_ROLES = (ModelRole.ENTITY_EXTRACTION, ModelRole.FACT_EXTRACTION)


@dataclass(frozen=True, slots=True)
class Preflight:
    """Report optional extras and model roles needed for an operation."""

    required: tuple[str, ...]
    missing: tuple[str, ...]
    missing_roles: tuple[ModelRole, ...]
    mode: Literal["keyless", "model"]

    def raise_for_missing(self, *, subcommand: str = "") -> None:
        """Raise actionable errors for missing extras or model roles."""
        if self.missing:
            features = tuple(
                feature
                for name in self.missing
                for feature in REQUIREMENTS[name].features
            )
            raise MissingExtraError(
                self.missing,
                features=features,
                needed=self.required,
                subcommand=subcommand,
            )
        if self.missing_roles:
            roles = ", ".join(role.value for role in self.missing_roles)
            msg = f"no model configured for {roles}"
            raise ModelNotConfiguredError(msg)


def check_requirements(
    *,
    models: RoleModels | None,
    graph: GraphTarget,
    state: StateTarget,
    embedding: EmbeddingSettings | None,
    mcp: bool = False,
    required_roles: frozenset[ModelRole] = frozenset(),
) -> Preflight:
    """Check required extras and model roles without importing optional packages."""
    required = [graph.backend]
    if state.backend == "postgres":
        required.append("postgres")
    if embedding is not None and choose_provider(embedding) == "fastembed":
        required.append("fastembed")
    if mcp:
        required.append("mcp")
    missing = []
    for name in required:
        try:
            present = all(
                find_spec(module) is not None for module in REQUIREMENTS[name].modules
            )
        except (ImportError, ValueError):
            present = False
        if not present:
            missing.append(name)
    missing_roles = tuple(
        role
        for role in ModelRole
        if models is not None and role in required_roles and not models.configured(role)
    )
    mode: Literal["keyless", "model"] = (
        "model"
        if models is not None
        and all(models.configured(role) for role in TEXT_WRITE_ROLES)
        else "keyless"
    )
    return Preflight(tuple(required), tuple(missing), missing_roles, mode)
