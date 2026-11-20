"""Generate the settings table for the configuration reference page.

The script reads ``model_fields`` on each settings class without instantiating
it, so no ``.env`` file is loaded. It also checks that every settings module
and every ``MEMORY_*`` name in the package appears in the generated table.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from types import ModuleType

from pydantic import BaseModel, SecretStr
from pydantic_core import PydanticUndefined
from pydantic_settings import BaseSettings

from ontogr_mem.combined_extraction import settings as combined_extraction_settings
from ontogr_mem.consolidation import config as consolidation_config
from ontogr_mem.entity_extraction import settings as entity_extraction_settings
from ontogr_mem.fact_extraction import settings as fact_extraction_settings
from ontogr_mem.graphstore.falkordb import settings as falkordb_settings
from ontogr_mem.graphstore.neo4j import settings as neo4j_settings
from ontogr_mem.llm import config as llm_config
from ontogr_mem.mcp_server import settings as mcp_server_settings
from ontogr_mem.relational_store import config as relational_store_config
from ontogr_mem.resolution import config as resolution_config
from ontogr_mem.retrieval import settings as retrieval_settings
from ontogr_mem.server import settings as server_settings
from ontogr_mem.service import bootstrap as service_bootstrap
from ontogr_mem.service.access import settings as access_settings
from ontogr_mem.service.forget import settings as forget_settings
from ontogr_mem.temporal import settings as temporal_settings
from ontogr_mem.vectorstore import settings as vectorstore_settings
from ontogr_mem.vectorstore.builtin import settings as builtin_settings


ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "ontogr_mem"
TARGET = ROOT / "docs" / "docs" / "reference" / "configuration.mdx"
BEGIN = "{/* BEGIN GENERATED CONFIG */}"
END = "{/* END GENERATED CONFIG */}"

SETTINGS_MODULES: tuple[ModuleType, ...] = (
    combined_extraction_settings,
    consolidation_config,
    entity_extraction_settings,
    fact_extraction_settings,
    falkordb_settings,
    neo4j_settings,
    llm_config,
    mcp_server_settings,
    relational_store_config,
    resolution_config,
    retrieval_settings,
    server_settings,
    service_bootstrap,
    access_settings,
    forget_settings,
    temporal_settings,
    vectorstore_settings,
    builtin_settings,
)

# Variables read by plain os code or by a caller, not by a settings field.
NON_SETTINGS = {
    "MEMORY_DATA_DIR": "Directory for data, state, and server files.",
    "FASTEMBED_CACHE_PATH": "Directory for local embedding model files.",
}

# Enum members and other names that look like variables in prose but are not.
IGNORED_TOKENS = frozenset({"MEMORY_ONLY"})

TOKEN = re.compile(r"MEMORY_[A-Z0-9_]+")


def _default(field: object) -> str:
    """Return the printed default of a field, hiding secret values."""
    annotation = getattr(field, "annotation", None)
    if isinstance(annotation, type) and issubclass(annotation, SecretStr):
        return "unset"
    default = getattr(field, "default", PydanticUndefined)
    if default is PydanticUndefined:
        return "unset"
    if isinstance(default, SecretStr):
        return "unset"
    text = str(default).replace("|", "\\|").replace("\n", " ")
    return text if text else "empty"


def _description(field: object) -> str:
    """Return the first line of a field's description."""
    doc = getattr(field, "description", "") or ""
    text = doc.strip().splitlines()[0] if doc.strip() else ""
    return text.replace("|", "\\|")


def _collect(
    model: type[BaseModel],
    prefix: str,
    rows: dict[str, tuple[str, str]],
    seen: frozenset[type[BaseModel]],
) -> None:
    """Add every environment name of a model and its nested models to rows."""
    for name, field in model.model_fields.items():
        alias = getattr(field, "validation_alias", None)
        key = alias if isinstance(alias, str) else f"{prefix}{name.upper()}"
        rows[key] = (_default(field), _description(field))
        annotation = field.annotation
        if isinstance(annotation, type) and issubclass(annotation, BaseModel):
            if annotation in seen or issubclass(annotation, BaseSettings):
                continue
            nested_prefix = f"{key}__"
            _collect(annotation, nested_prefix, rows, seen | {annotation})


def build_rows() -> dict[str, tuple[str, str]]:
    """Return environment names with their defaults and descriptions."""
    rows: dict[str, tuple[str, str]] = {}
    for module in SETTINGS_MODULES:
        for value in vars(module).values():
            if not isinstance(value, type) or not issubclass(value, BaseSettings):
                continue
            if value is BaseSettings or value.__module__ != module.__name__:
                continue
            prefix = value.model_config.get("env_prefix", "")
            if not prefix:
                continue
            _collect(value, prefix, rows, frozenset({value}))
    return dict(sorted(rows.items()))


def check_modules() -> None:
    """Fail when a module that sets env_prefix is not in the import tuple."""
    expected = {
        str(path.relative_to(ROOT))
        for path in PACKAGE.rglob("*.py")
        if "env_prefix" in path.read_text(encoding="utf-8")
    }
    imported = {
        str(Path(module.__file__).relative_to(ROOT)) for module in SETTINGS_MODULES
    }
    missing = expected - imported
    if missing:
        names = ", ".join(sorted(missing))
        raise SystemExit(f"Settings modules missing from the generator: {names}")


def check_completeness(rows: dict[str, tuple[str, str]]) -> None:
    """Fail when a MEMORY_* name in the package has no row in the table."""
    names = set(rows) | {key for key in NON_SETTINGS if key.startswith("MEMORY_")}
    found = {
        token
        for path in PACKAGE.rglob("*.py")
        for token in TOKEN.findall(path.read_text(encoding="utf-8"))
    }
    unknown = {
        token
        for token in found
        if not token.endswith("_")
        and token not in names
        and token not in IGNORED_TOKENS
    }
    if unknown:
        listed = ", ".join(sorted(unknown))
        raise SystemExit(f"MEMORY_* names missing from the config table: {listed}")


def render(rows: dict[str, tuple[str, str]]) -> str:
    """Return the Markdown table between the generated markers."""
    lines = [
        BEGIN,
        "",
        "| Variable | Default | Description |",
        "| --- | --- | --- |",
    ]
    for name, (default, description) in rows.items():
        lines.append(f"| `{name}` | `{default}` | {description} |")
    lines.extend(["", END])
    return "\n".join(lines)


def main() -> int:
    """Write or check the generated section of the configuration page."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    check_modules()
    rows = build_rows()
    check_completeness(rows)
    table = render(rows)

    current = TARGET.read_text(encoding="utf-8")
    start = current.find(BEGIN)
    end = current.find(END)
    if start < 0 or end < 0:
        raise SystemExit("Configuration page is missing the generated markers")
    updated = current[:start] + table + current[end + len(END) :]
    if current == updated:
        return 0
    if args.check:
        print("The generated configuration table is stale; run make docs-config")
        return 1
    TARGET.write_text(updated, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
