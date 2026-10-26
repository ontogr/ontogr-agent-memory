"""Tests for what importing the package loads."""

import ast
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest


BLOCKED_EXTRAS = (
    "falkordb",
    "redis",
    "neo4j",
    "asyncpg",
    "rdflib",
    "gliner",
    "spacy",
    "yaml",
    "opentelemetry",
)

OPTIONAL_SUBPACKAGES = (
    "ontogr_mem.graphstore.falkordb",
    "ontogr_mem.graphstore.neo4j",
    "ontogr_mem.vectorstore.graph",
    "ontogr_mem.vectorstore.pgvector",
)

GRAPHSTORE_KERNEL_MODULES = (
    "types",
    "protocol",
    "executor",
    "config",
    "errors",
    "values",
    "ident",
    "indices",
    "batching",
    "prepare",
)


def run_clean(code: str) -> subprocess.CompletedProcess[str]:
    """Run code in an isolated interpreter and return the finished process."""
    return subprocess.run(
        [sys.executable, "-I", "-c", textwrap.dedent(code)],
        capture_output=True,
        text=True,
        check=False,
    )


class TestImportBoundaries:
    """Importing the package must stay light and free of optional extras."""

    def test_data_models_loads_no_other_package_module(self):
        """data_models may load only the package root and the error module."""
        result = run_clean(
            """
            import sys
            import ontogr_mem.data_models
            allowed = {"ontogr_mem", "ontogr_mem.errors"}
            extra = sorted(
                name for name in sys.modules
                if name.startswith("ontogr_mem")
                and name not in allowed
                and not name.startswith("ontogr_mem.data_models")
            )
            print(extra)
            sys.exit(1 if extra else 0)
            """
        )

        assert result.returncode == 0, result.stdout + result.stderr

    def test_graphstore_kernel_modules_import_no_component(self):
        """Kernel modules do not import component packages."""
        result = run_clean(
            f"""
            import importlib
            import sys

            modules = {GRAPHSTORE_KERNEL_MODULES!r}
            components = {{
                "consolidation",
                "entity_extraction",
                "fact_extraction",
                "ingestion",
                "llm",
                "ontology",
                "resolution",
                "retrieval",
                "temporal",
                "vectorstore",
            }}

            for module in modules:
                importlib.import_module(f"ontogr_mem.graphstore.{{module}}")

            imported = sorted(
                name for name in sys.modules
                if name.startswith("ontogr_mem.")
                and name.split(".")[1] in components
            )
            print(imported)
            sys.exit(1 if imported else 0)
            """
        )

        assert result.returncode == 0, result.stdout + result.stderr

    @pytest.mark.parametrize("blocked", [BLOCKED_EXTRAS])
    def test_core_modules_import_without_any_extra(self, blocked):
        """Core modules import when optional backend and door packages are absent."""
        result = run_clean(
            f"""
            import importlib
            import importlib.abc
            import pkgutil
            import sys

            blocked = {blocked!r}
            skip = {OPTIONAL_SUBPACKAGES!r}

            class Blocker(importlib.abc.MetaPathFinder):
                def find_spec(self, name, path=None, target=None):
                    if name.split(".")[0] in blocked:
                        raise ImportError(name)
                    return None

            sys.meta_path.insert(0, Blocker())
            import ontogr_mem

            for info in pkgutil.walk_packages(
                ontogr_mem.__path__, "ontogr_mem."
            ):
                if info.name.startswith(skip):
                    continue
                importlib.import_module(info.name)
            """
        )

        assert result.returncode == 0, result.stdout + result.stderr

    def test_service_loads_without_door_extras(self):
        """The service package needs no optional dependency at import time."""
        result = run_clean(
            f"""
            import importlib.abc
            import sys

            blocked = {BLOCKED_EXTRAS!r}

            class Blocker(importlib.abc.MetaPathFinder):
                def find_spec(self, name, path=None, target=None):
                    if name.split(".")[0] in blocked:
                        raise ImportError(name)
                    return None

            sys.meta_path.insert(0, Blocker())
            import ontogr_mem.service.errors
            import ontogr_mem.service.contracts
            import ontogr_mem.service.principal
            import ontogr_mem.service.limits
            """
        )

        assert result.returncode == 0, result.stdout + result.stderr

    def test_observability_imports_only_allowed_package_modules(self):
        """Tracing helpers stay independent of service and component packages."""
        package = Path(__file__).parents[2] / "ontogr_mem" / "observability"
        allowed = {
            "ontogr_mem._extras",
            "ontogr_mem.errors",
            "ontogr_mem.observability",
        }

        for path in package.glob("*.py"):
            tree = ast.parse(path.read_text(), filename=str(path))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    names = {alias.name for alias in node.names}
                    assert not any(
                        name == "opentelemetry" or name.startswith("opentelemetry.")
                        for name in names
                    )
                    imports = names
                elif isinstance(node, ast.ImportFrom):
                    module = node.module or ""
                    assert module != "opentelemetry" and not module.startswith(
                        "opentelemetry."
                    )
                    imports = {module} if module else set()
                else:
                    continue

                assert all(
                    not name.startswith("ontogr_mem.")
                    or name in allowed
                    or name.startswith("ontogr_mem.observability.")
                    for name in imports
                ), path

    def test_components_data_models_and_stores_do_not_import_service(self):
        """Nothing outside the service package loads the service package."""
        result = run_clean(
            f"""
            import importlib
            import pkgutil
            import sys

            skip = (*{OPTIONAL_SUBPACKAGES!r}, "ontogr_mem.service")
            import ontogr_mem

            def walk(package, prefix):
                for info in pkgutil.iter_modules(package.__path__, prefix):
                    if info.name.startswith(skip):
                        continue
                    module = importlib.import_module(info.name)
                    if info.ispkg:
                        walk(module, info.name + ".")

            walk(ontogr_mem, "ontogr_mem.")

            loaded = sorted(
                name for name in sys.modules
                if name.startswith("ontogr_mem.service")
            )
            print(loaded)
            sys.exit(1 if loaded else 0)
            """
        )

        assert result.returncode == 0, result.stdout + result.stderr

    def test_service_models_do_not_load_the_graph_store(self):
        """The contract models stay free of the graph store."""
        result = run_clean(
            """
            import sys
            import ontogr_mem.service.contracts

            loaded = sorted(
                name for name in sys.modules
                if name.startswith("ontogr_mem.graphstore")
            )
            print(loaded)
            sys.exit(1 if loaded else 0)
            """
        )

        assert result.returncode == 0, result.stdout + result.stderr


class TestBackendImports:
    """A backend subpackage needs only its own extra."""

    def test_falkordb_backend_imports_without_the_other_backends(self):
        """The FalkorDB backend does not touch neo4j or asyncpg."""
        result = run_clean(
            """
            import importlib.abc
            import sys

            class Blocker(importlib.abc.MetaPathFinder):
                def find_spec(self, name, path=None, target=None):
                    if name.split(".")[0] in ("neo4j", "asyncpg"):
                        raise ImportError(name)
                    return None

            sys.meta_path.insert(0, Blocker())
            import ontogr_mem.graphstore.falkordb
            """
        )

        assert result.returncode == 0, result.stdout + result.stderr

    def test_missing_extra_names_the_install_command(self):
        """Without falkordb installed, importing the backend explains the fix."""
        result = run_clean(
            """
            import importlib.abc
            import sys

            class Blocker(importlib.abc.MetaPathFinder):
                def find_spec(self, name, path=None, target=None):
                    if name.split(".")[0] == "falkordb":
                        raise ModuleNotFoundError(name, name="falkordb")
                    return None

            sys.meta_path.insert(0, Blocker())
            try:
                import ontogr_mem.graphstore.falkordb
            except Exception as exc:
                print(type(exc).__name__, exc)
                sys.exit(0)
            sys.exit(1)
            """
        )

        assert result.returncode == 0, result.stdout + result.stderr
        assert 'pip install "ontogr-mem[falkordb]"' in result.stdout
