.PHONY: sync test test-integration services-up services-down test-all cov-report cov lint-typing lint-style lint-fmt lint-check lint-typos lint-actions lint-all security-bandit security-audit security build wheel-test changelog clean help

export UV_LOCKED = 1

help:
	@echo "Available make targets:"
	@echo "  make sync             - Sync project and install dependencies"
	@echo "  make test             - Run unit tests with coverage"
	@echo "  make test-integration - Run integration tests (requires external services; INTEGRATION_PATHS=<paths> picks modules)"
	@echo "  make services-up      - Start the FalkorDB, Neo4j and Postgres test services (SERVICES=<names> starts only those)"
	@echo "  make services-down    - Stop the test services and remove their volumes"
	@echo "  make test-all         - Run unit and integration tests in order"
	@echo "  make cov-report       - Generate coverage reports (html)"
	@echo "  make cov              - Run tests and generate coverage reports"
	@echo "  make lint-typing      - Type check with ty"
	@echo "  make lint-style       - Lint with ruff (check only)"
	@echo "  make lint-fmt         - Format code and lint with auto-fixes"
	@echo "  make lint-check       - Check formatting and lint without modifying files"
	@echo "  make lint-typos       - Check for typos"
	@echo "  make lint-actions     - Audit GitHub Actions workflows with zizmor"
	@echo "  make lint-all         - Run formatting, linting, and type checking"
	@echo "  make security-bandit  - Run Bandit security scan"
	@echo "  make security-audit   - Audit the locked dependencies with uv audit"
	@echo "  make security         - Run all security scans"
	@echo "  make build            - Build sdist and wheel into dist/"
	@echo "  make wheel-test       - Install the built wheel in a clean env and import it"
	@echo "  make changelog        - Add a release section to CHANGELOG.md (VERSION=x.y.z)"
	@echo "  make clean            - Clean build artifacts and cache"

sync:
	uv sync --locked --all-groups --all-extras

test:
	uv run pytest tests/unit \
		--cov=ontogr_mem \
		--cov-report=term-missing \
		--cov-report=xml \
		--junitxml=pytest-results.xml

# INTEGRATION_PATHS narrows the run to some modules. CI runs one module per job.
INTEGRATION_PATHS ?= tests/integration
INTEGRATION_WORKERS ?= auto

test-integration:
	uv run pytest $(INTEGRATION_PATHS) -v --maxfail=1 -n $(INTEGRATION_WORKERS) --dist loadscope \
		-o "addopts=--strict-markers --strict-config --disable-socket --allow-unix-socket -ra" \
		--cov=ontogr_mem --cov-report=xml:coverage-integration.xml \
		--junitxml=pytest-integration-results.xml

# SERVICES names the compose services to start: falkordb, neo4j, pgvector and
# relational. It starts all of them when empty.
SERVICES ?=

services-up:
	docker compose -f tests/integration/compose.yaml up -d --wait $(SERVICES)

services-down:
	docker compose -f tests/integration/compose.yaml down -v

test-all: test test-integration

cov-report:
	uv run coverage html

cov: test cov-report

lint-typing:
	uv run ty check ontogr_mem/ tests

lint-style:
	uv run ruff check .

lint-fmt:
	uv run ruff format .
	uv run ruff check --fix --unsafe-fixes .

lint-check:
	uv run ruff format --check .
	uv run ruff check .

lint-typos:
	uv run typos

lint-actions:
	uvx zizmor .github/workflows/

lint-all: lint-fmt lint-typing lint-typos lint-actions

security-bandit:
	uv run bandit -c pyproject.toml -r ontogr_mem/ --severity-level high --confidence-level high

security-audit:
	uv audit --preview-features audit-command

security: security-bandit security-audit

build:
	rm -rf dist
	uv build

wheel-test: build
	rm -rf .wheelenv
	uv venv .wheelenv
	uv pip install --python .wheelenv/bin/python dist/*.whl
	cd /tmp && "$(CURDIR)/.wheelenv/bin/python" -c "import ontogr_mem, ontogr_mem.consolidation, ontogr_mem.data_models, ontogr_mem.entity_extraction, ontogr_mem.fact_extraction, ontogr_mem.graphstore, ontogr_mem.ingestion, ontogr_mem.llm, ontogr_mem.observability, ontogr_mem.ontology, ontogr_mem.relational_store, ontogr_mem.resolution, ontogr_mem.resilience, ontogr_mem.retrieval, ontogr_mem.service, ontogr_mem.temporal, ontogr_mem.vectorstore; print(ontogr_mem.__version__)"

changelog:
	@test -n "$(VERSION)" || { echo "Usage: make changelog VERSION=x.y.z"; exit 1; }
	@# git-cliff groups commits by tag, so a shallow or tagless clone gives a wrong section.
	@test "$$(git rev-parse --is-shallow-repository)" = false && git describe --tags --abbrev=0 >/dev/null 2>&1 \
	  || { echo "Fetch the full history and tags first: git fetch --unshallow --tags"; exit 1; }
	@# Prepending keeps the earlier sections, including the hand-written first release.
	uvx git-cliff@2.14.2 --unreleased --tag v$(VERSION) --prepend CHANGELOG.md
	@# git-cliff ends the file with a blank line, which the end-of-file hook removes.
	@printf '%s\n' "$$(cat CHANGELOG.md)" > CHANGELOG.md.tmp && mv CHANGELOG.md.tmp CHANGELOG.md

clean:
	rm -rf .coverage coverage.xml htmlcov dist build .wheelenv *.egg-info pytest-results.xml pytest-integration-results*.xml coverage-integration.xml
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type d -name .pytest_cache -exec rm -rf {} +
	find . -type d -name .ruff_cache -exec rm -rf {} +
	find . -type d -name .ty_cache -exec rm -rf {} +
