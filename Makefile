.PHONY: sync test test-integration services-up services-down test-all cov-report cov lint-typing lint-style lint-fmt lint-check lint-typos lint-actions lint-all security-bandit security-audit security build wheel-test image-build image-up image-down check-distribution changelog clean help docs-generate docs-api docs-config docs-install docs-dev docs-build docs-lint-links docs-check-version docs-version docs-test sync-docs-pins

export UV_LOCKED = 1
UV_SYNC_ARGS ?= --no-extra fastembed-gpu

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
	@echo "  make image-build      - Build the local container image"
	@echo "  make image-up         - Start the image and FalkorDB for a Journey run"
	@echo "  make image-down       - Stop the image and remove test volumes"
	@echo "  make check-distribution - Check release and integration pins"
	@echo "  make changelog        - Add a release section to CHANGELOG.md (VERSION=x.y.z)"
	@echo "  make docs-generate    - Regenerate the API pages and the configuration table"
	@echo "  make docs-api         - Regenerate the Python API pages"
	@echo "  make docs-config      - Regenerate the configuration table"
	@echo "  make docs-install     - Install the docs site dependencies with npm ci"
	@echo "  make docs-dev         - Run the docs site locally"
	@echo "  make docs-build       - Build the docs site"
	@echo "  make docs-lint-links  - Reject absolute links between docs pages"
	@echo "  make docs-check-version - Require a docs freeze for the package version"
	@echo "  make docs-version     - Freeze the docs for the package version"
	@echo "  make docs-test        - Run the Python blocks in the docs pages"
	@echo "  make sync-docs-pins   - Sync the docs-api pre-commit pin from uv.lock"
	@echo "  make clean            - Clean build artifacts and cache"

sync:
	uv sync --locked --all-groups --all-extras $(UV_SYNC_ARGS)

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

# SERVICES names the compose services to start: falkordb, neo4j and relational.
# It starts all of them when empty.
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
	tar -tzf dist/*.tar.gz | grep -E '^[^/]+/(docs|node_modules)/' && exit 1 || true
	rm -rf .wheelenv
	uv venv .wheelenv
	uv pip install --python .wheelenv/bin/python dist/*.whl
	cd /tmp && "$(CURDIR)/.wheelenv/bin/python" -c "import ontogr_mem, ontogr_mem.consolidation, ontogr_mem.data_models, ontogr_mem.entity_extraction, ontogr_mem.fact_extraction, ontogr_mem.graphstore, ontogr_mem.ingestion, ontogr_mem.llm, ontogr_mem.observability, ontogr_mem.ontology, ontogr_mem.paths, ontogr_mem.relational_store, ontogr_mem.requirements, ontogr_mem.resolution, ontogr_mem.resilience, ontogr_mem.retrieval, ontogr_mem.server, ontogr_mem.service, ontogr_mem.store_urls, ontogr_mem.temporal, ontogr_mem.vectorstore; from ontogr_mem.ontology import default_ontology; default_ontology(); print(ontogr_mem.__version__)"
	cd /tmp && "$(CURDIR)/.wheelenv/bin/python" -c "from ontogr_mem.mcp_server.skills import SKILLS_ROOT; names = ['memory-write', 'memory-recall']; assert all((SKILLS_ROOT / name / 'SKILL.md').read_text(encoding='utf-8').strip() for name in names), 'packaged skill file is missing or empty'; print('packaged skills ok:', ', '.join(names))"

IMAGE ?= ontogr-agent-memory:local

image-build:
	docker build --tag $(IMAGE) .

image-up:
	ONTOGR_MEM_IMAGE=$(IMAGE) docker compose -f tests/integration/compose.yaml -f tests/integration/compose.image.yaml up -d --wait falkordb mcp

image-down:
	ONTOGR_MEM_IMAGE=$(IMAGE) docker compose -f tests/integration/compose.yaml -f tests/integration/compose.image.yaml down -v

check-distribution:
	uv run --no-project python scripts/check_distribution.py

changelog:
	@test -n "$(VERSION)" || { echo "Usage: make changelog VERSION=x.y.z"; exit 1; }
	@# git-cliff groups commits by tag, so a shallow or tagless clone gives a wrong section.
	@test "$$(git rev-parse --is-shallow-repository)" = false && git describe --tags --abbrev=0 >/dev/null 2>&1 \
	  || { echo "Fetch the full history and tags first: git fetch --unshallow --tags"; exit 1; }
	@# Prepending keeps the earlier sections, including the hand-written first release.
	uvx git-cliff@2.14.2 --unreleased --tag v$(VERSION) --prepend CHANGELOG.md
	@# git-cliff ends the file with a blank line, which the end-of-file hook removes.
	@printf '%s\n' "$$(cat CHANGELOG.md)" > CHANGELOG.md.tmp && mv CHANGELOG.md.tmp CHANGELOG.md

DOCS_GRIFFE2MD ?= uv run --group docs griffe2md
DOCS_PYTHON ?= uv run python
DOCS_API_PKGS ?= ontogr_mem.client ontogr_mem.errors ontogr_mem.service.contracts

docs-generate: docs-api docs-config

docs-api:
	mkdir -p docs/docs/api
	find docs/docs/api -mindepth 1 -type f ! -name index.md -delete
	find docs/docs/api -mindepth 1 -type d -empty -delete
	i=2; for p in $(DOCS_API_PKGS); do \
	  { printf '%s\n' '---' "title: $$p" "sidebar_position: $$i" '---' ''; \
	    $(DOCS_GRIFFE2MD) $$p -f; } > docs/docs/api/$$p.md.tmp \
	    && mv docs/docs/api/$$p.md.tmp docs/docs/api/$$p.md || exit 1; \
	  i=$$((i+1)); done
	$(DOCS_PYTHON) .github/scripts/docs_api_links.py docs/docs/api

docs-config:
	$(DOCS_PYTHON) .github/scripts/docs_config_table.py

docs-install:
	cd docs && npm ci

docs-dev: docs-api
	cd docs && npm start

# Absolute links like /guides/x always open the latest version. Relative file links stay in the version.
docs-lint-links:
	@grep -rEn --include='*.md' --include='*.mdx' '(\]\(|href=")/[A-Za-z0-9]' docs/docs --exclude-dir=api; \
	status=$$?; \
	if [ $$status -eq 0 ]; then echo "Link to docs pages with relative file paths such as ../concepts/how-data-flows.mdx."; exit 1; fi; \
	if [ $$status -ne 1 ]; then exit $$status; fi

# The version of the package. The docs version of a release has the same name.
DOCS_VERSION = $(shell $(DOCS_PYTHON) -c "import tomllib; print(tomllib.load(open('pyproject.toml', 'rb'))['project']['version'])")

# A pull request that raises the version in pyproject.toml must also freeze its docs.
docs-check-version:
	@grep -qF '"$(DOCS_VERSION)"' docs/versions.json \
	  || { echo "docs/versions.json has no $(DOCS_VERSION). Run 'make docs-version' in the pull request that bumps the version."; exit 1; }

# Freeze the docs of the version in pyproject.toml before the release is tagged.
docs-version: docs-api
	cd docs && npx docusaurus docs:version $(DOCS_VERSION)

docs-build: docs-generate docs-lint-links
	cd docs && { npm run build > build.log 2>&1; status=$$?; cat build.log; exit $$status; }
	@# Docusaurus reports many problems as warnings and still exits with success.
	@sed -E 's/\x1B\[[0-9;]*m//g' docs/build.log | grep -q '^\[SUCCESS\]' \
	  && ! sed -E 's/\x1B\[[0-9;]*m//g' docs/build.log | grep -qE '^\[(WARNING|ERROR)\]' \
	  || { echo "The docs build did not finish cleanly. Read docs/build.log."; exit 1; }

# The docs tests need a FalkorDB server on loopback. The generated API pages hold
# docstring examples that are not self-contained, so they never run.
docs-test:
	uv run --group docs pytest --markdown-docs docs/docs/get-started docs/docs/guides \
		-o addopts="" -p no:cacheprovider --forked -q --junitxml=docs-snippets-results.xml

sync-docs-pins:
	$(DOCS_PYTHON) .github/scripts/update_precommit_docs_pins.py

clean:
	rm -rf .coverage coverage.xml htmlcov dist build .wheelenv *.egg-info pytest-results.xml pytest-integration-results*.xml coverage-integration.xml docs/build docs/.docusaurus docs/build.log
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type d -name .pytest_cache -exec rm -rf {} +
	find . -type d -name .ruff_cache -exec rm -rf {} +
	find . -type d -name .ty_cache -exec rm -rf {} +
