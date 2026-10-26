# AGENTS.md

## Project

**`ontogr_mem`** — durable, traceable memory for AI agents.
Published to PyPI as `ontogr-mem`; imported as `ontogr_mem`.
The repo is hosted at `ontogr/ontogr-agent-memory`.

**Layout**: flat. The package is `ontogr_mem/` at the repo root and everything outside
`ontogr_mem/` (tests, examples, thoughts) is not part of the distribution.

- Python package: `ontogr_mem`.
- Public repo: `https://github.com/ontogr/ontogr-agent-memory`.
- License: Apache-2.0.
- Python: `>=3.11`.
- Package manager and command runner: `uv`.

## Pre-1.0 policy — breaking changes expected

Project is pre-1.0 with no stability promise. Breaking changes are allowed
and recommended if they produce better results. This section overrides any
stability or compatibility rule in this file or in a skill.

Default to a hard cut. Do not add backward-compat aliases, deprecated shims,
fallback parsing, migration bridges, or tests for the previous API unless the
user explicitly asks. Remove the old surface and write tests only for the
current behavior, as if the old API never existed.

Do not add backwards-compatibility shims, deprecation comments, or feature
flags for changes that have not been released yet. Just edit the code.

## Skills

Task-specific rules live in project skills under `.claude/skills/`. Use the
matching skill before you start that kind of work. An agent without skill
support can open `.claude/skills/<name>/SKILL.md` directly.

| When you are doing... | Use |
| --- | --- |
| Drafting or revising prose, including replies, instruction files, docs, comments, docstrings, commits, and PRs | `writing-plain` plus the relevant task skill |
| Adding or changing tests | `writing-tests` |
| Adding or updating docstrings or comments | `writing-docstrings` |
| Fixing type-check failures from `ty` | `fixing-ty-errors` |

For setup, contribution, PR, or broader project commands, read @CONTRIBUTING.md.
The Pre-1.0 policy above overrides every skill. Where this file and a skill
overlap, the skill is authoritative. More specific task instructions control
implementation details.

For prose style, `writing-plain` is authoritative. Its docwriter rules take precedence over its STE guidance. Task skills govern technical content, structure, and required formatting, not competing prose styles. Preserve technical meaning, uncertainty, and requirement strength, and apply the shared skill's scope and format exceptions.

## Repository structure

This is a flat-layout Python repository. Do not introduce a `src/` directory.

```text
ontogr_mem/           # the package; subpackages are added as components land

tests/
  unit/                 # unit tests, no network
  integration/          # integration tests with real external services
thoughts/               # shared plans, decisions and research notes, do not edit unless asked
CONTEXT.md              # project vocabulary; keep terms consistent with it
```

## Critical rules

- Do not add any documentation or comments that would not reside in a production codebase. Comments should be developer/user friendly not intermediate findings when performing a task.
- Do not include comments referring to internal plans, notes, or comments written by agents while performing a task.
- Keep the flat layout: `ontogr_mem/` lives at the repository root. Do not add
  `src/`.
- Use `uv run` for Python commands and Makefile targets for standard checks.
- Public component APIs are async unless an existing interface requires sync.
- Preserve public APIs within a release; the Pre-1.0 policy above governs
  breaking changes.
- Inject dependencies through constructors or function parameters; do not add a
  dependency-injection framework.
- Match existing patterns before introducing new abstractions.
- Prefer small, surgical edits. Do not refactor adjacent code opportunistically.
- Verify changed behavior with an end-to-end or integration test, following the
  Testing requirements below.
- Do not commit secrets, `.env`, API keys, tokens, or credentials.
- Do not put patient, customer, partner, or private organization names in code,
  docs, issues, PRs, commit messages, or tests unless they are already public
  fixture data approved for that use.
- Do not commit unless the user explicitly asks.

## Development commands

Use `uv` for environment and command execution. Standard commands are exposed as
Makefile targets:

```bash
make sync                  # install all dependencies with uv sync
make test                  # run unit tests with coverage
make test-integration      # run integration tests requiring external services
make lint-fmt              # format and auto-fix with Ruff
make lint-check            # check formatting and lint, no modifications
make lint-typing           # type check with ty
make lint-all              # format + lint + type check + typos; applies unsafe fixes
make clean                 # remove build artifacts and caches
```

Run focused tests while iterating:

```bash
uv run pytest tests/unit/test_package.py -v
```

Environment setup details are in @CONTRIBUTING.md. If a tool such as `uv`,
`pytest`, `ruff`, or `ty` is missing, check for the repo `.venv`. If present,
activate it and retry. If no environment exists, ask before installing or using
alternate tooling.

## Coding standards

- Use DDD and Clean Architecture principles where they fit the component.
- Default to functions and modules. Use classes only when there is genuine
  stateful behavior, shared `self` state, or a data container with behavior.
- Do not use classes only for namespacing, single-method wrappers, or grouping
  pure functions.
- Do not add abstractions for single-use code or hypothetical future needs.
  If you catch yourself adding "for future extensibility", stop. Ship the
  smallest code that satisfies this task; add configurability only when a
  second caller exists in this repo.
- Use `snake_case` for functions and variables, `PascalCase` for classes, and
  `test_*.py` for tests.
- Prefer descriptive names. Avoid abbreviations such as `ver` when `version` is
  clearer, or `rp` when `requires_python` is clearer.
- Prefer top-level imports. Use local imports only when needed to avoid cycles,
  optional dependency import costs, or test-specific patching patterns already
  used nearby.
- Add type hints to new and modified public APIs.
- Prefer `TypedDict` over broad `dict`/`Mapping` when documenting structured
  dictionaries in code.
- Keep state explicit. Do not dynamically add/read fields with `setattr` and
  `getattr` when normal attributes or models would be clearer.
- Implement relevant special methods such as `__repr__` or `__str__` when they
  improve debugging or match nearby domain models.
- Keep author attribution comments or module-docstring attribution if present.
- Prefer composition over inheritance for new code unless an existing framework
  or local interface expects inheritance.
- Use early returns to avoid deep nesting.
- Keep public APIs minimal and well-defined; do not expose internals.
- Prefix internal helpers with `_` when they carry no stability guarantee.
- Define `__all__` only in package `__init__.py` files that intentionally
  re-export public symbols. Do not put `__all__` in implementation modules.
- Do not access private members (`_`-prefixed attributes or methods) of other
  classes. Use public APIs or add an explicit interface when needed.
- Place module-level helper functions near the top of a file after imports.
  Within classes, place private/helper methods after public methods when that
  matches the surrounding file.

### Public API stability

The Pre-1.0 policy above controls. Within a release, public APIs are
contracts. Before changing a public function, class, constructor,
or exported symbol:

1. Check whether it is exported from a package `__init__.py`.
2. Search existing tests, examples, docs, and call sites.
3. Preserve function names, parameter names, positional argument behavior, and
   return shape when possible.
4. Add new parameters as keyword-only when practical:
   `*, new_parameter: str = "default"`.
5. Warn the user if a requested change is breaking or may require migration.

Ask: would this break code that used the package last week?

## Architecture and component boundaries

- Share domain models across features; do not define feature-local domain models
  when a shared model is appropriate.
- Keep memory logic storage-agnostic unless it is explicitly in a storage
  adapter package.
- Validate any interpolated query identifiers before formatting a query string.
- Settings classes subclass `pydantic_settings.BaseSettings`, load from the
  repo-root `.env`, and use clear environment prefixes such as `MEMORY_DB_`.
- Wrap databases, external services, network I/O, and message queues when a
  project-native boundary is useful. Do not wrap stable utility or ML libraries
  merely to hide imports.
- Use monkeypatching/mocking for OS calls, time, third-party SDKs, and test-only
  seams. Add explicit injection seams when tests would otherwise patch project
  internals.
- Prefer dependency injection over monkeypatching project class attributes in
  tests. Monkeypatching third-party SDK boundaries is acceptable when it is the
  narrowest reliable test seam.

## Adding project components

Follow nearby components first.

### Memory components

- Keep component logic pure and storage-agnostic when possible.
- Follow the relevant base class and neighboring implementation patterns.
- Cover the behavior with an end-to-end or integration test.

## Testing requirements

Use the `writing-tests` skill before adding or changing tests.

- Never write unit tests after you write code.
- Highly prefer E2E or integration tests as the sole testing mechanism. Use them to verify complex features work. At the end of those tests, produce a verifiable and repeatable artifact.
- If you must test a system in isolation, first write down all the ways it could fail, then write the code.
- A test that breaks under a behavior-preserving refactor is asserting implementation, not behavior. Do not add it.
- Never delete or weaken a failing test to make the suite pass. Fix the code, or ask.
- Do not add tests that restate the code, assert mock call shapes, grep source or docs text, compute the expected value with the code under test, or only keep a test-only export or seam alive.
- Use `pytest` and `pytest-asyncio`; async tests do not need
  `@pytest.mark.asyncio`.
- Tests mirror the `ontogr_mem/` package under `tests/unit/` and
  `tests/integration/`.
- Use one `Test<ClassName>` class per `test_<class>.py` file when practical.
- Use descriptive behavior names such as `test_returns_empty_when_no_results`.
- Use Arrange-Act-Assert structure.
- Test public interfaces and documented behavior, not private implementation
  details. Avoid accessing private members in tests unless no public seam exists.
- For tests over multiple inputs, use `@pytest.mark.parametrize`.
- Always read and copy the style of similar nearby tests before adding fixtures
  or helpers.
- For bug fixes, extend the E2E or integration test that covers the behavior so
  it fails without the fix. Add a unit test only when the rules above allow it.
- One focused test that fails without the fix is better than many shallow tests
  that do not prove behavior.
- Do not write assertion-free tests just to increase coverage.
- Target 80-90% coverage for core/domain logic; do not chase 100% coverage.
- Do not delete, skip, or weaken a failing test to hide the failure. Use `skip`
  or `xfail` only for a known, documented reason the user has accepted.

### Unit tests

Unit tests are the exception. Write one only for isolated logic with real branches that an E2E or integration test cannot reach, and only under the rules above.

- Location: `tests/unit/...`.
- No network access. Sockets are disabled by default.
- Mock databases, LLM APIs, and other external services at the driver/API
  boundary.
- If a unit test raises `SocketBlockedError`, fix the mock; do not enable
  network access.
- Do not add manual `enable_socket` markers to unit tests.

### Integration tests

- Location: `tests/integration/...`.
- Use real external services only when integration behavior is under test.
- `integration`, `enable_socket`, and flaky rerun markers are auto-applied by
  `tests/conftest.py`; do not add them manually.
- Integration tests can rely on real environment variables or `.env`; do not
  commit secrets.

## Type checking

This project uses `ty`. Use the `fixing-ty-errors` skill when fixing type
errors.

- Run focused checks with `uv run ty check <target>` while iterating.
- Run `make lint-typing` for the repository typing target.
- Use narrowing (`isinstance`, `if x is None: raise`, `hasattr`) before casts.
- Never use `assert` for type narrowing.
- Use `cast()` only after structural validation when the checker cannot infer
  the type.
- Use `# type: ignore[code]` only for genuine third-party stub defects, and never
  as a first resort.
- Do not add helper abstractions solely to satisfy the type checker.

## Writing style

Use the `writing-plain` skill before drafting or revising prose. Write literal explanations with familiar words and precise technical terms. Keep necessary facts and qualifications. The shared skill combines plain-writing rules with compatible STE guidance; it does not claim ASD-STE100 compliance.

## Docstrings and comments

Use the `writing-docstrings` skill before adding or changing docstrings or
comments.

Google style governs docstring structure and formatting; `writing-plain` governs prose.

- Use Google-style docstrings with triple double quotes.
- Start every docstring with a one-line summary ending in punctuation.
- Keep docstring summary lines and comment lines within the 88-character limit;
  rewrite concisely rather than awkwardly wrapping.
- Public APIs, nontrivial functions, classes, exceptions, properties, settings,
  and test modules need useful docstrings.
- Docstrings document caller-visible behavior, edge cases, constraints, return
  values, raised interface exceptions, and side effects.
- Types belong in annotations, not repeated in docstrings unless annotations are
  insufficient to describe units, shapes, or accepted values.
- Inline comments explain why, not what. Do not add comments that restate code.
- Add comments only when they explain non-obvious business logic,
  domain-specific constraints, safety concerns, algorithms, performance
  tradeoffs, or library workarounds. Prefer clearer code over comments for
  ordinary control flow.
- Do not add AI-progress notes, implementation transcripts, or comments saying
  something now works.
- Do not put comments at the end of code lines; place them above the relevant
  line or block.
- Newly added code comments must be ASCII only.

## Tooling, dependencies, generated files, and security

- Use `uv` for package management, virtual environments, and lockfiles.
- Keep dependency specs loose in `pyproject.toml`; rely on `uv.lock` for
  reproducibility.
- Add upper bounds only for known-breaking dependencies.
- Do not add dependencies unless strictly required; justify any new dependency.
- Run `deptry` or an equivalent check before removing a dependency, and account
  for dynamic imports and plugin entry points.
- Use Ruff for linting and formatting. Line length is 88.
- Do not change Ruff, typing, pytest, or CI configuration unless explicitly
  asked.
- In CI or installation scripts, never pipe remote scripts into a shell. Download
  artifacts to files, pin external tools to explicit versions, and verify
  checksums before installation.
- Use configured ignores for generated or unowned code instead of manual edits.
- Treat Bandit findings as review items unless CI makes them blocking.
- Never commit `.env`, API keys, tokens, credentials, or local secrets.

## Workflow expectations

- If the request is ambiguous and multiple valid interpretations materially
  change the solution, ask one narrow clarifying question.
- Ask before changing public API shape, persistence schema,
  dependencies, or config keys. Otherwise make a
  reasonable assumption and proceed.
- State assumptions when they affect the implementation. Surface tradeoffs and
  push back when a simpler or safer approach exists.
- Before changing code, inspect enough nearby code and tests to follow existing
  patterns.
- Do not assume existing code is always the right pattern. Follow good local
  patterns, but call out code smells or risky patterns when they affect the
  requested change.
- Keep every changed line traceable to the requested behavior.
- Do not add unrelated features, configuration, abstractions, or defensive code.
- If a solution starts becoming much larger than the problem requires, stop and
  simplify before continuing.
- Remove imports, variables, functions, or docs made obsolete by your change.
  Do not remove pre-existing dead code unless asked.
- If you notice unrelated issues, mention them separately instead of fixing them
  opportunistically.
- Prefer the non-mutating targets `make test`, `make lint-check`, and
  `make lint-typing`. Run `make lint-all` only for an explicit autofix run.
- Stop when the narrowest useful check passes, or report the concrete blocker.
- When asked to document or explain the codebase, describe what exists. Do not
  propose improvements unless asked.
- Check `thoughts/shared/plans/` before starting non-trivial implementation work.
- Define the narrowest useful verification check before running broad commands.

## Task completion guidelines

Use judgment based on scope, but default to these artifacts.

### Bug fixes

1. Reproduce the bug in an E2E or integration test that fails without the fix.
2. Implement the smallest correct fix.
3. Run the focused test.
4. Run lint/type checks when the touched area or risk justifies them.

### New features

1. Follow existing interfaces and component patterns.
2. Add an E2E or integration test that exercises the feature and leaves a
   verifiable, repeatable artifact.
3. Update docstrings and user-facing docs if behavior is public.
4. Run focused tests plus the narrowest broader check that proves integration.

### Refactors/internal changes

- Preserve public behavior unless the change itself intends a break under the
  Pre-1.0 policy; update tests to the current behavior and drop tests for
  removed APIs.
- Run existing tests that cover the refactored behavior.

### Documentation-only changes

- Follow `writing-plain` for prose and instruction files.
- Be concrete and concise. Include examples, command invocations, data formats,
  or short flows when they make behavior easier to understand.
- Explain the why behind non-obvious design decisions and tradeoffs.
- Use diagrams or short step-by-step flows for complex interactions.
- Verify links, commands, and referenced paths where practical.
- Do not change code unless required to make the docs true.

## Git and pull requests

- Do not commit unless explicitly asked.
- Do not revert, overwrite, or clean up user changes unless explicitly asked.
- Commit messages and PR titles should use a conventional commit prefix such as
  `fix:`, `feat:`, `docs:`, `refactor:`, `test:`, `build:`, `ci:`, `chore:`,
  `style:`, or `perf:`.
- Commit messages should be a single short imperative sentence with no trailing
  period, clear and informative.
- PR descriptions should follow @.github/pull_request_template.md.
- Do not add AI-agent attribution such as `Co-Authored-By: Claude`, `Generated
  with Claude Code`, or similar text to commits, PRs, comments, or issues.
- When writing public-facing PR descriptions, issues, comments, or commit
  messages, avoid emojis, hype, AI-sounding phrasing, and unnecessary bullet
  lists. Prefer concise human prose.

## Self-review checklist

Before handing off non-trivial changes, check:

- Correctness: the code does what the request says and handles relevant edge
  cases such as empty inputs, `None`, and boundary values.
- Public contract: breaking changes follow the Pre-1.0 policy (hard cut, old
  surface removed, no compat shims); no leftover aliases or old-API tests
  remain.
- Tests: changed behavior has an E2E or integration test that fails when the
  behavior is broken and survives a behavior-preserving refactor.
- Style: new or modified public functions are typed, docstrings are updated when
  needed, and Ruff formatting/linting expectations are respected.
- Design: no unnecessary abstractions, private-member access, god objects,
  monster files, or speculative configurability were introduced.
- Safety: no secrets, unsafe deserialization, injection risks, resource leaks,
  or unnecessary hot-path allocations were introduced.
- Docs: user-facing behavior changes are reflected in relevant docs or
  docstrings.

## Do not do these things

- Do not introduce a `src/` layout.
- Do not use `pip`, `poetry`, or `conda` for project dependency operations.
- Do not use mypy; use `ty`.
- Do not add manual integration/socket markers that `tests/conftest.py` already
  auto-applies.
- Do not enable sockets in unit tests to avoid mocking.
- Do not commit secrets or `.env` files.
- Do not add new dependencies casually or pin upper bounds without a known
  compatibility reason.
- Do not pipe remote install scripts into a shell in CI or documentation.
- Do not refactor, reformat, or clean adjacent code just because it is nearby.
- Do not change public APIs without explicit approval unless the Pre-1.0 policy
  above allows the break; never keep the old API beside the new one, and update
  docs and tests to the current behavior.
- Do not change CI, Ruff, pytest, or typing config unless explicitly asked.
- Do not add AI attribution to commits, branches, PR descriptions, comments, or
  issues.
