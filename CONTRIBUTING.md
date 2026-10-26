# Contributing to OntoGr Agent Memory

## Getting started

You need Python 3.11 or newer and [uv](https://docs.astral.sh/uv/).

```bash
git clone git@github.com:ontogr/ontogr-agent-memory.git
cd ontogr-agent-memory
make sync
uv run pre-commit install
```

`make sync` installs the project dependencies and development tools. Unit tests do not need external services. Integration tests may need services or credentials. Do not put credentials in Git. Use a local `.env` file for settings that read one.

## Make and test a change

1. Fork the repository and create a branch from `main`.
2. Make the change and add or update tests where needed. Unit tests live under `tests/unit/`; integration tests live under `tests/integration/`.
3. While you work, run a focused test by passing a test file or test name to pytest. For example:

   ```bash
   uv run pytest tests/unit/test_package.py -q
   ```

   A focused test gives faster feedback; it does not replace the full unit suite.
4. Before you open a pull request, run the unit suite and non-mutating lint and type checks:

   ```bash
   make test
   make lint-check
   make lint-typing
   ```

   `make test` runs unit tests with coverage. `make lint-check` checks Ruff formatting and lint without changing files. `make lint-typing` runs `ty`. `make lint-all` runs auto-formatting and unsafe lint fixes, so it can change files. Use it only when you want those changes.
5. Open a pull request. Use a [conventional commit](https://www.conventionalcommits.org/en/v1.0.0/) prefix in the title: `fix:`, `feat:`, `build:`, `chore:`, `ci:`, `docs:`, `style:`, `refactor:`, `perf:`, `test:`.

## Tests

- Unit tests must not touch the network. Sockets are disabled by default.
- Tests that need external services go in `tests/integration/`. `tests/conftest.py` applies the `integration`, `enable_socket`, and flaky rerun markers to tests in that directory. Do not add those markers by hand.
- Run integration tests with `make test-integration`. `make test-all` runs both suites.
- Aim for 80-90% coverage on new code. Do not write assertion-free tests to hit a number.

## Style

- Ruff handles formatting and linting.
- Write Google-style docstrings on public functions and classes.
- Add type annotations on public signatures.

## Third-party code

Do not copy code from another project unless its licence allows it and the licence is compatible with Apache-2.0. When you copy code, or adapt it with small changes:

- Keep the original copyright line and licence notice in the file header. Name the source project, the file and the commit that you copied from, and list what you changed.
- Add a comment above a copied function that says where it came from.
- Put the full text of the licence in a file under `licenses/` and list that file in `license-files` in `pyproject.toml`, so the wheel and the source distribution carry it.
- Mention the code in the pull request.

Code that you write from an idea in another project needs no notice. Name the project in a comment when it helps a reader.

## Commands

| Command | Description |
| --- | --- |
| `make sync` | Install all dependencies |
| `make test` | Run unit tests with coverage |
| `make test-integration` | Run integration tests with external services |
| `make test-all` | Run unit and integration tests |
| `make lint-check` | Check formatting and lint without modifying files |
| `make lint-typing` | Type check with `ty` |
| `make lint-all` | Format, apply lint fixes, type check, check typos, and audit GitHub Actions |
| `make security` | Run Bandit and uv audit |
| `make changelog` | Rebuild `CHANGELOG.md` from the git history with git-cliff. It needs the full history and the release tags, and stops if they are missing |
| `make wheel-test` | Build the wheel and import it from a clean environment |
| `make help` | List all targets |

Run `uv run pre-commit install` once to enable the commit hooks.

## Releasing

1. In a pull request, raise `version` in `pyproject.toml` and run `uv lock` to update the package version in `uv.lock`.
2. Update `CHANGELOG.md`. Run `uvx git-cliff@2.14.2 --tag v<VERSION> --output CHANGELOG.md`, replacing `<VERSION>` with the package version. The command generates a release section without creating a Git tag. Check the release date and entries, and keep an empty `Unreleased` section with a comparison link from the new tag to `HEAD`.
3. Run `make wheel-test`. It builds the package and checks imports in a clean environment.
4. After the pull request is merged, create a GitHub release targeting the merged commit with a tag that matches the package version, such as `v0.0.1` for `0.0.1`. Copy the version's changelog entries into the release notes. Creating the tag starts the PyPI publish workflow. Check that the workflow succeeds and that PyPI lists the new version.

## Reporting bugs and requesting features

Open an [issue](https://github.com/ontogr/ontogr-agent-memory/issues). For security vulnerabilities, see [SECURITY.md](SECURITY.md).

## Code of conduct

This project follows the [Contributor Covenant](CODE_OF_CONDUCT.md).
