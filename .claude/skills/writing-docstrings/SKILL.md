---
name: writing-docstrings
description: Google-style docstring and comment rules. Use when adding or changing docstrings or comments in Python code or test modules.
---

# Google-style docstrings and comments

Use the Google Python Style Guide for docstring structure, API coverage, and formatting. Use the examples below as the preferred local formatting examples. Use [writing-plain](../writing-plain/SKILL.md) for prose style, including its exceptions for concise summaries and field entries.

Diátaxis is primarily for docs pages. For docstrings, apply only the lightweight
version below: prefer precise reference-style API facts, with brief explanation
or examples only when they help the caller use the API correctly.

## Required docstring rules

- Always use triple double quotes (`"""`) for docstrings.
- Start each docstring with a one-line summary ending in `.`, `?`, or `!`.
  Keep the summary on one physical line no longer than 88 characters.
- Write docstrings as deliberate API documentation a senior engineer would keep
  in the codebase, not as agent notes about what was found, tried, fixed, or
  verified.
- If more detail follows the summary, add one blank line before the longer
  description.
- Function and method docstrings are required for public APIs, nontrivial
  functions, and functions with non-obvious logic. Small private helpers may use
  a one-line docstring when their signature and name are clear.
- Test module docstrings are not optional. Add them so that they provide useful
  context, such as what is being tested, unusual setup, dependencies, or how to run/update fixtures.
  Do not add boilerplate docstrings such as `"""Tests for foo."""`.
- Document call semantics and caller-visible side effects, not implementation
  details. Put implementation notes in comments near the code.
- Explain purpose, behavior, edge cases, and constraints. Do not restate the
  names as their descriptions. Repeat API or parameter names when needed to remove ambiguity.
- Keep docstrings synchronized with code changes. Update docstrings whenever a
  code change makes existing documentation stale.
- Keep each physical line within 88 characters. Wrap longer descriptions naturally without dropping necessary facts. Keep the summary on one line as required above.
- Use either descriptive style (`"""Fetches rows."""`) or imperative style
  (`"""Fetch rows."""`) consistently within a file.

## Inline comments

- Inline comments must explain why the code exists, not what the code does.
- Add comments only when they explain non-obvious business logic,
  domain-specific constraints, safety concerns, algorithms, performance
  tradeoffs, library workarounds, or temporary hacks with ticket references.
  Prefer clearer code over comments for ordinary control flow.
- Do not leave AI-slop comments: no progress notes, literal findings,
  self-congratulation, implementation transcripts, or statements that something
  "works". Comments should read like concise guidance from a human maintainer.
- Do not place comments at the end of code lines. If a comment describes a
  specific line or block, put the comment on the preceding line aligned with
  the code it describes.
- Do not add comments for self-explanatory code, trivial operations, changelog
  notes, or closing-brace/block markers.
- Do not use visual separator comments, such as `# ------- text -------` or
  `# ------------`.
- Do not write comments that merely restate code, such as `# increment x` above
  `x += 1`.
- Write comments in clear, professional prose with normal grammar and
  punctuation. Avoid dramatic language, all-caps emphasis, and boilerplate.

## Diátaxis influence on docstrings

- Default to reference-style precision: what the object is, what the caller can
  rely on, parameters, return values, raised interface exceptions, side effects,
  constraints, and edge cases.
- Add brief explanation only when context is necessary to use the API correctly,
  such as a non-obvious invariant, algorithm choice, domain-specific assumption,
  or safety constraint.
- Add short examples only when they materially clarify common usage. Keep them
  canonical, not exhaustive.
- Do not turn docstrings into tutorials or how-to guides. If a learner needs a
  walkthrough, write or link to a docs page instead.

## Sections

- Use section headers ending with a colon, such as `Args:`, `Returns:`,
  `Yields:`, `Raises:`, `Attributes:`, `Examples:`, and `Note:`.
- Use `Yields:` instead of `Returns:` for generators.
- Omit `Returns:` when the function returns only `None`, or when a one-line
  summary that starts with `Return`, `Returns`, `Yield`, or `Yields` fully
  describes the return value.
- In `Args:`, list every parameter by name. Do not repeat obvious types already
  provided by the function signature.
- List variable arguments as `*args` and `**kwargs`.
- Do not include `self` or `cls` in `Args:`.
- In `Raises:`, list only exceptions that are relevant to the public interface.
  Do not document exceptions that occur only when callers violate the documented
  API contract.
- Use a hanging indent of either two or four spaces inside sections. Be
  consistent within a file. This project prefers the four-space style shown
  below.

## Classes, exceptions, and properties

- Classes should have a docstring describing what an instance represents.
- Exception class docstrings should describe what the exception represents, not
  start with boilerplate such as `Raised when...`.
- Pydantic models and settings classes should use class docstrings with an
  `Attributes:` section for public fields. For environment-backed settings,
  include the environment variable name in the field description, such as
  `Env: MEMORY_TEMPORAL_BATCH_SIZE`.
- Public attributes, excluding properties, should be documented in an
  `Attributes:` section or inline near the attribute declaration. Do not mix the
  two styles within the same class/module.
- Document `__init__` parameters either in the class docstring or in the
  `__init__` docstring. Do not duplicate both forms.
- Document properties in the getter. If the setter has notable behavior, mention
  it in the getter docstring.
- An overridden method may omit a docstring when decorated with `@override` and
  the base method contract is unchanged. Add a docstring when the override
  changes or refines caller-visible behavior.

## Types and annotations

- PEP 484 type annotations are required for typed public APIs.
- When parameters, attributes, and return values are annotated, do not repeat
  obvious types in the docstring.
- Include type information in docstrings only when annotations are absent or
  insufficient to explain accepted values, units, shapes, or other semantics.

## Project examples

Use these local patterns when documenting common ontogr_mem component types.

### Frozen dataclass

```python
@dataclass(frozen=True, slots=True)
class TypedFact:
    """A fact with its relation mapped to a stable type id.

    Attributes:
        fact: The extracted fact. Its dense `relation_type_id` is only meaningful
            with the projection it came from.
        relation_type: Stable relation type id, or None for a fact without a
            usable relation.
        catalog_version: Version of the projection whose dense ids the fact used.
    """
```

### Settings class

```python
class TemporalSettings(BaseSettings):
    """Retry and batch limits for the temporal engine.

    Environment variables use the prefix ``MEMORY_TEMPORAL_``.

    Attributes:
        max_retries: Retries for a slot that lost a revision race. Env:
            ``MEMORY_TEMPORAL_MAX_RETRIES``.
        batch_size: Observations written in one store call. Env:
            ``MEMORY_TEMPORAL_BATCH_SIZE``.
        episode_ttl_s: Seconds an episode may stay ``PENDING`` before the audit
            reports it. Env: ``MEMORY_TEMPORAL_EPISODE_TTL_S``.
    """
```
