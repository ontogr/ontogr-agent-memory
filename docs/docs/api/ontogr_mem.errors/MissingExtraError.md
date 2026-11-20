---
title: ontogr_mem.errors.MissingExtraError
sidebar_label: MissingExtraError
---

# `ontogr_mem.errors.MissingExtraError` \{#ontogr_mem-errors-MissingExtraError}

```python
MissingExtraError(extras:Sequence[str], *, features:Sequence[str] = (), needed:Sequence[str] = (), subcommand:str = '') -> None
```

Bases: <code>[AgentMemoryError](AgentMemoryError.md)</code>

One or more optional extras are not installed.

**Attributes:**

- [**extras**](#ontogr_mem-errors-MissingExtraError-extras) – Names of the extras that are not installed.

**Parameters:**

- **extras** (<code>Sequence\[str\]</code>) – Extras that are not installed.
- **features** (<code>Sequence\[str\]</code>) – Features that need the missing extras.
- **needed** (<code>Sequence\[str\]</code>) – Every extra the command needs. Defaults to `extras`.
- **subcommand** (<code>str</code>) – Command placed after the uvx invocation, such as mcp.

## `extras` \{#ontogr_mem-errors-MissingExtraError-extras}

```python
extras = tuple(extras)
```
