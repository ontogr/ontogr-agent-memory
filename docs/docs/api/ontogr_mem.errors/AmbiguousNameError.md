---
title: ontogr_mem.errors.AmbiguousNameError
sidebar_label: AmbiguousNameError
---

# `ontogr_mem.errors.AmbiguousNameError` \{#ontogr_mem-errors-AmbiguousNameError}

```python
AmbiguousNameError(ids:Sequence[str]) -> None
```

Bases: <code>[AgentMemoryError](AgentMemoryError.md)</code>

A name resolves to more than one entity.

**Attributes:**

- [**code**](#ontogr_mem-errors-AmbiguousNameError-code) – Stable error code for this condition.
- [**ids**](#ontogr_mem-errors-AmbiguousNameError-ids) – Distinct matching entity ids in sorted order.

## `code` \{#ontogr_mem-errors-AmbiguousNameError-code}

```python
code = 'ambiguous_name'
```

## `ids` \{#ontogr_mem-errors-AmbiguousNameError-ids}

```python
ids = tuple(sorted(set(ids)))
```
