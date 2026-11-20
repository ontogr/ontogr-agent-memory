---
title: ontogr_mem.service.contracts.ForgetMode
sidebar_label: ForgetMode
---

# `ontogr_mem.service.contracts.ForgetMode` \{#ontogr_mem-service-contracts-ForgetMode}

Bases: <code>StrEnum</code>

How much forget_memory removes.

INVALIDATE closes the fact now and keeps its history. DELETE removes the
observations and the stored episode. MEMORY_ONLY removes the derived facts
and keeps the episode record. ERASE deletes and then scans the graph and
indexes for residue, failing when it finds any.

**Attributes:**

- [**DELETE**](#ontogr_mem-service-contracts-ForgetMode-DELETE) –
- [**ERASE**](#ontogr_mem-service-contracts-ForgetMode-ERASE) –
- [**INVALIDATE**](#ontogr_mem-service-contracts-ForgetMode-INVALIDATE) –
- [**MEMORY_ONLY**](#ontogr_mem-service-contracts-ForgetMode-MEMORY_ONLY) –

## `DELETE` \{#ontogr_mem-service-contracts-ForgetMode-DELETE}

```python
DELETE = 'delete'
```

## `ERASE` \{#ontogr_mem-service-contracts-ForgetMode-ERASE}

```python
ERASE = 'erase'
```

## `INVALIDATE` \{#ontogr_mem-service-contracts-ForgetMode-INVALIDATE}

```python
INVALIDATE = 'invalidate'
```

## `MEMORY_ONLY` \{#ontogr_mem-service-contracts-ForgetMode-MEMORY_ONLY}

```python
MEMORY_ONLY = 'memory_only'
```
