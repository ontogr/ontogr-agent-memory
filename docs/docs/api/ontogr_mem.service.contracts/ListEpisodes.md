---
title: ontogr_mem.service.contracts.ListEpisodes
sidebar_label: ListEpisodes
---

# `ontogr_mem.service.contracts.ListEpisodes` \{#ontogr_mem-service-contracts-ListEpisodes}

Bases: <code>[ScopedRequest](ScopedRequest.md)</code>

Page through stored episodes, newest first.

**Attributes:**

- [**cursor**](#ontogr_mem-service-contracts-ListEpisodes-cursor) (<code>str | None</code>) –
- [**limit**](#ontogr_mem-service-contracts-ListEpisodes-limit) (<code>int</code>) –
- [**model_config**](#ontogr_mem-service-contracts-ListEpisodes-model_config) –
- [**session_id**](#ontogr_mem-service-contracts-ListEpisodes-session_id) (<code>str | None</code>) –
- [**tenant**](#ontogr_mem-service-contracts-ListEpisodes-tenant) (<code>[TenantId](TenantId.md) | None</code>) –

## `cursor` \{#ontogr_mem-service-contracts-ListEpisodes-cursor}

```python
cursor: str | None = None
```

## `limit` \{#ontogr_mem-service-contracts-ListEpisodes-limit}

```python
limit: int = Field(default=20, ge=1, le=MAX_RESULTS)
```

## `model_config` \{#ontogr_mem-service-contracts-ListEpisodes-model_config}

```python
model_config = ConfigDict(extra='forbid')
```

## `session_id` \{#ontogr_mem-service-contracts-ListEpisodes-session_id}

```python
session_id: str | None = None
```

## `tenant` \{#ontogr_mem-service-contracts-ListEpisodes-tenant}

```python
tenant: TenantId | None = None
```
