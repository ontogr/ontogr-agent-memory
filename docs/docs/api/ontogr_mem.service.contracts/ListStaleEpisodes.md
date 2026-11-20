---
title: ontogr_mem.service.contracts.ListStaleEpisodes
sidebar_label: ListStaleEpisodes
---

# `ontogr_mem.service.contracts.ListStaleEpisodes` \{#ontogr_mem-service-contracts-ListStaleEpisodes}

Bases: <code>[ScopedRequest](ScopedRequest.md)</code>

Page through episodes extracted under an older ontology version.

**Attributes:**

- [**cursor**](#ontogr_mem-service-contracts-ListStaleEpisodes-cursor) (<code>str | None</code>) –
- [**limit**](#ontogr_mem-service-contracts-ListStaleEpisodes-limit) (<code>int</code>) –
- [**model_config**](#ontogr_mem-service-contracts-ListStaleEpisodes-model_config) –
- [**tenant**](#ontogr_mem-service-contracts-ListStaleEpisodes-tenant) (<code>[TenantId](TenantId.md) | None</code>) –

## `cursor` \{#ontogr_mem-service-contracts-ListStaleEpisodes-cursor}

```python
cursor: str | None = None
```

## `limit` \{#ontogr_mem-service-contracts-ListStaleEpisodes-limit}

```python
limit: int = Field(default=20, ge=1, le=MAX_RESULTS)
```

## `model_config` \{#ontogr_mem-service-contracts-ListStaleEpisodes-model_config}

```python
model_config = ConfigDict(extra='forbid')
```

## `tenant` \{#ontogr_mem-service-contracts-ListStaleEpisodes-tenant}

```python
tenant: TenantId | None = None
```
