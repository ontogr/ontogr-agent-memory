---
title: ontogr_mem.service.contracts.StaleEpisodePage
sidebar_label: StaleEpisodePage
---

# `ontogr_mem.service.contracts.StaleEpisodePage` \{#ontogr_mem-service-contracts-StaleEpisodePage}

Bases: <code>[ContractModel](ContractModel.md)</code>

One page of stale episodes and the current ontology version.

**Attributes:**

- [**current_version**](#ontogr_mem-service-contracts-StaleEpisodePage-current_version) (<code>int</code>) –
- [**episodes**](#ontogr_mem-service-contracts-StaleEpisodePage-episodes) (<code>list\[[StaleEpisodeOut](StaleEpisodeOut.md)\]</code>) –
- [**model_config**](#ontogr_mem-service-contracts-StaleEpisodePage-model_config) –
- [**next_cursor**](#ontogr_mem-service-contracts-StaleEpisodePage-next_cursor) (<code>str | None</code>) –
- [**tenant**](#ontogr_mem-service-contracts-StaleEpisodePage-tenant) (<code>[TenantId](TenantId.md)</code>) –

## `current_version` \{#ontogr_mem-service-contracts-StaleEpisodePage-current_version}

```python
current_version: int
```

## `episodes` \{#ontogr_mem-service-contracts-StaleEpisodePage-episodes}

```python
episodes: list[StaleEpisodeOut]
```

## `model_config` \{#ontogr_mem-service-contracts-StaleEpisodePage-model_config}

```python
model_config = ConfigDict(extra='forbid')
```

## `next_cursor` \{#ontogr_mem-service-contracts-StaleEpisodePage-next_cursor}

```python
next_cursor: str | None = None
```

## `tenant` \{#ontogr_mem-service-contracts-StaleEpisodePage-tenant}

```python
tenant: TenantId
```
