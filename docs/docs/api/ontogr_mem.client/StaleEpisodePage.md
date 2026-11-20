---
title: ontogr_mem.client.StaleEpisodePage
sidebar_label: StaleEpisodePage
---

# `ontogr_mem.client.StaleEpisodePage` \{#ontogr_mem-client-StaleEpisodePage}

Bases: <code>[ContractModel](../ontogr_mem.service.contracts/ContractModel.md)</code>

One page of stale episodes and the current ontology version.

**Attributes:**

- [**current_version**](#ontogr_mem-client-StaleEpisodePage-current_version) (<code>int</code>) –
- [**episodes**](#ontogr_mem-client-StaleEpisodePage-episodes) (<code>list\[[StaleEpisodeOut](../ontogr_mem.service.contracts/StaleEpisodeOut.md)\]</code>) –
- [**model_config**](#ontogr_mem-client-StaleEpisodePage-model_config) –
- [**next_cursor**](#ontogr_mem-client-StaleEpisodePage-next_cursor) (<code>str | None</code>) –
- [**tenant**](#ontogr_mem-client-StaleEpisodePage-tenant) (<code>[TenantId](../ontogr_mem.service.contracts/TenantId.md)</code>) –

## `current_version` \{#ontogr_mem-client-StaleEpisodePage-current_version}

```python
current_version: int
```

## `episodes` \{#ontogr_mem-client-StaleEpisodePage-episodes}

```python
episodes: list[StaleEpisodeOut]
```

## `model_config` \{#ontogr_mem-client-StaleEpisodePage-model_config}

```python
model_config = ConfigDict(extra='forbid')
```

## `next_cursor` \{#ontogr_mem-client-StaleEpisodePage-next_cursor}

```python
next_cursor: str | None = None
```

## `tenant` \{#ontogr_mem-client-StaleEpisodePage-tenant}

```python
tenant: TenantId
```
