---
title: ontogr_mem.service.contracts.EpisodePage
sidebar_label: EpisodePage
---

# `ontogr_mem.service.contracts.EpisodePage` \{#ontogr_mem-service-contracts-EpisodePage}

Bases: <code>[ContractModel](ContractModel.md)</code>

A page of episodes and the cursor for the next one.

**Attributes:**

- [**episodes**](#ontogr_mem-service-contracts-EpisodePage-episodes) (<code>list\[[EpisodeOut](EpisodeOut.md)\]</code>) –
- [**model_config**](#ontogr_mem-service-contracts-EpisodePage-model_config) –
- [**next_cursor**](#ontogr_mem-service-contracts-EpisodePage-next_cursor) (<code>str | None</code>) –

## `episodes` \{#ontogr_mem-service-contracts-EpisodePage-episodes}

```python
episodes: list[EpisodeOut]
```

## `model_config` \{#ontogr_mem-service-contracts-EpisodePage-model_config}

```python
model_config = ConfigDict(extra='forbid')
```

## `next_cursor` \{#ontogr_mem-service-contracts-EpisodePage-next_cursor}

```python
next_cursor: str | None = None
```
