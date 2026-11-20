---
title: ontogr_mem.service.contracts.StaleEpisodeOut
sidebar_label: StaleEpisodeOut
---

# `ontogr_mem.service.contracts.StaleEpisodeOut` \{#ontogr_mem-service-contracts-StaleEpisodeOut}

Bases: <code>[ContractModel](ContractModel.md)</code>

One episode that predates the current ontology version.

**Attributes:**

- [**episode_id**](#ontogr_mem-service-contracts-StaleEpisodeOut-episode_id) (<code>str</code>) –
- [**ingested_at**](#ontogr_mem-service-contracts-StaleEpisodeOut-ingested_at) (<code>[UtcDatetime](UtcDatetime.md)</code>) –
- [**kind**](#ontogr_mem-service-contracts-StaleEpisodeOut-kind) (<code>str</code>) –
- [**model_config**](#ontogr_mem-service-contracts-StaleEpisodeOut-model_config) –
- [**ontology_version**](#ontogr_mem-service-contracts-StaleEpisodeOut-ontology_version) (<code>int</code>) –

## `episode_id` \{#ontogr_mem-service-contracts-StaleEpisodeOut-episode_id}

```python
episode_id: str
```

## `ingested_at` \{#ontogr_mem-service-contracts-StaleEpisodeOut-ingested_at}

```python
ingested_at: UtcDatetime
```

## `kind` \{#ontogr_mem-service-contracts-StaleEpisodeOut-kind}

```python
kind: str
```

## `model_config` \{#ontogr_mem-service-contracts-StaleEpisodeOut-model_config}

```python
model_config = ConfigDict(extra='forbid')
```

## `ontology_version` \{#ontogr_mem-service-contracts-StaleEpisodeOut-ontology_version}

```python
ontology_version: int
```
