---
title: ontogr_mem.service.contracts.EpisodeOut
sidebar_label: EpisodeOut
---

# `ontogr_mem.service.contracts.EpisodeOut` \{#ontogr_mem-service-contracts-EpisodeOut}

Bases: <code>[ContractModel](ContractModel.md)</code>

One stored episode's metadata.

**Attributes:**

- [**actor_id**](#ontogr_mem-service-contracts-EpisodeOut-actor_id) (<code>str</code>) –
- [**actor_name**](#ontogr_mem-service-contracts-EpisodeOut-actor_name) (<code>str</code>) –
- [**actor_role**](#ontogr_mem-service-contracts-EpisodeOut-actor_role) (<code>str</code>) –
- [**created_at**](#ontogr_mem-service-contracts-EpisodeOut-created_at) (<code>[UtcDatetime](UtcDatetime.md)</code>) –
- [**episode_id**](#ontogr_mem-service-contracts-EpisodeOut-episode_id) (<code>str</code>) –
- [**job_id**](#ontogr_mem-service-contracts-EpisodeOut-job_id) (<code>str</code>) –
- [**kind**](#ontogr_mem-service-contracts-EpisodeOut-kind) (<code>str</code>) –
- [**model_config**](#ontogr_mem-service-contracts-EpisodeOut-model_config) –
- [**session_id**](#ontogr_mem-service-contracts-EpisodeOut-session_id) (<code>str | None</code>) –
- [**source_description**](#ontogr_mem-service-contracts-EpisodeOut-source_description) (<code>str</code>) –
- [**tenant**](#ontogr_mem-service-contracts-EpisodeOut-tenant) (<code>[TenantId](TenantId.md)</code>) –
- [**thread**](#ontogr_mem-service-contracts-EpisodeOut-thread) (<code>str | None</code>) –

## `actor_id` \{#ontogr_mem-service-contracts-EpisodeOut-actor_id}

```python
actor_id: str
```

## `actor_name` \{#ontogr_mem-service-contracts-EpisodeOut-actor_name}

```python
actor_name: str
```

## `actor_role` \{#ontogr_mem-service-contracts-EpisodeOut-actor_role}

```python
actor_role: str
```

## `created_at` \{#ontogr_mem-service-contracts-EpisodeOut-created_at}

```python
created_at: UtcDatetime
```

## `episode_id` \{#ontogr_mem-service-contracts-EpisodeOut-episode_id}

```python
episode_id: str
```

## `job_id` \{#ontogr_mem-service-contracts-EpisodeOut-job_id}

```python
job_id: str = ''
```

## `kind` \{#ontogr_mem-service-contracts-EpisodeOut-kind}

```python
kind: str
```

## `model_config` \{#ontogr_mem-service-contracts-EpisodeOut-model_config}

```python
model_config = ConfigDict(extra='forbid')
```

## `session_id` \{#ontogr_mem-service-contracts-EpisodeOut-session_id}

```python
session_id: str | None = None
```

## `source_description` \{#ontogr_mem-service-contracts-EpisodeOut-source_description}

```python
source_description: str = ''
```

## `tenant` \{#ontogr_mem-service-contracts-EpisodeOut-tenant}

```python
tenant: TenantId
```

## `thread` \{#ontogr_mem-service-contracts-EpisodeOut-thread}

```python
thread: str | None = None
```
