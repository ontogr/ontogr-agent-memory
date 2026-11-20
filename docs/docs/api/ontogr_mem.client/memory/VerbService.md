---
title: ontogr_mem.client.memory.VerbService
sidebar_label: VerbService
---

# `ontogr_mem.client.memory.VerbService` \{#ontogr_mem-client-memory-VerbService}

Bases: <code>Protocol</code>

Async service operations used by the public Memory client.

**Functions:**

- [**add_memory**](#ontogr_mem-client-memory-VerbService-add_memory) – Add memory and return its write receipt.
- [**drain**](#ontogr_mem-client-memory-VerbService-drain) – Wait for pending writes to finish.
- [**forget_memory**](#ontogr_mem-client-memory-VerbService-forget_memory) – Forget memory and return the outcome.
- [**get_entity**](#ontogr_mem-client-memory-VerbService-get_entity) – Return one entity view.
- [**get_job**](#ontogr_mem-client-memory-VerbService-get_job) – Return the status of one write job.
- [**get_ontology**](#ontogr_mem-client-memory-VerbService-get_ontology) – Return the ontology vocabulary for one tenant.
- [**invalidate**](#ontogr_mem-client-memory-VerbService-invalidate) – Invalidate memory and return its write receipt.
- [**list_entities**](#ontogr_mem-client-memory-VerbService-list_entities) – List active entities for one tenant.
- [**search_memory**](#ontogr_mem-client-memory-VerbService-search_memory) – Search memory for relevant context.
- [**stale_episodes**](#ontogr_mem-client-memory-VerbService-stale_episodes) – List episodes that need extraction under the current ontology.

## `add_memory` \{#ontogr_mem-client-memory-VerbService-add_memory}

```python
add_memory(principal:Principal, request:AddMemory) -> WriteReceipt
```

Add memory and return its write receipt.

## `drain` \{#ontogr_mem-client-memory-VerbService-drain}

```python
drain() -> None
```

Wait for pending writes to finish.

## `forget_memory` \{#ontogr_mem-client-memory-VerbService-forget_memory}

```python
forget_memory(principal:Principal, request:ForgetMemory) -> ForgetResult
```

Forget memory and return the outcome.

## `get_entity` \{#ontogr_mem-client-memory-VerbService-get_entity}

```python
get_entity(principal:Principal, request:GetEntity) -> EntityView
```

Return one entity view.

## `get_job` \{#ontogr_mem-client-memory-VerbService-get_job}

```python
get_job(principal:Principal, job_id:str) -> JobView
```

Return the status of one write job.

## `get_ontology` \{#ontogr_mem-client-memory-VerbService-get_ontology}

```python
get_ontology(principal:Principal, request:GetOntology) -> OntologyView
```

Return the ontology vocabulary for one tenant.

## `invalidate` \{#ontogr_mem-client-memory-VerbService-invalidate}

```python
invalidate(principal:Principal, request:InvalidateMemory) -> WriteReceipt
```

Invalidate memory and return its write receipt.

## `list_entities` \{#ontogr_mem-client-memory-VerbService-list_entities}

```python
list_entities(principal:Principal, request:ListEntities) -> EntityPage
```

List active entities for one tenant.

## `search_memory` \{#ontogr_mem-client-memory-VerbService-search_memory}

```python
search_memory(principal:Principal, request:SearchMemory) -> MemoryContext
```

Search memory for relevant context.

## `stale_episodes` \{#ontogr_mem-client-memory-VerbService-stale_episodes}

```python
stale_episodes(principal:Principal, request:ListStaleEpisodes) -> StaleEpisodePage
```

List episodes that need extraction under the current ontology.
