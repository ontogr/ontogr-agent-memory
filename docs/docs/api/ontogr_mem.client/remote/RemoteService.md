---
title: ontogr_mem.client.remote.RemoteService
sidebar_label: RemoteService
---

# `ontogr_mem.client.remote.RemoteService` \{#ontogr_mem-client-remote-RemoteService}

```python
RemoteService(client:Any) -> None
```

Send tenant-scoped memory operations to a remote MCP server.

**Functions:**

- [**aclose**](#ontogr_mem-client-remote-RemoteService-aclose) – Close the MCP connection.
- [**add_memory**](#ontogr_mem-client-remote-RemoteService-add_memory) – Add memory through the remote server.
- [**drain**](#ontogr_mem-client-remote-RemoteService-drain) – Wait for remote writes tracked by this connection to finish.
- [**forget_memory**](#ontogr_mem-client-remote-RemoteService-forget_memory) – Reject unsupported direct forget requests.
- [**get_entity**](#ontogr_mem-client-remote-RemoteService-get_entity) – Return one entity view from the remote server.
- [**get_job**](#ontogr_mem-client-remote-RemoteService-get_job) – Return the status of one remote write job.
- [**get_ontology**](#ontogr_mem-client-remote-RemoteService-get_ontology) – Return the remote tenant ontology.
- [**invalidate**](#ontogr_mem-client-remote-RemoteService-invalidate) – Invalidate remote memory and return its write receipt.
- [**list_entities**](#ontogr_mem-client-remote-RemoteService-list_entities) – List active entities through the remote server.
- [**open**](#ontogr_mem-client-remote-RemoteService-open) – Open one authenticated MCP connection for a tenant.
- [**search_memory**](#ontogr_mem-client-remote-RemoteService-search_memory) – Search remote memory for relevant context.
- [**stale_episodes**](#ontogr_mem-client-remote-RemoteService-stale_episodes) – List stale episodes through the remote server.

## `aclose` \{#ontogr_mem-client-remote-RemoteService-aclose}

```python
aclose() -> None
```

Close the MCP connection.

## `add_memory` \{#ontogr_mem-client-remote-RemoteService-add_memory}

```python
add_memory(principal:Principal, request:AddMemory) -> WriteReceipt
```

Add memory through the remote server.

## `drain` \{#ontogr_mem-client-remote-RemoteService-drain}

```python
drain() -> None
```

Wait for remote writes tracked by this connection to finish.

## `forget_memory` \{#ontogr_mem-client-remote-RemoteService-forget_memory}

```python
forget_memory(principal:Principal, request:Any) -> Any
```

Reject unsupported direct forget requests.

## `get_entity` \{#ontogr_mem-client-remote-RemoteService-get_entity}

```python
get_entity(principal:Principal, request:GetEntity) -> EntityView
```

Return one entity view from the remote server.

## `get_job` \{#ontogr_mem-client-remote-RemoteService-get_job}

```python
get_job(principal:Principal, job_id:str) -> JobView
```

Return the status of one remote write job.

## `get_ontology` \{#ontogr_mem-client-remote-RemoteService-get_ontology}

```python
get_ontology(principal:Principal, request:GetOntology) -> OntologyView
```

Return the remote tenant ontology.

## `invalidate` \{#ontogr_mem-client-remote-RemoteService-invalidate}

```python
invalidate(principal:Principal, request:InvalidateMemory) -> WriteReceipt
```

Invalidate remote memory and return its write receipt.

## `list_entities` \{#ontogr_mem-client-remote-RemoteService-list_entities}

```python
list_entities(principal:Principal, request:ListEntities) -> EntityPage
```

List active entities through the remote server.

## `open` \{#ontogr_mem-client-remote-RemoteService-open}

```python
open(*, url:str, token:str | None, tenant:str) -> RemoteService
```

Open one authenticated MCP connection for a tenant.

## `search_memory` \{#ontogr_mem-client-remote-RemoteService-search_memory}

```python
search_memory(principal:Principal, request:SearchMemory) -> MemoryContext
```

Search remote memory for relevant context.

## `stale_episodes` \{#ontogr_mem-client-remote-RemoteService-stale_episodes}

```python
stale_episodes(principal:Principal, request:ListStaleEpisodes) -> StaleEpisodePage
```

List stale episodes through the remote server.
