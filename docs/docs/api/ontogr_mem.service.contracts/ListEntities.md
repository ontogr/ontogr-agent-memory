---
title: ontogr_mem.service.contracts.ListEntities
sidebar_label: ListEntities
---

# `ontogr_mem.service.contracts.ListEntities` \{#ontogr_mem-service-contracts-ListEntities}

Bases: <code>[ScopedRequest](ScopedRequest.md)</code>

List active entities in stable id order.

**Attributes:**

- [**cursor**](#ontogr_mem-service-contracts-ListEntities-cursor) (<code>str | None</code>) –
- [**entity_type**](#ontogr_mem-service-contracts-ListEntities-entity_type) (<code>str | None</code>) –
- [**limit**](#ontogr_mem-service-contracts-ListEntities-limit) (<code>int</code>) –
- [**model_config**](#ontogr_mem-service-contracts-ListEntities-model_config) –
- [**tenant**](#ontogr_mem-service-contracts-ListEntities-tenant) (<code>[TenantId](TenantId.md) | None</code>) –

## `cursor` \{#ontogr_mem-service-contracts-ListEntities-cursor}

```python
cursor: str | None = None
```

## `entity_type` \{#ontogr_mem-service-contracts-ListEntities-entity_type}

```python
entity_type: str | None = Field(default=None, min_length=1, max_length=128)
```

## `limit` \{#ontogr_mem-service-contracts-ListEntities-limit}

```python
limit: int = Field(default=20, ge=1, le=MAX_RESULTS)
```

## `model_config` \{#ontogr_mem-service-contracts-ListEntities-model_config}

```python
model_config = ConfigDict(extra='forbid')
```

## `tenant` \{#ontogr_mem-service-contracts-ListEntities-tenant}

```python
tenant: TenantId | None = None
```
