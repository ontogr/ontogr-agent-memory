---
title: ontogr_mem.service.contracts.EntityOut
sidebar_label: EntityOut
---

# `ontogr_mem.service.contracts.EntityOut` \{#ontogr_mem-service-contracts-EntityOut}

Bases: <code>[ContractModel](ContractModel.md)</code>

A canonical entity.

**Attributes:**

- [**proposed_type**](#ontogr_mem-service-contracts-EntityOut-proposed_type) (<code>str</code>) – Type name the model gave while the type is Unknown.

## `aliases` \{#ontogr_mem-service-contracts-EntityOut-aliases}

```python
aliases: list[str] = Field(default_factory=list)
```

## `entity_id` \{#ontogr_mem-service-contracts-EntityOut-entity_id}

```python
entity_id: str
```

## `model_config` \{#ontogr_mem-service-contracts-EntityOut-model_config}

```python
model_config = ConfigDict(extra='forbid')
```

## `name` \{#ontogr_mem-service-contracts-EntityOut-name}

```python
name: str
```

## `proposed_type` \{#ontogr_mem-service-contracts-EntityOut-proposed_type}

```python
proposed_type: str = ''
```

## `summary` \{#ontogr_mem-service-contracts-EntityOut-summary}

```python
summary: str = ''
```

## `tenant` \{#ontogr_mem-service-contracts-EntityOut-tenant}

```python
tenant: TenantId
```

## `type_label` \{#ontogr_mem-service-contracts-EntityOut-type_label}

```python
type_label: str = ''
```
