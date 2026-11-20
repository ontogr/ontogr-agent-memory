---
title: ontogr_mem.service.contracts.OntologyView
sidebar_label: OntologyView
---

# `ontogr_mem.service.contracts.OntologyView` \{#ontogr_mem-service-contracts-OntologyView}

Bases: <code>[ContractModel](ContractModel.md)</code>

Ontology vocabulary exposed to callers.

**Attributes:**

- [**entity_types**](#ontogr_mem-service-contracts-OntologyView-entity_types) (<code>list\[[EntityTypeOut](EntityTypeOut.md)\]</code>) –
- [**model_config**](#ontogr_mem-service-contracts-OntologyView-model_config) –
- [**relation_types**](#ontogr_mem-service-contracts-OntologyView-relation_types) (<code>list\[[RelationTypeOut](RelationTypeOut.md)\]</code>) –
- [**tenant**](#ontogr_mem-service-contracts-OntologyView-tenant) (<code>[TenantId](TenantId.md)</code>) –
- [**version**](#ontogr_mem-service-contracts-OntologyView-version) (<code>int</code>) –

## `entity_types` \{#ontogr_mem-service-contracts-OntologyView-entity_types}

```python
entity_types: list[EntityTypeOut]
```

## `model_config` \{#ontogr_mem-service-contracts-OntologyView-model_config}

```python
model_config = ConfigDict(extra='forbid')
```

## `relation_types` \{#ontogr_mem-service-contracts-OntologyView-relation_types}

```python
relation_types: list[RelationTypeOut]
```

## `tenant` \{#ontogr_mem-service-contracts-OntologyView-tenant}

```python
tenant: TenantId
```

## `version` \{#ontogr_mem-service-contracts-OntologyView-version}

```python
version: int
```
