---
title: ontogr_mem.client.OntologyView
sidebar_label: OntologyView
---

# `ontogr_mem.client.OntologyView` \{#ontogr_mem-client-OntologyView}

Bases: <code>[ContractModel](../ontogr_mem.service.contracts/ContractModel.md)</code>

Ontology vocabulary exposed to callers.

**Attributes:**

- [**entity_types**](#ontogr_mem-client-OntologyView-entity_types) (<code>list\[[EntityTypeOut](../ontogr_mem.service.contracts/EntityTypeOut.md)\]</code>) –
- [**model_config**](#ontogr_mem-client-OntologyView-model_config) –
- [**relation_types**](#ontogr_mem-client-OntologyView-relation_types) (<code>list\[[RelationTypeOut](../ontogr_mem.service.contracts/RelationTypeOut.md)\]</code>) –
- [**tenant**](#ontogr_mem-client-OntologyView-tenant) (<code>[TenantId](../ontogr_mem.service.contracts/TenantId.md)</code>) –
- [**version**](#ontogr_mem-client-OntologyView-version) (<code>int</code>) –

## `entity_types` \{#ontogr_mem-client-OntologyView-entity_types}

```python
entity_types: list[EntityTypeOut]
```

## `model_config` \{#ontogr_mem-client-OntologyView-model_config}

```python
model_config = ConfigDict(extra='forbid')
```

## `relation_types` \{#ontogr_mem-client-OntologyView-relation_types}

```python
relation_types: list[RelationTypeOut]
```

## `tenant` \{#ontogr_mem-client-OntologyView-tenant}

```python
tenant: TenantId
```

## `version` \{#ontogr_mem-client-OntologyView-version}

```python
version: int
```
