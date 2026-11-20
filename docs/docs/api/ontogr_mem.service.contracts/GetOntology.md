---
title: ontogr_mem.service.contracts.GetOntology
sidebar_label: GetOntology
---

# `ontogr_mem.service.contracts.GetOntology` \{#ontogr_mem-service-contracts-GetOntology}

Bases: <code>[ScopedRequest](ScopedRequest.md)</code>

Read the tenant's ontology labels and descriptions.

**Attributes:**

- [**include_deprecated**](#ontogr_mem-service-contracts-GetOntology-include_deprecated) (<code>bool</code>) –
- [**model_config**](#ontogr_mem-service-contracts-GetOntology-model_config) –
- [**tenant**](#ontogr_mem-service-contracts-GetOntology-tenant) (<code>[TenantId](TenantId.md) | None</code>) –

## `include_deprecated` \{#ontogr_mem-service-contracts-GetOntology-include_deprecated}

```python
include_deprecated: bool = False
```

## `model_config` \{#ontogr_mem-service-contracts-GetOntology-model_config}

```python
model_config = ConfigDict(extra='forbid')
```

## `tenant` \{#ontogr_mem-service-contracts-GetOntology-tenant}

```python
tenant: TenantId | None = None
```
