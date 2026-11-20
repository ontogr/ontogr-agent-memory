---
title: ontogr_mem.service.contracts.ForgetResult
sidebar_label: ForgetResult
---

# `ontogr_mem.service.contracts.ForgetResult` \{#ontogr_mem-service-contracts-ForgetResult}

Bases: <code>[ContractModel](ContractModel.md)</code>

The outcome of a forget, or of its dry run.

**Attributes:**

- [**collateral_fact_ids**](#ontogr_mem-service-contracts-ForgetResult-collateral_fact_ids) (<code>list\[str\]</code>) –
- [**confirm_token**](#ontogr_mem-service-contracts-ForgetResult-confirm_token) (<code>str | None</code>) –
- [**dry_run**](#ontogr_mem-service-contracts-ForgetResult-dry_run) (<code>bool</code>) –
- [**episode_ids**](#ontogr_mem-service-contracts-ForgetResult-episode_ids) (<code>list\[str\]</code>) –
- [**fact_ids**](#ontogr_mem-service-contracts-ForgetResult-fact_ids) (<code>list\[str\]</code>) –
- [**mode**](#ontogr_mem-service-contracts-ForgetResult-mode) (<code>[ForgetMode](ForgetMode.md) | Literal['clear']</code>) –
- [**model_config**](#ontogr_mem-service-contracts-ForgetResult-model_config) –
- [**not_cleaned**](#ontogr_mem-service-contracts-ForgetResult-not_cleaned) (<code>list\[str\]</code>) –
- [**receipt**](#ontogr_mem-service-contracts-ForgetResult-receipt) (<code>[WriteReceipt](WriteReceipt.md) | None</code>) –

## `collateral_fact_ids` \{#ontogr_mem-service-contracts-ForgetResult-collateral_fact_ids}

```python
collateral_fact_ids: list[str] = Field(default_factory=list)
```

## `confirm_token` \{#ontogr_mem-service-contracts-ForgetResult-confirm_token}

```python
confirm_token: str | None = None
```

## `dry_run` \{#ontogr_mem-service-contracts-ForgetResult-dry_run}

```python
dry_run: bool
```

## `episode_ids` \{#ontogr_mem-service-contracts-ForgetResult-episode_ids}

```python
episode_ids: list[str] = Field(default_factory=list)
```

## `fact_ids` \{#ontogr_mem-service-contracts-ForgetResult-fact_ids}

```python
fact_ids: list[str] = Field(default_factory=list)
```

## `mode` \{#ontogr_mem-service-contracts-ForgetResult-mode}

```python
mode: ForgetMode | Literal['clear']
```

## `model_config` \{#ontogr_mem-service-contracts-ForgetResult-model_config}

```python
model_config = ConfigDict(extra='forbid')
```

## `not_cleaned` \{#ontogr_mem-service-contracts-ForgetResult-not_cleaned}

```python
not_cleaned: list[str] = Field(default_factory=list, description='Residue erase found and could not clean; empty in other modes.')
```

## `receipt` \{#ontogr_mem-service-contracts-ForgetResult-receipt}

```python
receipt: WriteReceipt | None = None
```
