---
title: ontogr_mem.service.contracts.TenantId
sidebar_label: TenantId
---

# `ontogr_mem.service.contracts.TenantId` \{#ontogr_mem-service-contracts-TenantId}

```python
TenantId = Annotated[str, AfterValidator(validate_tenant_id)]
```

A tenant id: 1 to 60 letters, digits, dashes or underscores.
