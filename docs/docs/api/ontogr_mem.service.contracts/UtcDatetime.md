---
title: ontogr_mem.service.contracts.UtcDatetime
sidebar_label: UtcDatetime
---

# `ontogr_mem.service.contracts.UtcDatetime` \{#ontogr_mem-service-contracts-UtcDatetime}

```python
UtcDatetime = Annotated[datetime, AfterValidator(require_utc)]
```

A timezone-aware datetime; one without a timezone fails validation.
