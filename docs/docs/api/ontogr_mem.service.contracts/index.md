---
title: ontogr_mem.service.contracts
sidebar_position: 4
---


# `ontogr_mem.service.contracts` \{#ontogr_mem-service-contracts}

The request and result models every verb shares.

**Classes:**

- [**AddMemory**](AddMemory.md) – Store a fact or extract facts from text.
- [**ClearTenant**](ClearTenant.md) – Remove everything in a tenant. Needs a dry run, then its token.
- [**ConsolidateMemory**](ConsolidateMemory.md) – Run the consolidation stages for a tenant as a background job.
- [**ContractModel**](ContractModel.md) – Base for every contract model: unknown fields are errors.
- [**EntityOut**](EntityOut.md) – A canonical entity.
- [**EntityPage**](EntityPage.md) – One page of active entities.
- [**EntityRef**](EntityRef.md) – An entity named in a fact.
- [**EntityTypeOut**](EntityTypeOut.md) – Public description of an entity type.
- [**EntityView**](EntityView.md) – An entity and the facts that touch it.
- [**EpisodeOut**](EpisodeOut.md) – One stored episode's metadata.
- [**EpisodePage**](EpisodePage.md) – A page of episodes and the cursor for the next one.
- [**ErrorInfo**](ErrorInfo.md) – The wire form of a service error.
- [**FactOut**](FactOut.md) – One fact with its validity window and provenance.
- [**ForgetMemory**](ForgetMemory.md) – Close or remove memory. Hard modes need a dry run, then the token.
- [**ForgetMode**](ForgetMode.md) – How much forget_memory removes.
- [**ForgetResult**](ForgetResult.md) – The outcome of a forget, or of its dry run.
- [**GetEntity**](GetEntity.md) – Look up one entity by id or name, with its facts.
- [**GetOntology**](GetOntology.md) – Read the tenant's ontology labels and descriptions.
- [**HostFact**](HostFact.md) – A host-extracted link supported by the accompanying source text.
- [**InvalidateMemory**](InvalidateMemory.md) – Close one current fact while retaining its version history.
- [**JobView**](JobView.md) – A job's state visible to an authorized caller.
- [**ListEntities**](ListEntities.md) – List active entities in stable id order.
- [**ListEpisodes**](ListEpisodes.md) – Page through stored episodes, newest first.
- [**ListStaleEpisodes**](ListStaleEpisodes.md) – Page through episodes extracted under an older ontology version.
- [**MemoryContext**](MemoryContext.md) – Search output. `state` separates "nothing stored" from "not ready".
- [**MemoryFeedback**](MemoryFeedback.md) – A score on one entity or on a fact's two entities.
- [**MemoryKind**](MemoryKind.md) – What add_memory was given.
- [**OntologyView**](OntologyView.md) – Ontology vocabulary exposed to callers.
- [**PatternOut**](PatternOut.md) – A permitted source and target entity type pair.
- [**RelationTypeOut**](RelationTypeOut.md) – Public description of a relation type.
- [**ScopedRequest**](ScopedRequest.md) – A request within one tenant; `None` means the principal's tenant.
- [**SearchMemory**](SearchMemory.md) – Recall facts, entities and a ready-to-prompt context.
- [**StaleEpisodeOut**](StaleEpisodeOut.md) – One episode that predates the current ontology version.
- [**StaleEpisodePage**](StaleEpisodePage.md) – One page of stale episodes and the current ontology version.
- [**StatusOut**](StatusOut.md) – Service health.
- [**UpdateMemory**](UpdateMemory.md) – Replace a fact: close the old version and assert the new one.
- [**WriteReceipt**](WriteReceipt.md) – What a write returns at once: a job to follow.
- [**WriteRequest**](WriteRequest.md) – A request that queues a job and may wait for it.

**Functions:**

- [**validate_request**](validate_request.md) – Parse a raw request mapping into its contract model.
- [**validation_error**](validation_error.md) – Turn pydantic error dicts into one typed service error.

**Attributes:**

- [**DEFAULT_MODALITIES**](DEFAULT_MODALITIES.md) (<code>tuple\[Literal['actual', 'planned', 'conditional', 'desired'], ...\]</code>) – The modalities a search returns unless the caller names others.
- [**ModelT**](ModelT.md) –
- [**TenantId**](TenantId.md) – A tenant id: 1 to 60 letters, digits, dashes or underscores.
- [**UtcDatetime**](UtcDatetime.md) – A timezone-aware datetime; one without a timezone fails validation.
