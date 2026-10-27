<h1 align="center"><a href="https://github.com/ontogr/ontogr-agent-memory">OntoGr Agent Memory</a></h1>

<h3 align="center">Durable, traceable memory for AI agents.</h3>
<h4 align="center"><a href="https://github.com/ontogr/ontogr-agent-memory#features">Features</a> · <a href="https://github.com/ontogr/ontogr-agent-memory#quickstart">Quickstart</a> · <a href="https://github.com/ontogr/ontogr-agent-memory#core-concepts">Concepts</a> · <a href="https://github.com/ontogr/ontogr-agent-memory#how-data-flows">Data flow</a> · <a href="https://github.com/ontogr/ontogr-agent-memory#service-api">Service API</a> · <a href="https://github.com/ontogr/ontogr-agent-memory#configuration">Configuration</a> · <a href="https://github.com/ontogr/ontogr-agent-memory/blob/main/CONTEXT.md">Vocabulary</a> · <a href="https://github.com/ontogr/ontogr-agent-memory/blob/main/CHANGELOG.md">Changelog</a></h4>

<div align="center">

[![PyPI](https://img.shields.io/pypi/v/ontogr-mem.svg)](https://pypi.org/project/ontogr-mem/)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue)](https://pypi.org/project/ontogr-mem/)
[![DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/ontogr/ontogr-agent-memory)
[![License](https://img.shields.io/github/license/ontogr/ontogr-agent-memory?color=green)](https://github.com/ontogr/ontogr-agent-memory/blob/main/LICENSE)

</div>

<div align="center">

[![CI](https://img.shields.io/github/actions/workflow/status/ontogr/ontogr-agent-memory/ci.yml?branch=main&label=CI&logo=githubactions)](https://github.com/ontogr/ontogr-agent-memory/actions/workflows/ci.yml)
[![Integration tests](https://img.shields.io/github/actions/workflow/status/ontogr/ontogr-agent-memory/integration.yml?branch=main&label=integration%20tests&logo=pytest)](https://github.com/ontogr/ontogr-agent-memory/actions/workflows/integration.yml)
[![Codecov](https://codecov.io/gh/ontogr/ontogr-agent-memory/branch/main/graph/badge.svg)](https://codecov.io/gh/ontogr/ontogr-agent-memory)

</div>

OntoGr Agent Memory is an open source Python library that stores what an agent learned, where it learned it, and when it was true. It ingests source material as episodes, extracts entities and facts, resolves duplicate entities, reconciles facts on two time axes, and returns a budgeted context block that is ready for an answering model.

- **Facts with sources:** Every fact keeps the verbatim quote and the episode it came from. A hash-chained ledger records each temporal write, so an edited, inserted, or deleted entry can be found.
- **Facts that change over time:** Each fact has a validity window and a record of when the system learned it. A new value closes the old version, so a read can ask what was true on a date or what the system believed on that date.
- **Contradictions get a ruling:** Rules, an optional model judge, and a person's decision settle which facts conflict, and a failed judge call is never stored as "no conflict".
- **Fits your stack:** The graph runs on FalkorDB or Neo4j, vectors live on the graph or in pgvector, and jobs and audit rows live in SQLite or Postgres. Each tenant is a separate partition.

<p align="center">
  <img src="https://raw.githubusercontent.com/ontogr/ontogr-agent-memory/main/img/hero.svg" alt="Source items enter the extraction layer as episodes. Extracted facts go to the temporal and consistency engine, which writes fact versions to the storage layer. Retrieval reads the storage layer and returns a budgeted context block." width="860">
</p>

## Features

### Extraction layer

*Episodes become typed entities and grounded facts, and the ontology decides which types are allowed.*

<p align="center">
  <img src="https://raw.githubusercontent.com/ontogr/ontogr-agent-memory/main/img/extraction-layer.svg" alt="A source item becomes an episode. The entity extractor and the fact extractor both read it, and the ontology manager gives each the types it allows. Entity resolution links extracted names to canonical entities in the graph store, and the fact extractor passes facts to the temporal engine." width="860">
</p>

Ingestion stores each item as an episode with its source and actor. Extraction then reads the episode in two passes, one for entities and one for facts. The ontology sets the allowed types, and resolution links each name to a canonical entity.

#### Episode ingestion

*Each item becomes an episode that records where it came from and who produced it.*

An agent can only explain or correct a memory if it can find the source. Ingestion stores each item as an episode with its source and actor, and every later step points back to that episode.

Ingestion accepts five source kinds:

| Kind | Input | Output |
| --- | --- | --- |
| `message` | One chat message | One episode |
| `document` | A document split into parts | One episode per non-empty part |
| `tool_output` | One tool call result | One episode |
| `event` | One provider event | One episode |
| `record` | One row from a system of record | One episode per row update |

- **Batches are validated first:** The pipeline validates the whole batch before it writes anything.
- **Repeated content is skipped:** The episode id is deterministic and comes from the source, the occurrence time, and the content hash.
- **Crashes are safe:** The content hash is written last as a commit marker. A crash before it leaves the episode pending, and the next run writes it again.
- **Callbacks are yours:** New episodes can be handed to a callback, such as entity extraction. The caller owns that connection and its retry policy.

#### Entity and relation extractor

*Entity extraction finds the names in an episode, and every name must appear in the text.*

Entity extraction and fact extraction are separate passes over the same episode. Entity extraction returns entities. The relation between two entities comes from fact extraction, where each fact carries a relation that the ontology maps to a stable type. Entity extraction supports these extractors:

- an LLM extractor with a numbered type catalog
- GLiNER and spaCy extractors
- a record mapper for structured rows and events
- a memory extractor that adds descriptions and attributes
- a composite extractor that combines results
- a fallback extractor that runs a cheap pass first and escalates to a model only when triggers fire

Entity extraction follows these rules:

- **Names must appear in the episode:** Code finds each span, and a name that cannot be found is discarded.
- **Failures are visible:** A failed model call is recorded with a retryable status that differs from an empty result.

#### Fact and statement extractor

*Each fact is a dated statement with a verbatim quote from the episode.*

A language model can state things that the text never says. Fact extraction checks each result against the episode and keeps the quote behind each fact, so a reader can verify the fact without trusting the model.

- **Every fact has a quote:** Each extracted fact has a subject, a relation, and an object. It also carries a statement, a polarity, a modality, and a valid-time window, and it needs a verbatim quote from the current episode.
- **A bad part is cleared:** A bad relation type or a time that cannot be parsed clears that part, and the fact stays with a reason code. Relative times resolve against the episode's occurrence time, and stated precision is kept.
- **Results are reused:** Each episode has a processing status and a stored result. A completed `ok` or `empty` result is reused, and a `parse_failed`, `error`, or `skipped` episode can be processed again.

#### Ontology manager

*Each tenant decides which entity types and relation types are allowed.*

Free-form labels drift, so one relation can appear under many names and its facts cannot be compared. A catalog gives each type a stable id, and a change to the catalog goes through review.

Each tenant has a catalog of entity types and relation types, plus attributes and aliases. Type ids are stable integers. A type can be deprecated or merged, but old ids remain resolvable. A catalog change is a change set applied by one pure function and committed with a compare-and-swap against the current head, so a stale head fails instead of overwriting another change.

Unknown type labels are handled by mode:

| Mode | Behavior |
| --- | --- |
| `strict` | The item is dropped and counted. |
| `preferred` | The item is dropped, and an unmatched row is recorded for review. |
| `open` | The label is registered immediately. |

- **Patterns are enforced:** Relation types can declare source and target patterns. The patterns appear in the prompt, and the code also enforces them.
- **People approve proposals:** A model can draft proposals from labels that appear in enough distinct episodes. A person accepts or rejects each proposal, and a proposal is never applied automatically.
- **Changes report their reach:** After a catalog change, the ontology reports which episodes may produce a different result so the caller can queue them for extraction again.
- **A default catalog exists:** A new tenant can start from a default catalog of common types for entities and relations.

#### Entity resolution

*Names that refer to the same thing resolve to one entity, and a merge can be reversed.*

Facts connect entities, so a duplicate entity splits what the system knows into pieces. A wrong merge does more harm, because it mixes facts about different things. The ladder below uses the cheapest evidence first and asks a model only about pairs that remain ambiguous, and every merge keeps what it needs to be undone.

<p align="center">
  <img src="https://raw.githubusercontent.com/ontogr/ontogr-agent-memory/main/img/resolution-ladder.svg" alt="An extracted name goes down four steps: URI match, strict key or alias match, fuzzy match, and a model verdict. The first step that is sure links the name to a canonical entity. The model can also call a pair distinct, which creates a new entity, or leave it uncertain, which creates a provisional one." width="860">
</p>

Resolution links extracted names to canonical entities through a ladder:

1. a URI match
2. a strict key or alias match
3. a fuzzy Jaccard match
4. a model verdict for the remaining ambiguous pairs

- **Ids survive merges:** An entity id is a UUID5 over the first mention, so it survives a merge and a deleted duplicate is not recreated from its name. A type-family policy decides which entities may merge.
- **Three outcomes:** A name links to an existing entity, becomes a new entity when no candidate remains or the verifier calls every candidate distinct, or becomes a provisional entity that is checked again later.
- **Merges are reversible:** A merge never deletes the absorbed entity. It keeps a `merged_into` pointer and a snapshot, so an unmerge can restore both sides.
- **Mentions and aliases are records:** Entity verdicts record same, distinct, or uncertain decisions for later runs.
- **Model failures never merge:** Deferred deduplication uses blocking and vector candidates, then verifies the pairs that need a decision. An unreadable verdict is not remembered as a decision.

### Temporal and consistency engine

*Facts keep two time axes, and conflicts are settled by rules, a judge, or a person.*

#### Bi-temporal model

*Each fact records when it was true and when the system learned it.*

<p align="center">
  <img src="https://raw.githubusercontent.com/ontogr/ontogr-agent-memory/main/img/time-axes.svg" alt="Two rows show one slot over time. Valid time shows when each fact was true in the world. System time shows when the system recorded each version. A change writes a closed copy of the old fact and a new fact, and stamps expired_at on the old version, so a read can ask what was true or what was known on a date." width="860">
</p>

Facts change, and corrections often arrive after the event. Two time axes let a read answer both what is true now and what the system believed at an earlier date. A change writes a new version and replaces nothing, so the history stays available.

Facts use two time axes:

| Axis | Fields | Meaning |
| --- | --- | --- |
| Valid time | `valid_from`, `valid_to` | When the fact was true in the world. |
| System time | `recorded_at`, `expired_at` | When the system recorded or closed the fact. |

The default read returns the current belief. `valid_at`, `known_at`, overlap windows, and `include_history` change the view. A repeated fact after an end opens a new run instead of extending the old one.

- **Arrival order does not matter:** The timeline is a pure function of the whole observation set.
- **Slot commits are atomic:** A commit compares the slot revision, writes new fact versions, writes supersession rows, and appends ledger entries in one statement. A revision mismatch raises a conflict for the caller to reload and retry. Pending temporal writes are recovered on service start.

Three operations look similar but differ:

| Operation | Effect |
| --- | --- |
| Retraction | Marks an episode withdrawn, keeps its node, and writes a ledger entry. Facts built from it are closed in history. |
| Reset | Clears what extraction and reconciliation produced for one episode, keeps the node, and re-derives the affected slots. |
| Deletion | Removes an episode and what hangs on it. |

#### Contradiction handler

*Rules, an optional model judge, and a person's decision settle which facts conflict.*

<p align="center">
  <img src="https://raw.githubusercontent.com/ontogr/ontogr-agent-memory/main/img/contradiction-handling.svg" alt="Observations in one slot go to the rules when the relation is single-valued. Pairs from other slots go to an optional model judge, which stores its verdicts. The handler reads the rules and the stored verdicts, and a person's decision overrides both. The result is a set of slot directives that the engine commits as fact versions." width="860">
</p>

When two facts cannot both stand, the handler decides which one is current and keeps the other in the history. A person's decision overrides every rule and every verdict.

- **Rules:** Rules handle single-valued relations, where two values cannot both stand in the same slot.
- **Model judge:** An optional judge handles the remaining cross-slot pairs.
- **Person's decision:** A decision recorded through `TemporalEngine.record_decision` overrides the rules and the judge.
- **Conflict classes:** A verdict is one of `update`, `correction`, `dispute`, or `uncertain`.
- **Verdicts go stale:** A verdict is keyed by the pair, a hash of the inputs, and the prompt version, so a changed fact makes it stale. A failed judge call is counted and retried, and it is never stored as "no contradiction".

#### Provenance tracking

*Every fact leads back to a quote and an episode, and the ledger makes edits detectable.*

<p align="center">
  <img src="https://raw.githubusercontent.com/ontogr/ontogr-agent-memory/main/img/provenance.svg" alt="A read result traces to a fact version, which carries the ids of its observations. Each observation holds a statement and a verbatim quote, and points to its episode. Separately, every temporal write appends a ledger entry that holds the hash of the entry before it, and verify checks the chain." width="860">
</p>

Every fact can be traced back to the text that supports it, and every temporal write can be checked afterward.

- **Observation:** An observation holds one dated claim with its statement, the verbatim quote, and the episode id.
- **Fact edge:** Each fact edge carries the ids of the observations that support it.
- **Episode:** An episode keeps its source, its actor, and the content hash that marks it committed.
- **Ledger:** Every temporal write appends to a hash-chained ledger, and `verify` finds an edited, inserted, or deleted entry in the middle of a chain.

### Storage layer

*The graph holds the memory, vectors have their own store, and a separate relation store holds jobs and audit rows.*

<p align="center">
  <img src="https://raw.githubusercontent.com/ontogr/ontogr-agent-memory/main/img/storage-layer.svg" alt="Components reach storage through async ports. The graph store holds entities and facts, and it holds episodes with their raw content. The vector store holds embeddings keyed by graph ids, on the graph or in pgvector. The relation store holds jobs, operation rows, audit rows, and the erase journal in SQLite or Postgres." width="860">
</p>

Every component talks to storage through its own async port. A port has domain verbs and no raw query method, so domain code sees only ports.

| Store | Holds | Backends |
| --- | --- | --- |
| Graph store | Episodes with their sources and actors, entities, aliases, mentions, observations, fact versions, the temporal ledger, the ontology, and consolidation records. | FalkorDB (default) or Neo4j |
| Vector store | Embeddings of entity names, fact statements, and episode text, each with a hash of the embedded text. | Graph-native (default) or pgvector |
| Relation store | Jobs with their request payloads, operation rows, audit rows, and the erase journal. | SQLite (default) or Postgres |

A deployment that shares state between processes can use FalkorDB for the graph, pgvector for vectors, and Postgres for the relation store. The relation store and the vector store also have in-memory backends for tests and local runs. The relation store requires `allow_ephemeral` to use one.

- **Contract tests:** Each port has one contract suite. It runs against each live database adapter, and concurrent tenants and queries are tested on live servers.

#### Graph store

*The graph holds the memory, with one graph per tenant on FalkorDB or Neo4j.*

- **One graph per tenant:** All component labels for a tenant live in one graph. FalkorDB uses one graph named `m_{tenant}` per tenant, and Neo4j uses one shared database with `_tenant` on every node and edge.
- **Declared schema:** Each component declares the labels it uses along with their indexes and constraints, and the graph store combines them when it opens a tenant.
- **Graph kernel:** The kernel supplies connections and tenancy, checks schemas, retries failed calls, and trips a circuit breaker. It also provides a scoped runner for adapter Cypher and a guarded write that writes nothing when its guard fails.

#### Vector store

*Embeddings are keyed by graph ids and sit beside the graph objects or in Postgres.*

- **What is embedded:** Entity names, fact statements, and episode text, keyed by the ids of the graph objects they describe. With graph-native vectors they are properties on those objects, and pgvector keeps them in Postgres tables with HNSW indexes, an outbox, and a reconciler.
- **Hash-checked writes:** A vector is replaced only when the hash of its text changes.
- **Pinned spaces:** A registry pins the model for each kind along with its dimension and text template, and a different model on the same tenant raises a space mismatch error.
- **Honest search:** Every backend returns a normalized cosine score, and a failed search raises instead of returning an empty list.

#### Episode store

*Raw source text lives on episode nodes in the graph store.*

- **Raw source text:** An episode node in the graph store holds the content, its format and metadata, the source system and reference, the actor, and the content hash.
- **Commit state:** The episode also records its temporal status and the hash that was committed, which lets startup settle writes that a crash left pending.
- **Vector copy:** The embedding of the episode text goes to the vector store.

#### Relation store

*Service control data lives apart from memory content, in SQLite or Postgres.*

- **Control data only:** Jobs with their request payloads, operation rows with usage and cost, the hash-chained audit log, and the erase journal live here. Memory content does not.
- **Callers bring identity:** The service does not store users, API keys, or grants. The caller passes a principal with its grants on each call.
- **Backends:** SQLite is the default, and Postgres serves a deployment that shares state.

### Four-lane retrieval and prompt context

*Four lanes search the same tenant, and one budgeted context block comes back.*

<p align="center">
  <img src="https://raw.githubusercontent.com/ontogr/ontogr-agent-memory/main/img/retrieval.svg" alt="A query is embedded once. The vector, keyword, and entity lanes run at the same time, and the graph lane walks outward from seed entities that those lanes supply. Reciprocal rank fusion merges the lane results, an optional reranker rescores a shortlist of passages, and the result becomes a budgeted, escaped context block." width="860">
</p>

No single search method finds everything. Vector search finds paraphrases, keyword search finds exact terms, the entity lane finds named things, and the graph lane finds connected facts. Running all four and fusing their ranks removes the need to choose a strategy for each query.

Retrieval runs four fixed lanes. There is no query planner and no strategy choice.

| Lane | Searches | Notes |
| --- | --- | --- |
| Vector | Fact and episode vectors | Cosine search with a normalized score. |
| Keyword | Observation statements and episode text | Tenant-scoped fulltext search. |
| Entity | Explicit mentions, alias text, and entity vectors | Uses strict and loose resolver keys. |
| Graph | A bounded walk from seed entities | Seeded by the other lanes or by the request. |

Lanes fuse by reciprocal rank fusion. The top fact in one result scores 1.0, and scores are relative to that result, so they cannot be compared across queries. An optional reranker scores a shortlist of passages and blends its scores after fusion.

- **Time filters apply in every lane:** `valid_at`, `known_at`, overlap windows, and `include_history` shape all results the same way.
- **Requests can narrow the search:** A request can limit results to one session, flag or exclude disputed facts, and search several tenants the caller may read.
- **The context block is budgeted:** It has a time frame, a token budget, facts, entities, and episodes. Items are placed whole within section shares, and unused space moves to the next section. Stored text is escaped, so it cannot break the frame. The block reports how many items were dropped and which lanes failed.

A lane reports one of four statuses:

| Status | Meaning |
| --- | --- |
| `ok` | The lane ran and found something. |
| `empty` | The lane ran and found nothing. |
| `failed` | The lane raised a typed error. |
| `skipped` | The lane had no input or was not configured. |

`empty` and `failed` are never merged. A failed lane degrades the result, and the other lanes still answer. Retrieval returns evidence and context. It does not plan the query and does not write an answer.

### Consolidation

*A background run keeps one tenant's memory tidy over time.*

<p align="center">
  <img src="https://raw.githubusercontent.com/ontogr/ontogr-agent-memory/main/img/consolidation.svg" alt="One consolidation run takes a per-tenant lease and runs nine stages in order. A failure in the first stage stops the run. A watermark is stored for a stage only when it finishes without failure, and a run can request another pass while it holds the lease." width="860">
</p>

Writes arrive one episode at a time, so duplicate entities, stale summaries, and low-value items build up between runs. Consolidation handles that work in one ordered pass that runs as its own job, apart from the write path.

Consolidation runs nine stages in order for one tenant:

| Stage | What it does |
| --- | --- |
| 1. Ingest cleanup | Finishes temporal writes that a crash left pending. This stage is fatal on failure. |
| 2. Deduplication | Finds duplicate entities and runs the resolution pipeline. |
| 3. Feedback weights | Applies new feedback to streaming weights and drops neutral ones. |
| 4. Descriptions | Regenerates summaries for entities marked stale. |
| 5. Communities | Finds communities with label propagation and builds a summary hierarchy. |
| 6. Reflection | Distills session entries into lesson proposals for review. |
| 7. Scoring | Recomputes entity and current-fact importance from recency, frequency, and salience. |
| 8. Prune and archive | Archives low-score items and optionally deletes them after retention. |
| 9. Report | Collects near misses, stale summaries, disputed facts, and pending proposals. |

- **One run at a time:** A per-tenant lease allows one run at a time. The lease expires after its time to live, so a crashed worker cannot block the tenant forever. A run can request another pass while it holds the lease.
- **Watermarks are honest:** A watermark is stored only when a stage finishes without failure.
- **Feedback decays:** Weights start at 0.5, which is neutral. Each rating moves the weight part of the way toward the rating, and the weight drifts back toward neutral while the item gets no feedback.
- **Archiving is reversible:** An archived item is hidden from normal reads but is not deleted. Retention deletion is off by default.

### Access control

*Grants per tenant decide who can read, write, forget, or export.*

<p align="center">
  <img src="https://raw.githubusercontent.com/ontogr/ontogr-agent-memory/main/img/access-control.svg" alt="Each call passes a rate limit, opens an operation row, and checks the principal's grant for the tenant. The check writes an audit row for the decision. A held grant runs the verb, and a missing grant returns the same not_found error whether the tenant, the object, or the grant is missing." width="860">
</p>

Memory can hold private details. Every call is checked against the caller's grant for the tenant, and every decision is recorded in the audit log.

A principal holds grants per tenant:

| Grant | Allows |
| --- | --- |
| `read` | Search, entity reads, history, episode lists, stats, operations, and job reads. |
| `write` | Add, update, consolidate, and feedback. |
| `forget` | Invalidate and forget. |
| `share` | Reserved for delegation. No public share verb yet. |
| `export` | Tenant export. |
| `admin` | Every other grant, plus tenant clear and audit reads. |

- **Wildcard tenant:** The tenant key `*` covers every tenant.
- **No probing:** A refusal uses the same `not_found` error for a missing tenant, a missing object, and a missing grant, so a caller cannot probe tenant ids.
- **Agent principals:** An agent principal has a `parent_id`, and `Principal.is_within` checks that its grants stay within its parent's grants.
- **Audited decisions:** Every access decision is written to the configured audit sink. Durable audit rows are hash-chained, so an edit or a deletion in the middle is detectable. The chain does not stop a database owner who rewrites every later hash.

### Forget, erase, and export

*Removal is previewed, confirmed, and resumable after a crash.*

<p align="center">
  <img src="https://raw.githubusercontent.com/ontogr/ontogr-agent-memory/main/img/forget.svg" alt="A hard forget starts with a preview that returns a confirmation token. The confirmed call writes a journal row first, runs the removal steps in order while it updates the row after each one, scans for residue, closes the row as completed or failed, and writes an audit row. Startup resumes any row that is still running." width="860">
</p>

A removal that stops halfway leaves content in some stores and not in others. A hard forget therefore writes a journal row first and records each finished step, and startup resumes any unfinished row. The preview and confirmation token prevent removing the wrong target.

Forget has four modes:

| Mode | Effect |
| --- | --- |
| `invalidate` | Closes the fact now and keeps its history. |
| `delete` | Removes the observations and the stored episode. |
| `memory_only` | Removes derived facts and keeps the episode record. |
| `erase` | Deletes, then scans the graph, indexes, and model cache for residue. |

- **Dry run first:** A hard forget and a tenant clear require a preview. The preview returns a short-lived HMAC token bound to the caller, tenant, target, and mode, and the confirmed call must present it. A preview also reports collateral facts in touched slots and facts that may reopen.
- **Erase reports residue:** Erase fails when it finds residue it cannot clean and lists what it could not remove.
- **Tenant clear keeps grants:** It removes memory, vectors, and jobs, but keeps the tenant's grants.
- **Export is deterministic:** Export returns one JSONL document with a metadata row, entities, fact versions, and episodes, plus a schema version and row counts. Export is audited.

## Quickstart

In this example you store one structured fact and read it back. It uses FalkorDB for the graph and SQLite for state, so it needs a running FalkorDB server but no language model.

### 1. Install

The package installs as `ontogr-mem` and imports as `ontogr_mem`.

```bash
uv add ontogr-mem
```

The core package has no database driver. Add the extra for your backend:

```bash
uv add "ontogr-mem[falkordb]"
uv add "ontogr-mem[neo4j]"
uv add "ontogr-mem[pgvector]"
uv add "ontogr-mem[postgres]"
uv add "ontogr-mem[otel]"
```

Optional extraction extras are `gliner` and `spacy`. The full set is in [pyproject.toml](https://github.com/ontogr/ontogr-agent-memory/blob/main/pyproject.toml).

### 2. Run the example

Save the following as `example.py` and run `python example.py`. Replace `HashingEmbedder` with your embedding model for real use.

```python
import asyncio
from collections.abc import Sequence
from datetime import UTC, datetime

from ontogr_mem.relational_store.config import StateSettings
from ontogr_mem.service.bootstrap import ServiceSettings, open_service
from ontogr_mem.service.contracts import AddMemory, MemoryKind, SearchMemory
from ontogr_mem.service.principal import Grant, Principal


class HashingEmbedder:
    """A deterministic embedder for the example. Use a real model in production."""

    model_id = "demo-hash"
    dim = 8
    max_input_tokens = 8192

    async def embed_batch(self, texts: Sequence[str]) -> list[list[float]]:
        vectors = []
        for text in texts:
            vector = [0.0] * self.dim
            for index, char in enumerate(text.lower()):
                vector[index % self.dim] += ord(char) / 255.0
            length = sum(value * value for value in vector) ** 0.5 or 1.0
            vectors.append([value / length for value in vector])
        return vectors


async def main() -> None:
    settings = ServiceSettings(backend="falkordb")
    state = StateSettings(backend="sqlite")

    async with open_service(
        settings,
        state_settings=state,
        embedder=HashingEmbedder(),
    ) as runtime:
        service = runtime.service
        principal = Principal(
            id="demo",
            grants={"default": frozenset({Grant.ADMIN})},
            default_tenant="default",
        )

        receipt = await service.add_memory(
            principal,
            AddMemory(
                kind=MemoryKind.FACT,
                subject="Ana",
                relation="works_at",
                object="Acme",
                valid_from=datetime(2024, 1, 1, tzinfo=UTC),
                wait_seconds=10,
            ),
        )

        job = await service.get_job(principal, receipt.job_id)
        print(job.status, job.result)

        result = await service.search_memory(
            principal,
            SearchMemory(query="Where does Ana work?", format="both"),
        )
        print(result.state)
        print(result.rendered)


asyncio.run(main())
```

The write returns a job receipt, and `wait_seconds` makes the call wait for the job. The search result holds ranked facts, entity cards, and a rendered context block. `result.state` is one of `found`, `none`, `indexing`, or `degraded`.

A free-text or message write needs an injected `LLMClient`, because the text must first be extracted into entities and facts.

### 3. Move to durable storage

The example above already uses FalkorDB for the graph and SQLite for state. The matching environment variables are:

```bash
export MEMORY_SERVICE_BACKEND=falkordb
export MEMORY_STATE_BACKEND=sqlite
export MEMORY_FALKORDB_HOST=localhost
export MEMORY_FALKORDB_PORT=6379
```

The relation store refuses its in-memory backend unless you set `MEMORY_STATE_ALLOW_EPHEMERAL=true`. See [Configuration](https://github.com/ontogr/ontogr-agent-memory#configuration) and [.env.example](https://github.com/ontogr/ontogr-agent-memory/blob/main/.env.example) for more settings.

## Core concepts

| Term | Meaning |
| --- | --- |
| Tenant | The only partition of memory. Nothing is shared between tenants. |
| Episode | One unit of source content, such as a chat message, document part, tool output, event, or record row. |
| Source | The originating system and item id for an episode, stored as `system\|ref`. |
| Actor | The speaker or producer of an episode, with a role. It is stored apart from the content. |
| Extracted entity | A name found in one episode, before it is matched to a canonical entity. |
| Entity | The canonical thing that names and aliases point to. Facts connect entities. |
| Mention | The record that one episode mentions one entity. |
| Alias | Another name that resolves to an entity. |
| Observation | One dated claim from one episode, with a statement and a verbatim quote. |
| Slot | One subject and relation type, plus the object when the relation can have many values. |
| Fact | What the system currently believes about a relation, with a validity window. |
| Fact version | An immutable version of a fact. A change writes a new version and closes the old one. |
| Ontology | The entity types, relation types, attributes, and aliases a tenant allows. |
| Consolidation | The scheduled maintenance run over one tenant's memory. |
| Principal | The caller, with an id, grants per tenant, a default tenant, and an optional parent. |
| Grant | One permission on one tenant: read, write, forget, share, export, or admin. |
| Job | One unit of background work for one tenant. |
| Operation row | A record of a write, read, invalidate, erase, export, or history call. |
| Audit row | An entry in the hash-chained audit log. It holds ids and counts, never memory text. |

Memory moves through these steps:

```text
source
  -> episode
  -> extracted entities -> canonical entities -> mentions and aliases
  -> observations -> fact versions -> retrieval -> context block
  -> consolidation -> summaries, communities, scores, archives
```

First, ingestion stores a source item as an episode. Second, entity extraction finds names, and entity resolution links them to canonical entities. Third, fact extraction finds grounded claims and their valid times, and temporal reconciliation turns the observations into fact versions. Fourth, retrieval searches facts, episodes, and entities and builds a context block. Finally, consolidation runs in the background.

## How data flows

### A write

1. The service checks the caller's rate limit, opens an operation row in the relation store, and checks the write grant for the tenant. Each access decision is appended to the audit log.
2. The service stores the request as a job in the relation store and returns a receipt with the job id. A repeated idempotency key returns the original job, and `wait_seconds` makes the call wait for the job to finish.
3. The tenant's job lane runs the job. A lane runs one job at a time for its tenant, and lanes for different tenants run in parallel. The job records each attempt.
4. Ingestion writes the episode with its source and actor to the graph, and records the content hash last.
5. For a text or message write, extraction finds entities, and resolution links them to canonical entities in the graph. Resolution also writes the entity name vectors. A structured fact skips extraction.
6. Fact extraction returns claims with quotes and valid times, and the ontology maps each relation to a stable type.
7. The temporal engine marks the episode pending, writes the observations, and reconciles each affected slot into fact versions in the graph.
8. After each slot commit, the fact vectors are written. A fact is in the graph before any vector points at it.
9. The engine appends a ledger entry, and then marks the episode committed with its content hash.
10. The service writes the episode metadata and the episode vector, marks the job done, and closes the operation row with its outcome, token use, and cost.

Each step is safe to run again after a crash:

- Ingestion records the content hash last, so an episode without a hash is written again.
- The engine marks an episode pending before it writes observations, and appends the ledger entry before the committed mark. At startup, the service settles episodes left pending and commits them.
- At startup, the job runner puts running jobs back in the queue. A replayed job finds the episode already committed with the same hash and skips it.
- A failed vector write does not roll back the fact. The writer keeps the failed item for a retry.
- With pgvector, vector work goes through an outbox that a reconciler drains, as described in the Storage layer.

### A read

1. The service checks the caller's rate limit, opens an operation row, and checks the read grant for each tenant in the request. Each decision is audited.
2. The retriever embeds the query once.
3. The vector, keyword, and entity lanes run at the same time. The vector lane searches fact and episode vectors, and the other two lanes search the graph.
4. The graph lane walks outward from seed entities taken from the other lanes.
5. Reciprocal rank fusion merges the lane results. An optional reranker, the ranking penalties, and the evidence lookup follow.
6. The retriever builds the context block within the token budget, and the service closes the operation row with its usage.

A read does not create a job. A search that finds nothing reports an `indexing` state while jobs for the searched tenants are still queued or running, so an empty result can be told apart from a write that has not finished.

### A hard forget

1. The service checks the forget grant and verifies the confirmation token from the preview.
2. The workflow writes a row to the erase journal in the relation store.
3. The workflow removes temporal evidence, then vectors, entity text, cached model responses, and episodes, and updates the journal row after each step. An erase also scans for residue.
4. The workflow closes the journal row as completed, or as failed with the residue it found. The service appends an audit row for the result.
5. After a crash, startup resumes every journal row that is still running. Audit rows and operation rows stay after a forget.

## Service API

`MemoryService` is the async facade for every verb. It applies access checks, rate limits, operation records, and tenant rules.

### Write verbs

| Verb | Grant | Behavior |
| --- | --- | --- |
| `add_memory` | write | Queues a fact, text, or message write. |
| `update_memory` | write | Closes one current fact and asserts its replacement. |
| `invalidate` | forget | Closes one current fact and keeps its history. |
| `consolidate_memory` | write | Queues a consolidation run. |
| `memory_feedback` | write | Records a rating on an entity or a fact. |

### Read verbs

| Verb | Grant | Behavior |
| --- | --- | --- |
| `search_memory` | read | Searches one or more tenants and returns facts, entities, and context. |
| `get_entity` | read | Returns one entity and its current or historical facts. |
| `entity` | read | Returns an entity card with aliases and facts. |
| `graph` | read | Returns a bounded neighbourhood around seed entities. |
| `list_episodes` | read | Pages through episode metadata, newest first. |
| `history` | read | Returns every visible version of one fact's slot. |
| `stats` | read | Returns tenant counts and usage. |
| `operations` | read | Lists recent operation rows. |
| `get_job` | read | Returns one job after checking its tenant grant. |
| `get_status` | read | Reports service health and visible tenants. |

### Forget and export verbs

| Verb | Grant | Behavior |
| --- | --- | --- |
| `preview_forget` | forget | Returns a forget plan and a short-lived confirmation token. |
| `forget` | forget | Invalidates a fact or confirms a hard forget. |
| `forget_memory` | forget | Applies the request contract to a forget selector. |
| `preview_clear_tenant` | admin | Returns a tenant clear plan and its confirmation token. |
| `clear_tenant` | admin | Removes every episode, fact, entity, vector, and job for a tenant. |
| `audit_log` | admin | Lists recent audit rows. |
| `export` | export | Returns the tenant as deterministic JSON lines. |

### Jobs

Writes return a `WriteReceipt` with a job id and status. A positive `wait_seconds` waits for completion. Jobs run in serial lanes per tenant, and different tenants run in parallel. Each write accepts an optional idempotency key, and a repeated key returns the original job instead of creating duplicate work. A job records attempts, progress, result, error, and trace id, and the runner retries up to a configured maximum.

### Errors

Service errors carry a stable code, a message, and an optional hint:

| Code | Meaning |
| --- | --- |
| `invalid_argument` | A field is missing, malformed, or out of range. |
| `not_found` | The tenant or object does not exist, or the caller cannot see it. |
| `conflict` | The request collides with current state. |
| `llm_unavailable` | A verb that needs a model was called without one. |
| `rate_limited` | The caller spent its token budget for the minute. |
| `payload_too_large` | A field is over its byte limit. |
| `confirmation_required` | A destructive verb is missing its confirmation token. |
| `internal` | An unexpected failure inside the service. |

The default limits are 60 writes and 600 reads per caller per minute, 64 KiB per content field, and 100 results per read. The limiter is per process.

### Using components directly

`MemoryService` is the high-level path, but it is not required. Each component defines an async port and takes its collaborators by constructor. The caller can open a tenant, build the adapters, and pass them in. For example, retrieval can run without the service:

```python
from ontogr_mem.retrieval import RetrievalRequest, Retriever
from ontogr_mem.temporal import FactReader


async def read_memory(query, now, *, vectors, keywords, temporal, entities, episodes):
    retriever = Retriever(
        vectors=vectors,
        keywords=keywords,
        facts=FactReader(temporal),
        entities=entities,
        episodes=episodes,
    )
    return await retriever.retrieve(
        RetrievalRequest(query=query, now=now, limit=5, budget_tokens=1200)
    )
```

`now` must include a timezone. The result holds ranked facts and episodes, entity cards, lane records, and `context.text` for an answering model. Pass `count_tokens` when you need a model-specific budget. The default estimate uses four characters per token.

## Language model and embedding layer

The package ships no provider client. You supply an `LLMClient` and an `Embedder`. `LLMGate` is the path for every model call, and it provides:

- model tiers (`small` and `medium`) with separate concurrency limits
- structured output parsing and a schema repair loop
- input-size and deadline checks
- a response cache with in-memory and SQLite implementations, and tag-based eviction
- pricing and usage accounting
- a circuit breaker and retry policy for provider failures
- fenced prompts for untrusted text

The embedding gateway batches texts, splits long inputs, retries transient failures, and enforces a request deadline. A failed embedding is returned with a typed reason, and the vector writer keeps failed writes for an explicit retry. A committed fact or episode is not rolled back when its vector write fails.

## Observability

| Facility | What it records |
| --- | --- |
| Spans | Timed steps for writes, retrieval, erase, export, forget, history, model calls, and auth checks. |
| Tracers | No-op, in-memory, and OpenTelemetry implementations. |
| Metrics | Plain counters and latency observations, exported by the caller. |
| Operation rows | One row per write, read, invalidate, erase, export, or history call, with timing, outcome, error summary, and usage. |
| Audit rows | Access decisions and destructive operations, linked by a hash chain. |
| Request context | Principal, tenant, trace id, session, operation id, latency budget, usage, and cache tags for one call. |

Trace attributes are redacted and size limited. Audit rows never hold memory text. `stats` returns tenant counts and usage without creating an operation row.

## Configuration

Settings classes use `pydantic-settings`. They read the repo-root `.env` and use an environment prefix per area:

| Prefix | Area |
| --- | --- |
| `MEMORY_SERVICE_` | Service graph backend. |
| `MEMORY_STATE_` | Relation store backend for jobs, operations, and audit rows. |
| `MEMORY_FALKORDB_` | FalkorDB connection and resilience. |
| `MEMORY_NEO4J_` | Neo4j connection and resilience. |
| `MEMORY_PGVECTOR_` | pgvector connection, search, and outbox limits. |
| `MEMORY_VECTORSTORE_` | Embedding batches, retries, timeouts, and vector indexes. |
| `MEMORY_RETRIEVAL_` | Candidate counts, fusion constants, penalties, and context shares. |
| `MEMORY_KEYWORD_` | Tenant-scoped fulltext search limits. |
| `MEMORY_RESOLUTION_` | Entity resolution thresholds and verification budgets. |
| `MEMORY_TEMPORAL_` | Temporal retries, batch size, and episode pending time. |
| `MEMORY_JUDGE_` | Conflict judge threshold, model tier, and token budget. |
| `MEMORY_CONSOLIDATION_` | Consolidation leases, feedback, scoring, communities, and retention. |
| `MEMORY_ONTOLOGY_` | Proposal support, retries, and prompt version. |
| `MEMORY_ENTITY_EXTRACTION_` | Entity extractor concurrency, context, and thresholds. |
| `MEMORY_FACT_EXTRACTION_` | Fact extractor concurrency, context, and version. |
| `MEMORY_LLM_` | Model concurrency, retries, and breaker settings. |
| `MEMORY_FORGET_` | Forget confirmation secret. |
| `MEMORY_` | Local stdio principal id, tenant, and grants. |

Nested settings join their names with `__`, for example `MEMORY_FALKORDB_RESILIENCE__MAX_ATTEMPTS`. See [.env.example](https://github.com/ontogr/ontogr-agent-memory/blob/main/.env.example) for the test-service values.

## Development

You need Python 3.11 or newer and [uv](https://docs.astral.sh/uv/).

```bash
make sync
uv run pre-commit install
make test
```

| Command | Description |
| --- | --- |
| `make sync` | Install all dependencies. |
| `make test` | Run unit tests with coverage. |
| `make test-integration` | Run integration tests against the test services. `INTEGRATION_PATHS=<paths>` picks modules and `INTEGRATION_WORKERS=<n>` sets the number of workers. |
| `make test-all` | Run unit and integration tests. |
| `make lint-check` | Check formatting and lint without modifying files. |
| `make lint-typing` | Type check with `ty`. |
| `make lint-all` | Format, apply lint fixes, type check, check typos, and audit workflows. |
| `make security` | Run Bandit and a dependency audit. |
| `make services-up` | Start the FalkorDB, Neo4j, pgvector, and Postgres test services. `SERVICES=<names>` starts only those. |
| `make services-down` | Stop the test services and remove their volumes. |
| `make build` | Build the sdist and wheel. |
| `make wheel-test` | Build the wheel and import it in a clean environment. |

Unit tests do not touch the network. Integration tests use real services and can read real environment variables or `.env`. Run `make services-up` first. A test fails, and is not skipped, when a service it needs is not running.

## Contributing

We welcome bug fixes, new features, documentation improvements, and other contributions. Please read the [Contributing guide](https://github.com/ontogr/ontogr-agent-memory/blob/main/CONTRIBUTING.md) before you submit a pull request. Rules for coding agents are in [AGENTS.md](https://github.com/ontogr/ontogr-agent-memory/blob/main/AGENTS.md).

## License

OntoGr Agent Memory is licensed under the [Apache License 2.0](https://github.com/ontogr/ontogr-agent-memory/blob/main/LICENSE).
