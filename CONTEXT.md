# Agent Memory

Vocabulary for this project's domain. Add one entry per term, in the form below, and link terms with `[[Term]]`.

## Language

**Tenant**:
The only partition of memory. Nothing is shared between tenants.
_Avoid_: scope, group, group_id, namespace, dataset

**Episode**:
One unit of source content (a chat message, a document chunk, a tool output, an event or a record row) that has been given a stable id. It is the unit that is extracted from, embedded and cited, and there is no smaller stored part.
_Avoid_: chunk, message, record (as the general input unit)

**Episode store**:
The port that stores [[Episode]]s with their [[Source]]s and [[Actor]]s for one [[Tenant]], and reads the [[Previous episodes]].
_Avoid_: GraphStore (that is the generic graph storage underneath)

**Episode metadata**:
The session, thread, subject reference and fact ids of an [[Episode]], stored in the [[Tenant]] graph as one `EpisodeMeta` node per episode.
_Avoid_: episode extras (for these typed fields), episode metadata dict (the episode's free-form extras)

**Job**:
One unit of background work for one [[Tenant]], run by the job runner.
_Avoid_: task (for a stored job)

**Job lane**:
The jobs of one [[Tenant]], run one after another. Different tenants run in parallel.
_Avoid_: lane (alone, it means a [[Retrieval]] lane)

**State store**:
The port that keeps jobs and idempotency keys outside the graph. It also records erase progress for recovery. It is separate from the [[Episode store]]. It lives in `ontogr_mem/relational_store/` with the audit chain and operation rows.
_Avoid_: episode store, graph store (for job state)

**Source**:
Where an [[Episode]] came from: the originating system and the item's id in it. Its id is `system|ref`.

**Actor**:
The speaker or producer of an [[Episode]], with a role. Kept apart from the content, never written as a prefix in the text.

**Previous episodes**:
The latest episodes of the same [[Tenant]] that occurred before an episode. Extraction reads their text as context for the episode, but never quotes it as evidence.

**Extracted entity**:
A name found in one [[Episode]] by extraction, with a type. It has not yet been matched to an [[Entity]], and it is not stored.
_Avoid_: mention group, candidate

**Mention**:
A recorded link between one [[Episode]] and the [[Entity]] that a name in it refers to.
_Avoid_: mention group

**Entity**:
The thing that extracted entities and aliases point to. Facts connect entities.
_Avoid_: node

**Alias**:
Another name that resolves to an [[Entity]].
_Avoid_: synonym, surface form

**Extracted fact**:
One claim found in one [[Episode]] by extraction, using local entity names. It is not stored, and the accepted ones become [[Observation]]s.
_Avoid_: statement, claim, triple

**Typed fact**:
An [[Extracted fact]] whose relation is given as a stable relation type id, not as a position in one ontology projection. It has no relation type id when its relation cannot be used. It records the catalog version of the projection that was used.

**Observation**:
One dated claim about a relation between two [[Entity]]s, taken from one [[Episode]]. It is the stored record of an extracted fact and the only input to reconciliation.

**Slot**:
The unit in which [[Observation]]s are reconciled: one subject and one relation type, and also the object when the relation can have many values.

**Fact**:
What the system currently believes about a relation between two [[Entity]]s, with a validity period, built from one or more [[Observation]]s.
_Avoid_: statement, claim, triple

**Deletion**:
Removing an [[Episode]] and what hangs on it from the graph.
_Avoid_: removal (when retraction or reset is meant)

**Retraction**:
Marking an [[Episode]] as withdrawn, keeping its node and writing a ledger entry, so the [[Fact]]s built from it are closed in history and not erased.
_Avoid_: deletion (for this)

**Reset**:
Clearing what extraction and reconciliation produced for an [[Episode]] (its [[Observation]]s and [[Mention]]s and its processing status) so it can be ingested again. The node stays, the facts that rested on the removed observations are re-derived, and the episode is not marked retracted. The caller asks for it.
_Avoid_: retraction, re-run (as a name for the operation)

**Ontology**:
The entity types, relation types and attributes that a [[Tenant]] allows, together with its version history.
_Avoid_: schema, catalog (for the full version history)

**Ontology mode**:
What happens to a type that is not in the [[Ontology]]: it is dropped (strict), recorded for review (preferred) or registered at once (open).

**Proposal**:
A set of additions to the [[Ontology]] drafted from labels that extraction could not place and that appeared in enough [[Episode]]s. A person accepts or rejects it; it is never applied automatically.
_Avoid_: suggestion, candidate type

**Unmatched row**:
A record that an [[Episode]] used a type label the [[Ontology]] does not hold. It is written only in preferred mode, and a new extraction of the episode replaces its rows.

**Ontology store**:
The port that stores a [[Tenant]]'s [[Ontology]] versions, [[Unmatched row]]s and [[Proposal]]s. A commit moves the head only if the head is still the version the change started from.
_Avoid_: catalog store

**Deduplication**:
Finding [[Entity]]s that are the same thing and merging them, with a reversible record of each merge.
_Avoid_: consolidation (for this step), dedup

**Entity verdict**:
A recorded answer that two [[Entity]]s are the same, distinct or uncertain.
_Avoid_: pair verdict, verdict

**Fact verdict**:
A recorded judgement of whether two [[Fact]]s contradict each other, are compatible, refine one another, duplicate one another or are unrelated, or whether the judgement is uncertain.
_Avoid_: pair verdict, verdict

**Conflict**:
A pair of [[Fact]]s in a single-valued or judged relation that cannot both stand, with its class (update, correction, dispute or uncertain) and the disposition the handler chose. A conflict is recomputed on every reconcile; its open, resolved or dismissed state is computed from the [[Decision]]s and the facts that still exist.
_Avoid_: contradiction (for the stored record), clash, disagreement

**Decision**:
A person's recorded answer to a [[Conflict]]: choose one fact, mark one fact wrong, mark both valid, or mark the pair compatible. A decision overrides every rule and every [[Fact verdict]] and survives a re-run.
_Avoid_: resolution, ruling, override, judgement

**Lesson verdict**:
The reflection writer's decision on a proposed lesson: accept, already known, not durable or unsupported.
_Avoid_: verdict

**Consolidation**:
The scheduled maintenance job that runs a fixed sequence of nine stages over one [[Tenant]]'s memory.
_Avoid_: sleep job (informal), dedup tier

**Stage**:
One step of a [[Consolidation]] run. A stage may be skipped by its gate, and exactly one stage per run is fatal.

**Lease**:
The per-[[Tenant]] lock that lets one [[Consolidation]] run at a time. A lease expires after its time to live, so a crashed worker cannot block the tenant forever.

**Watermark**:
The record of how far a [[Stage]] has processed. It is stored only when the stage finishes without failure.

**Community**:
A group of [[Entity]]s that are densely connected, with a summary. Label propagation finds the groups, and a community needs a minimum number of members.

**Bucket**:
A node in the summary hierarchy built above the [[Community]] entries. Each bucket has a level and a fixed maximum number of children.

**Archive record**:
The record that an [[Entity]] or [[Fact]] was archived, with the reason and the time. The source item is kept until the retention period ends, and an archived item can be restored.

**Feedback weight**:
The running feedback score of one memory item, from 0 to 1. It starts at 0.5, which is neutral. Each rating moves it part of the way toward the rating, and it drifts back toward neutral while the item gets no feedback.

**Port**:
The interface that a component defines for the storage it needs, implemented by an adapter for each backend.

**In-memory twin**:
A component port implementation that keeps its data in memory, used in tests and run through the same contract suite as the database adapters. The low-level [[Graph store]] has no in-memory implementation.

**Contract suite**:
One set of tests for a [[Port]], run against every implementation of it.

**Database mode** and **property mode**:
The two ways Neo4j separates tenants: one database for each [[Tenant]] (database mode), or one shared database where every node carries a tenant property (property mode).

**Graph store**:
The shared graph adapter package includes backend-neutral execution code and component port adapters. It is not a kernel or a [[Port]] of any one component. It opens one graph for each [[Tenant]], writes and reads nodes and edges by `id`, and offers a [[Scoped runner]] and the [[Guarded write]]. The execution code does not import component packages.
_Avoid_: layer 11, database client

**Scoped runner**:
The object a component adapter uses to run its own Cypher against one [[Tenant]]'s graph. In property mode, adapters must include the tenant scope in their patterns. Supplying a tenant parameter alone does not isolate a query.
_Avoid_: raw query, public run

**Guarded write**:
One all-or-nothing write that first checks properties of a guard node, then updates the guard and writes nodes and edges. It returns false and writes nothing when the check fails. Concurrent callers with the same expectation have exactly one winner.
_Avoid_: update_if_rev, compare and swap (as a method name)

**Schema part**:
The node labels, indexes, constraints and edge keys that one component needs. The graph store combines the parts of the components it serves when it opens a [[Tenant]]'s graph.
_Avoid_: ontology (the [[Ontology]] is tenant data, not storage layout)

**Retrieval**:
Reading memory for one query by searching several [[Lane]]s and fusing their rankings. It returns ranked [[Fact]]s, ranked [[Episode]]s, [[Entity card]]s and one [[Context block]]. It does not plan the query or write an answer.
_Avoid_: search (for the whole operation), recall, query engine

**Keyword search**:
A tenant-bound [[Port]] that searches alias names, observation statements and quotes, or episode text by tokens. Its scores are local to the backend, so [[Retrieval]] combines ranks rather than comparing raw scores.
_Avoid_: retrieval (for this one lane), substring search

**Fact reader**:
The temporal reader that applies valid-time and system-time filters to [[Fact]] versions and adds evidence text, dispute counterparts and decisions.
_Avoid_: retriever, graph store

**Entity card**:
A resolved [[Entity]]'s name, type, aliases and summary in a [[Retrieval]] result. It provides context for the facts and is not a separate retrieval target.
_Avoid_: entity result, summary fact

**Reranker**:
An optional [[Port]] that scores a shortlist of passages against the query after lane fusion. [[Retrieval]] blends the scores before applying fact-status penalties.
_Avoid_: planner, answer model

**Vector writer**:
The component that writes fact vectors after a slot commit and episode vectors after ingest. It retains failed writes for an explicit retry without undoing the committed data.
_Avoid_: vector store, automatic retry

**Lane**:
One way of searching inside a [[Retrieval]]: vector, keyword, entity or graph. A lane returns a ranked list of ids and a status.
_Avoid_: channel, method, path, retriever

**Lane status**:
The outcome of one [[Lane]]: ok, empty (it ran and found nothing), failed (it raised a typed error) or skipped (not configured or no input). Empty and failed are never merged.

**Fused score**:
The score of a retrieved [[Fact]]: the rank-fusion value divided by the best value in the same result, so the top fact is 1.0. It is relative to one result and is not comparable across queries.
_Avoid_: relevance, confidence, similarity (a vector similarity is a different number)

**Context block**:
The text a [[Retrieval]] assembles for a model: a frame with the as-of time and the token budget, then facts, entities and episodes placed whole within shares of the budget, a count of what was dropped and a note of any failed [[Lane]].
_Avoid_: prompt, snippet list

**Invalidation**:
Closing one [[Fact]] at a stated time while keeping its history, so an as-of read can still see it. It acts on one Fact, not on an [[Episode]].
_Avoid_: retraction (for this), forget, delete

**Erase**:
A [[Deletion]] followed by a scan for content that remains in the graph, the indexes and the model cache. It fails when it finds residue and lists what it cannot clean.
_Avoid_: forget, purge

**Principal**:
Who is calling a verb, with an id, [[Grant]]s per [[Tenant]], a default tenant and an optional parent.
_Avoid_: user, caller (when it means the principal), key

**Grant**:
One permission on one [[Tenant]]: read, write, forget, share, export or admin, with `*` standing for every tenant. Admin implies the other five.
_Avoid_: permission, role, scope, namespace

**Authorizer**:
The component that checks a [[Principal]]'s grants before an action and records the decision.
_Avoid_: permission checker

**Audit row**:
One entry of the [[Audit chain]]: an action, its outcome, the principal, the [[Tenant]], and ids and counts. It never holds memory text.
_Avoid_: audit log entry, log line

**Audit chain**:
The audit rows linked by hash, so that an edit or a deletion in the middle is found by `verify`. It does not stop an owner who rewrites every later hash.
_Avoid_: audit log (for the table), ledger (for this)

**Operation row**:
One record of a write, retrieve, invalidate, erase, export or history, with its timing, outcome, error summary and usage.
_Avoid_: job (for this), run, span

**Request context**:
The values of one call held per task: principal, [[Tenant]], trace id, session, operation id, latency budget, usage and cache tags.
_Avoid_: session state, global context

**Memory service**:
The one async service that applies access checks and memory rules for every verb. See [one memory service](thoughts/shared/decisions/0069-one-memory-service-with-thin-doors.md).
_Avoid_: interface, platform service

**Tenant stack**:
The stores and ports for one [[Tenant]] on one backend.
_Avoid_: store bundle, runtime (for these stores)

**Confirm token**:
A short-lived HMAC that binds a hard forget or [[Clear]] to its preview.
_Avoid_: key, password

**Collateral Fact**:
A current [[Fact]] in a slot that a forget operation touches.
_Avoid_: neighbour, related fact

**Reopened Fact**:
A closed [[Fact]] that a forget operation would make current again.
_Avoid_: restored fact

**Clear**:
Admin-only removal of every [[Episode]], [[Fact]], [[Entity]] and vector for one [[Tenant]].
_Avoid_: reset, wipe, purge

**Model cache**:
A stored model completion keyed by the [[Tenant]] and request, which can be evicted by tag or text.
_Avoid_: prompt history

**Circuit breaker**:
A limit that refuses calls to a failing model provider until its recovery period ends.
_Avoid_: retry policy

**Span**:
One timed step of a memory operation, with redacted and size-limited attributes.
_Avoid_: event, trace (alone)

**Tracer**:
The port that opens spans, with no-op, in-memory and OpenTelemetry implementations.
_Avoid_: telemetry client

**Trace id**:
The id that links an operation row, a job and log lines to one request.
_Avoid_: request id, correlation id

**Job progress**:
The stage, completed count and total count of a running [[Job]].
_Avoid_: status percent

**Fact history**:
The versions of a [[Fact]] read at a valid time and a known time.
_Avoid_: change log

**Neighbourhood**:
A bounded set of [[Entity]]s and [[Fact]]s around the requested seeds.
_Avoid_: graph dump

**Tenant export**:
A JSONL record of one [[Tenant]]'s stored memory.
_Avoid_: backup

**Evaluation case**:
One set of sessions and questions run on a fresh [[Memory service]].
_Avoid_: test case

**Gate**:
A threshold that a report must meet to pass.
_Avoid_: check (for a report threshold)

**Internal suite**:
A generated, seeded set of update, isolation or erasure checks.
_Avoid_: dataset suite

**Dataset suite**:
A benchmark loaded from a local file.
_Avoid_: internal suite

**Agent principal**:
A [[Principal]] with a `parent_id`, created by another principal. Its grants are a subset of its parent's.

**Untrusted data**:
Text from memory that a model may read but must never follow as an instruction. Every text field of a result is data, already escaped inside the [[Context block]].
