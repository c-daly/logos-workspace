# Cognitive Loop — What It Does and Doesn't Do

An honest accounting of the cognitive loop feature, covering the initial implementation (logos #490/#491, sophia #125/#127, hermes #82/#84) and the performance sprint (sophia #128/#131, hermes #85, apollo #161, logos #492). Updated 2026-05-30 against the 2026-05-29 code-grounded audit.

> **Scope note.** This documents the *ingestion* arc of the loop — extract → store
> → retrieve context → enrich the LLM prompt. That arc is built and runs, but it
> is **not** the full cognitive loop the vision describes: there is no autonomous
> reasoning, feedback does not mutate the graph, and Sophia runs request/response
> rather than as a persistent event-driven process. Building the rest is the work
> of the six MTS sprints (`docs/plans/2026-05-29-roadmap-to-mts.md`).
>
> **Keystone caveat (read before trusting "embeddings written to Milvus" below).**
> The audit found embeddings **silently fail to persist** — the `hcg_*_embeddings`
> Milvus collections read 0 in the live flow. Sophia swallows the write failure in
> a warn-only `try/except` (`sophia/src/sophia/ingestion/proposal_processor.py:519-524`),
> and the logos Milvus sync is insert-only against an `auto_id` PK
> (`logos/logos_hcg/sync.py:155,315,365`), so re-syncs append duplicates instead of
> upserting. With no retrievable vectors, the type classifier degrades to the
> fallback type and #505 emergence loads 0 members. This is the Sprint-1 keystone
> (sophia#146 depends-on logos#528). Where this document says an embedding is
> "written to Milvus" as a working side effect, treat it as *attempted but
> currently a no-op* until that fix lands.

## The End-to-End Flow

```
User sends message
    │
    ▼
Hermes POST /llm
    │
    ├─ 1. Extract last user message text
    │
    ├─ 2. ProposalBuilder.build(text)
    │     ├─ spaCy NER: extract named entities
    │     ├─ For each entity: generate 384-dim embedding (all-MiniLM-L6-v2)
    │     │   └─ Side effect: each embedding written to Milvus
    │     ├─ Generate document-level embedding of full text
    │     │   └─ Side effect: written to Milvus
    │     ├─ spaCy RelationExtractor: extract relations between entities
    │     │   └─ Verb-to-relation mapping (e.g. “uses” → USES)
    │     ├─ For each relation: generate edge embedding from phrase
    │     │   └─ “Paris located in France” → 384-dim embedding
    │     └─ Build proposed_edges list (source, target, relation, embedding)
    │
    ├─ 3. POST /ingest/hermes_proposal → Sophia
    │     │
    │     ▼
    │   Sophia ProposalProcessor.process()
    │     ├─ Search Milvus with document embedding → relevant_context (top 10)
    │     ├─ For each proposed node:
    │     │   ├─ Search Milvus for similar entity (L2 < 0.5)
    │     │   │   ├─ Match found → skip creation, add existing node to context
    │     │   │   └─ No match → MERGE node into Neo4j, store embedding in Milvus
    │     │   └─ Return stored_node_ids + relevant_context
    │     ├─ For each proposed edge:
    │     │   ├─ Resolve source/target names → UUIDs (with Neo4j fallback)
    │     │   ├─ Create reified edge node in Neo4j
    │     │   └─ Store edge embedding in Milvus "Edge" collection
    │     │
    │     └─ Side effect: enqueue FeedbackPayload to Redis
    │
    ├─ 4. Build context system message from relevant_context
    │     └─ "Relevant knowledge from memory:\n- Paris (location): ..."
    │
    ├─ 5. Insert system message before last user message
    │
    └─ 6. Call LLM with enriched message list
         └─ Return response to user
```

**This flow is fully functional** when all services are running (Hermes, Sophia, Neo4j, Milvus, and optionally Redis).

## What Works

### Hermes Side

| Component | Status | Detail |
|-----------|--------|--------|
| `ProposalBuilder.build()` | Working | Runs spaCy NER, generates embeddings, builds proposal dict |
| `_get_sophia_context()` | Working | HTTP POST to Sophia with 10s timeout, graceful fallback to `[]` on any error |
| `_build_context_message()` | Working | Formats Sophia's context into a system message, filters out provenance keys |
| Context injection | Working | System message inserted at correct position (before last user turn) |
| Graceful degradation | Working | If Sophia is down, no token configured, or any error occurs — LLM call proceeds without context |
| `RelationExtractor` | Working | spaCy dependency parsing extracts verb-mediated relations between NER entities. Configurable via `RELATION_EXTRACTOR` env var |
| `EmbeddingProvider` protocol | Working | Pluggable embedding generation. Default: SentenceTransformerProvider (all-MiniLM-L6-v2). Configurable via `EMBEDDING_PROVIDER` and `EMBEDDING_MODEL` env vars |
| Parallel pipeline | Working | NER extraction, embedding generation, and proposal building run in parallel (hermes #85) |
| Redis context cache | Working | Context responses cached in Redis to avoid redundant Sophia round-trips (hermes #85) |

### Sophia Side

| Component | Status | Detail |
|-----------|--------|--------|
| `POST /ingest/hermes_proposal` | Working | Accepts proposals, dispatches to ProposalProcessor, returns 201 |
| `ProposalProcessor.process()` | Working | Dedup via Milvus similarity search, Neo4j node creation, context retrieval |
| HCG Client `add_node()` | Working | `MERGE` with SHACL validation, JSON-encoded nested properties |
| HCG Client `add_edge()` | Working | Reified edge pattern with `MERGE` on (source, target, relation) |
| HCG Client `query_neighbors()` | Working | Bidirectional traversal through reified edges |
| Milvus dedup (ENTITY_MATCH_THRESHOLD=0.5) | Working | L2 distance check, entities below threshold are skipped |
| Context retrieval | Working | Searches 5 Milvus collections (Entity, Concept, State, Process, Edge), top 10 by L2 score |
| Feedback dispatch to Redis | Working (transport only) | Enqueues FeedbackPayload via LPUSH, worker dequeues via BRPOP. **The payload then goes nowhere** — `/feedback` is a stub (see below); making feedback mutate the graph is Sprint 2 |
| Feedback worker | Working | Retries with exponential backoff, max 5 attempts, dead-letter queue |
| Inter-service event bus (`logos_events`) | Working, on main | Centralized Redis pub/sub (logos #519) carries **real** inter-service traffic — Sophia publishes ontology events, Hermes subscribes for runtime type sync (sophia #501). Earlier docs claimed the bus carried no inter-service traffic; that is no longer true |
| Maintenance scheduler | Wired, on main | sophia #508 — runs as a process, but the reasoning jobs it would dispatch (#503/#504/#506) are unbuilt, so it currently does ~nothing |
| Edge processing | Working | Resolves entity names to UUIDs, creates reified edge nodes in Neo4j, stores edge embeddings in Milvus "Edge" collection |
| Experiment tracking | Working | Creates experiment_run nodes with PRODUCED edges to track pipeline configuration and outputs |
| Batch proposal processing | Working | Proposals are batched and processed in parallel (sophia #131), reducing per-proposal overhead |
| Async proposal dispatch | Working | Proposal processing is async with reduced loop overhead (sophia #128) |
| Type classification via centroids | Working | Sophia owns type semantics — classifies entities by embedding distance to type centroids (sophia #129, logos #494). Replaces Hermes-side classification |

### Logos Foundry Side

| Component | Status | Detail |
|-----------|--------|--------|
| HCG data models | Complete | Goal, Plan, PlanStep, Capability, Fact, Association, Rule, Provenance |
| HCGPlanner | Complete | Backward-chaining planner over the graph, depth-limited, greedy |
| Type hierarchy | Complete | CWM types (cwm_a, cwm_g, cwm_e), node types (entity, concept, state, process, location, object, action, goal) |
| HCGMilvusSync | Complete | 5 collections including Edge, L2 metric, IVF_FLAT index, upsert/search/delete |
| HCGSeeder | Complete | Full type hierarchy + demo scenario + persona diary |
| SHACL validation | Complete | Validates uuid, name, is_type_definition, type, ancestors on all nodes |
| Configurable embedding dim | Complete | `LOGOS_EMBEDDING_DIM` env var (default 384) with error-handling fallback |

## What Doesn't Work / Is Stubbed

### Hermes `/feedback` endpoint — Dead letter box

The endpoint accepts a rich schema (`FeedbackPayload` with `correlation_id`, `state_diff`, `step_results`, `node_ids_created`) but **does nothing with it**. It logs the payload and returns `{"status": "accepted"}`. No state is updated, no routing occurs, no learning happens.

The feedback worker in sophia does successfully POST to this endpoint, but the information goes nowhere.

### Provider/model metadata — Always "unknown"

`ProposalBuilder.build()` accepts `llm_provider` and `model` parameters, but `_get_sophia_context()` calls it without passing the actual provider/model from the LLM request. Every proposal arrives at Sophia with `llm_provider: "unknown"` and `model: "unknown"`.

### No embedding deduplication in Hermes

Every `/llm` call generates `n_entities + 1` embeddings in memory (one per NER entity + one for the full document). These are sent as payload in the proposal to Sophia — Hermes does not write to Milvus directly. However, there is no cache or dedup on the Hermes side: if a user mentions "Paris" in 100 messages, Hermes generates a fresh embedding for "Paris" each time and sends 100 proposals. Sophia's ProposalProcessor handles dedup (L2 < 0.5 = skip), so duplicate nodes aren't created, but the redundant embedding generation and HTTP round-trips add unnecessary latency.

### No auth on `/ingest/hermes_proposal`

Every other write endpoint in Sophia requires `Authorization: Bearer <token>`. The proposal ingestion endpoint is deliberately unauthenticated (noted as intentional for local dev). Any process that can reach Sophia's port can inject nodes into the knowledge graph.

### FeedbackConfig default URL is wrong

`SOPHIA_FEEDBACK_HERMES_URL` defaults to `http://localhost:18000` but Hermes runs on port `17000`. Without explicitly setting this env var, feedback delivery fails silently (retried 5 times, then moved to `sophia:feedback:failed` Redis list).

### Integration test queries wrong property

`test_ingest_proposal_nodes_retrievable` queries Neo4j for `{id: proposal_id}` but the ProposalProcessor stores nodes with property `uuid`, not `id`. The test always finds 0 results and passes vacuously — it doesn't actually verify anything.

## What's Not Implemented (Pending in Task Queue)

The original task queue (from PR #490) listed several pending items. Current status:

| Task | Description | Status |
|------|-------------|--------|
| 00a | `RedisConfig` in logos_config/settings.py | Not done — sophia hardcodes Redis URL |
| 00b | Redis port in `RepoPorts` | Not done |
| 01a | Hermes `ProposalBuilder` | Done (hermes PR #82) |
| 02a | Extended `HermesProposalRequest` fields | Partially done — no relationship field |
| 02b | Sophia `EmbeddingStore` | Done via HCGMilvusSync |
| 02c | Sophia `ProposalProcessor` | Done (sophia PR #125) |
| 03a | Hermes `llm_generate` refactor to proposal-first | Done (context injection) |
| 03b | Hermes feedback Redis persistence | Not done — `/feedback` is a stub |

## Test Coverage

### What's tested

| Component | Test Type | Notes |
|-----------|-----------|-------|
| `ProposalProcessor.process()` | Unit (4 tests) | Mock HCG client + mock Milvus. Tests: node ingestion, context retrieval, dedup skip, empty name skip |
| `/ingest/hermes_proposal` endpoint | Unit (5 tests) | Processor is None (not initialized), so only tests the HTTP layer, not actual processing |
| HCG Client `add_node`, `add_edge` | Unit | Mock Neo4j driver |

> **Stale-claim correction (2026-05-30).** Earlier revisions of this section said
> `ProposalBuilder` and the Hermes Sophia-integration path had *zero tests*. That
> is no longer true: `hermes/tests/unit/test_proposal_builder.py` covers NER
> extraction, entity proposal, pipeline metadata, graceful degradation, and the
> combined-extractor pipeline (6 tests), and `test_llm_endpoint_dual_proposal.py`
> exercises the `/llm` context-injection path. The rows below are updated to match.

### What's NOT tested (or under-tested)

| Component | Impact |
|-----------|--------|
| `ProposalBuilder` (hermes) | **Now unit-tested** (`test_proposal_builder.py`, 6 tests). Real-service embedding/Milvus behavior still only exercised at the seam, not end-to-end |
| `/llm` context injection (hermes) | **Now covered** by `test_llm_endpoint_dual_proposal.py`. The live Sophia round-trip is still mocked, not integration-tested |
| Full pipeline (hermes → sophia → Neo4j → Milvus → context return) | Integration tests exist but are **vacuous** at the seam — `test_hermes_ingestion_integration.py:139` queries `{id: $proposal_id}` while nodes are stored with `uuid`, so it asserts nothing (S1-04 / logos#529). This is *why* the keystone embedding bug went undetected |
| Feedback loop (sophia → Redis → worker → hermes) | Zero tests |
| Milvus dedup with real embeddings | Zero tests. The L2 threshold of 0.5 is untested against real data |
| SHACL validation rejecting a malformed node | Zero tests |

## Architecture Observations

### Good decisions

1. **Hermes knows nothing about the graph.** All graph semantics live in Sophia. Hermes only understands language (NER, embeddings). This is clean separation of concerns.

2. **Graceful degradation everywhere.** If Sophia is down, Milvus is down, spaCy model isn't loaded, or any network error occurs — the `/llm` endpoint still works. The cognitive loop is entirely optional from the caller's perspective.

3. **Reified edges.** Storing relationships as nodes allows rich provenance tracking (who created this edge, when, confidence, source service). This is unusual but well-suited to a cognitive architecture that needs to reason about its own knowledge.

4. **Idempotent writes.** Both `add_node` and `add_edge` use `MERGE`, so replaying proposals doesn't create duplicates.

5. **Type classification in Sophia, not Hermes.** Type semantics are owned by the cognitive core, not the language service. Sophia classifies entities by embedding distance to type centroids (sophia #129, logos #494). Hermes extracts entities and relations but does not assign types — that's Sophia's job.

### Concerns

1. **Redundant embedding generation.** Every `/llm` call generates O(n_entities + 1) embeddings in memory before the LLM is even called. Hermes now caches context responses in Redis (hermes #85), which reduces redundant Sophia round-trips. Sophia deduplicates on storage. The parallel pipeline also mitigates latency by overlapping embedding generation with other work.

2. **L2 threshold is a magic number.** `ENTITY_MATCH_THRESHOLD = 0.5` is hardcoded, undocumented, and untested. For 384-dim MiniLM embeddings, this is a reasonable guess, but it should be validated empirically. Too low = excessive dedup (legitimate new entities get skipped). Too high = no dedup (every mention creates a new node).

3. **Basic relationship extraction.** Hermes now extracts verb-mediated relations via spaCy dependency parsing (e.g., "Paris is located in France" → LOCATED_IN edge). This covers simple subject-verb-object patterns but misses complex multi-hop relationships, implicit relations, and context-dependent semantics. LLM-based extraction would improve coverage.

4. **Feedback goes nowhere.** The infrastructure is built (Redis queue, worker, retry logic, hermes endpoint) but the terminal destination is a log line. No learning, adaptation, or state update happens based on feedback.

5. **No embedding caching in Hermes.** Hermes generates embeddings in-memory and sends them as payload in the proposal to Sophia. Sophia is the sole persistence layer (Neo4j + Milvus). This is architecturally clean, but means Hermes can't short-circuit proposals for recently-seen entities without asking Sophia first.

## What Expanding the Loop Looks Like

The cognitive loop as shipped is the thinnest possible slice: extract entities → store in graph → retrieve as context → enrich LLM prompt. Expanding it into the loop the vision describes is exactly the arc of the six **MTS sprints** (`docs/plans/2026-05-29-roadmap-to-mts.md`); the axes below map onto those sprints.

0. **Stabilize the spine (Sprint 1, now)** — before adding anything, the keystone embedding-persistence bug (sophia#146 / logos#528) must land so vectors actually persist, the vacuous integration tests must assert real outcomes (logos#529), and #505 emergent-type discovery must merge live-verified. Don't build cognition on a foundation that's secretly a no-op.

1. **Relationship extraction** — ✅ Implemented. Hermes extracts verb-mediated relations via spaCy and proposes edges with embeddings. Sophia resolves names to UUIDs and stores edges in Neo4j + Milvus. Future work: LLM-based extraction for more complex relationships.

2. **Feedback processing (Sprint 2)** — Make the `/feedback` endpoint actually mutate the graph: update confidence scores, dedup/deprecate nodes, adjust the threshold, trigger re-embedding. Today it logs and discards.

3. **Orchestrator + event-driven loop (Sprint 3)** — Sophia runs request/response today. Turn it into a persistent process reacting to bus events, with ≥1 daemon transforming a real signal. (Design: `docs/plans/2026-03-14-event-driven-cognitive-loop-design.md`.)

4. **K-lines / curiosity / CWM-E gain (Sprint 4)** — stimulus activates a constellation; non-activation emits curiosity; CWM-E state measurably modulates processing. The CWM-E state models exist; the mechanisms don't.

5. **Memory tiers + planning integration (Sprint 5)** — knowledge promotes ephemeral→STM→LTM by criteria, and new knowledge enabling a goal triggers the `HCGPlanner` (which exists in the foundry but is **not invoked anywhere in the loop today**, and still co-exists with a planner stub, logos #403).

6. **Context quality / multi-turn memory** — richer context (subgraph snippets, confidence-weighted, summarized) and linking `/llm` turns via `correlation_id` into conversation-level context. Folds into Sprints 5–6.

7. **Talos integration (Horizon 2)** — Proposals from sensor data (camera frames, IMU) rather than just text, connecting the loop to the physical world. Deliberately out of MTS scope.
