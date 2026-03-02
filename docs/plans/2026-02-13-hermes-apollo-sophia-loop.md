# Hermes/Apollo/Sophia Cognitive Loop Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Close the cognitive loop so Apollo chat → Hermes (builds graph-ready proposals, queries Sophia for context, enriches LLM response) → Sophia (decides what enters the HCG, returns relevant context) works end-to-end.

**Architecture:** Hermes is the language utility — the only component that understands text. It translates conversation turns into structured graph-ready proposals (JSON/Cypher annotated with embeddings) because Sophia is non-linguistic. Sophia works with structures, embeddings, and graph relationships — never raw text. Each turn, Hermes sends Sophia a proposal ("here's how you'd add this, you decide if it's relevant") and Sophia responds with relevant existing context (nodes, relationships, scores). Hermes translates that context back into language for the LLM prompt. Apollo drives the UI.

**Tech Stack:** Python/FastAPI (Hermes, Sophia), spaCy 3.8 (NER), sentence-transformers (embeddings), Neo4j (graph), Milvus/pymilvus (vectors), Redis (feedback queue), React/TypeScript (Apollo webapp)

---

## Context: Ticket Audit Findings

This plan was informed by a full audit of existing tickets against the codebase. Key findings:

| Ticket | Issue Found |
|--------|-------------|
| sophia#13 (CLOSED) | `/ingest/hermes_proposal` only logs proposals, returns `stored_node_ids=[]`. Has TODO: "Pass to cognitive processing." Contract exists but implementation is a stub. |
| hermes#17 (CLOSED) | Title says "persist" but `/feedback` endpoint only logs. No storage to any database. |
| hermes#38 (OPEN) | Appears fixed — scripts work, service names consistent, 207 tests collect. Should be closed. |
| logos#246 (OPEN) | Persona diary IS implemented (`logos_persona/diary.py` + API). May be closeable. |
| logos#265 (OPEN/BLOCKED) | Blocked by #246, but #246 appears done. Blocker may be stale. |
| logos#403 (OPEN) | Both `planner_stub` and `HCGPlanner` coexist (23 refs to stub). Deprecation not started. |
| logos#469 (OPEN) | Zero Redis anywhere — no imports, no config, no docker-compose entries. Feedback system designed for Redis has nothing to talk to. |
| sophia#20 (OPEN) | Executor is a 52-line stub — queue/dequeue only, no execution logic. |
| sophia#101 (OPEN) | No session management code exists at all. |

Some of these older implementations predate the current infrastructure and will be superseded by this plan.

---

## Prerequisite: Ticket Housekeeping

### Step 1: Close hermes#38

hermes#38 ("Fix integration test scripts") is OPEN but resolved — scripts functional, service names consistent, 207 tests collect.

Run: `cd /home/fearsidhe/projects/LOGOS/hermes && gh issue close 38 -c "Verified fixed: scripts functional, service names consistent, 207 tests collect."`

### Step 2: Create new ticket for proposal processing pipeline

sophia#13 ("Add Hermes→Sophia ingestion contract") is CLOSED and the contract IS done. Create a new ticket for making the endpoint actually process proposals (Task 3 below).

### Step 3: Evaluate logos#246 for closure

logos#246 ("CWM-E persona diary storage & API exposure") — `logos_persona/diary.py` and `logos_persona/api.py` are implemented with full CRUD + sentiment. Verify scope is met; if closeable, unblock logos#265.

---

## Per-Turn Flow

This is the interaction that happens on every conversational turn:

```
User → Apollo → Hermes                          Sophia
                  │                                │
                  ├─ 1. Process turn               │
                  ├─ 2. Extract entities (NER)     │
                  ├─ 3. Generate embeddings        │
                  ├─ 4. Build proposal             │
                  │     (JSON/Cypher + embeddings)  │
                  │                                │
                  ├────── proposal ────────────────→│
                  │                                ├─ 5. Search HCG (embeddings)
                  │                                ├─ 6. Decide what to ingest
                  │                                ├─ 7. Create/update nodes
                  │                                ├─ 8. Store embeddings in Milvus
                  │                                │
                  │←───── relevant context ────────┤
                  │       (nodes + scores)          │
                  │                                │
                  ├─ 9. Translate context to text  │
                  ├─ 10. Inject into LLM prompt    │
                  ├─ 11. Generate LLM response     │
                  │                                │
User ← Apollo ← Hermes                          Sophia
```

**Key principle:** Hermes handles all language because Sophia is non-linguistic. Hermes translates conversation → structured proposals (going to Sophia) and graph context → natural language (coming back from Sophia). Sophia works only with structures, embeddings, and graph relationships.

**Key principle:** The proposal is a proposal, not a command. Hermes says "here's how to add this stuff (because I know text and you don't), but you decide if it's relevant." Sophia has cognitive authority over the HCG.

**Key principle:** A lot of the proposal content comes from ephemeral memory — Hermes's conversation record. As the memory infrastructure matures (logos#411), proposals will get richer, but the pipe stays the same.

---

## Task 1: Deploy Redis Infrastructure

**Why:** Sophia's feedback system (sophia#16) was designed around Redis but Redis was never deployed. Also needed for ephemeral memory (Phase 3). Without Redis, Sophia→Hermes feedback is dead.

**Repo:** logos (shared infra), sophia, hermes

**Files:**
- Modify: shared docker-compose (add Redis service with health check)
- Modify: `sophia/containers/docker-compose.test.yml`
- Modify: `hermes/tests/e2e/stack/hermes/docker-compose.test.yml`
- Modify: `logos_config/` — add Redis host/port configuration (following `Neo4jConfig`/`MilvusConfig` pattern)
- Verify: `sophia/src/sophia/feedback/queue.py` connects successfully
- Test: `sophia/tests/integration/test_feedback_redis.py` (new)

**Steps:**

1. Add `RedisConfig` to `logos_config` (host, port, db)
2. Add Redis service to shared docker-compose with health check
3. Write integration test: Sophia feedback queue can enqueue/dequeue via Redis
4. Run test — verify it fails (no Redis), start Redis, verify it passes
5. Add Redis to Sophia's test compose stack
6. Add Redis to Hermes's test compose stack
7. Verify `FeedbackWorker` successfully sends to Hermes `/feedback` endpoint end-to-end
8. Commit

**Port allocation** (following LOGOS convention):

| Repo | Redis Port |
|------|-----------|
| Shared/dev | 6379 |
| Hermes (+10000) | 16379 |
| Sophia (+40000) | 46379 |

---

## Task 2: Hermes — Build Graph-Ready Proposals

**Why:** `_forward_llm_to_sophia` currently sends raw LLM text as a flat payload. Hermes should translate the conversational turn into structured, graph-ready data (JSON or Cypher) annotated with embeddings. This is Hermes's core contribution to the loop — it's the only component that understands text, so it must do the linguistic heavy lifting.

**Repo:** hermes

**Files:**
- Modify: `hermes/src/hermes/main.py` — `_forward_llm_to_sophia()` (line 471)
- Create: `hermes/src/hermes/proposal_builder.py` — builds structured proposals from conversation turns
- Modify: `hermes/src/hermes/services.py` — may need helpers combining NER + embedding
- Test: `hermes/tests/unit/test_proposal_builder.py` (new)
- Test: `hermes/tests/unit/test_api.py` — update existing forward-to-sophia tests

**Existing infrastructure (already in Hermes):**
- `get_spacy_model()` at `services.py:93` — loads `en_core_web_sm`
- `process_nlp(text, ["ner"])` at `services.py:213` — returns `{entities: [{text, label, start, end}]}`
- `generate_embedding(text)` at `services.py:258` — returns `{embedding, dimension, model, embedding_id}`
- `Entity` model at `main.py:148` — `{text, label, start, end}` already defined

**Steps:**

1. Write failing test: `ProposalBuilder.build(conversation_turn)` returns a structured proposal with `entities` (each annotated with embeddings), `relationships` (proposed edges), and a `document_embedding`
2. Run test, verify it fails
3. Implement `ProposalBuilder`:
   - `build(raw_text: str, metadata: dict) -> dict` — the main entry point
   - Run NER via `process_nlp(raw_text, ["ner"])` to extract entities
   - For each entity, generate embedding via `generate_embedding(entity.text)`
   - Build proposed nodes: `[{name, type, embedding, embedding_id, dimension, model}]`
   - Build proposed relationships where detectable: `[{from_name, to_name, relationship}]`
   - Generate document-level embedding for the full text
   - Return structured proposal dict ready for Sophia
   - Graceful degradation: if spaCy/sentence-transformers unavailable, return minimal proposal (log warning)
4. Run test, verify it passes
5. Write test: `ProposalBuilder` produces valid output when NER is unavailable
6. Run test, verify it passes
7. Update `_forward_llm_to_sophia` to use `ProposalBuilder` instead of raw text forwarding
8. Commit

**Proposal payload structure:**
```python
{
    "proposal_id": "uuid",
    "correlation_id": "request-uuid",
    "source_service": "hermes",
    "generated_at": "iso-timestamp",
    "confidence": 0.7,

    # Graph-ready structured data
    "proposed_nodes": [
        {
            "name": "entity text",
            "type": "PERSON|ORG|CONCEPT|...",
            "embedding": [0.1, 0.2, ...],
            "embedding_id": "uuid",
            "dimension": 384,
            "model": "all-MiniLM-L6-v2",
            "properties": {}
        }
    ],
    "proposed_relationships": [
        {
            "from_name": "entity_a",
            "to_name": "entity_b",
            "relationship": "RELATES_TO",
            "properties": {}
        }
    ],
    "document_embedding": {
        "embedding": [0.1, 0.2, ...],
        "embedding_id": "uuid",
        "dimension": 384,
        "model": "all-MiniLM-L6-v2"
    },

    # Provenance
    "llm_provider": "openai",
    "model": "gpt-4o-mini",
    "metadata": {"source": "hermes_llm", "derivation": "observed"}
}
```

---

## Task 3: Sophia — Proposal Processing + HCG Update + Context Retrieval

**Why:** `/ingest/hermes_proposal` logs proposals and returns `stored_node_ids=[]`. This task makes Sophia actually process proposals — deciding what to ingest, creating nodes, storing embeddings — and return relevant existing context. This is the cognitive core of the loop.

**Design note:** Sophia is the cognitive center. The endpoint handler should delegate to a `ProposalProcessor` class that owns the cognitive intake logic — evaluating proposals, creating nodes, storing embeddings, and linking to existing graph context. This keeps the HTTP layer thin and gives a clean seam for future cognitive behavior (persona entries, relationship inference, memory tier decisions) without conflating intake with reflection. How and when Sophia reflects on ingested knowledge is a separate concern (logos#265, Phase 3 memory) that reads from the graph after intake.

**Repo:** sophia

**Files:**
- Modify: `sophia/src/sophia/api/models.py` — update `HermesProposalRequest` to accept structured proposals; add `ProposalResponse` with context
- Modify: `sophia/src/sophia/api/app.py` — `ingest_hermes_proposal()` (line 1233) — delegate to `ProposalProcessor`
- Modify: `sophia/src/sophia/hcg_client/client.py` — currently Milvus is health-check only
- Create: `sophia/src/sophia/ingestion/proposal_processor.py` — cognitive intake logic
- Create: `sophia/src/sophia/hcg_client/embedding_store.py` — Milvus collection management + search
- Test: `sophia/tests/unit/test_proposal_processor.py` (new)
- Test: `sophia/tests/unit/test_embedding_store.py` (new)
- Test: `sophia/tests/integration/test_proposal_to_graph.py` (new)

**Existing infrastructure:**
- `HCGClient.add_node(name, node_type, uuid, properties, source, derivation, confidence, tags, links)` — creates Neo4j nodes, returns UUID
- `HCGClient.add_edge(from_uuid, to_uuid, relationship, properties)` — creates relationships
- Milvus connection params accepted by HCGClient but unused (`client.py:42-59`, comment: "currently unused")
- Hermes Milvus schema at `hermes/src/hermes/milvus_client.py`: fields are `embedding_id` (VARCHAR PK), `embedding` (FLOAT_VECTOR), `model` (VARCHAR), `text` (VARCHAR), `timestamp` (INT64)

**Steps:**

1. Write failing test: updated `HermesProposalRequest` accepts `proposed_nodes`, `proposed_relationships`, and `document_embedding` fields
2. Run test, verify it fails
3. Update model in `models.py`; also add context fields to response: `relevant_context: List[dict]`
4. Run test, verify it passes
5. Commit

6. Write failing test: `EmbeddingStore` can insert an embedding and search by similarity
7. Run test, verify it fails
8. Implement `EmbeddingStore` in `embedding_store.py`:
   - Collection schema compatible with Hermes (add `node_uuid` field to link back to Neo4j)
   - `store_embedding(node_uuid, embedding, embedding_id, text, model, dimension)` — insert into Milvus
   - `search_similar(query_embedding, top_k=10)` — return matching `{node_uuid, score}` pairs
   - Collection auto-creation on first use (idempotent)
9. Run test, verify it passes
10. Commit

11. Write failing test: `ProposalProcessor.process()` searches for relevant context AND creates nodes from proposed_nodes
12. Run test, verify it fails
13. Implement `ProposalProcessor` in `ingestion/proposal_processor.py`:
    - Constructor takes `hcg_client` and `embedding_store`
    - `process(request: HermesProposalRequest) -> ProposalResult`:
      - **Retrieve**: Use `document_embedding` to search Milvus for relevant existing nodes
      - **Decide**: For each proposed node, determine if it should be ingested (start simple: accept all, iterate later)
      - **Ingest**: For accepted nodes, call `hcg_client.add_node(...)` and `embedding_store.store_embedding(...)`
      - **Link**: For proposed relationships, call `hcg_client.add_edge(...)` where both endpoints exist
      - **Return**: `ProposalResult(stored_node_ids=[...], relevant_context=[{node_uuid, node_name, node_type, score, properties}])`
      - Skip nodes with empty names, log warnings for missing embeddings
14. Update `ingest_hermes_proposal()` to delegate: `result = proposal_processor.process(request)` and return both stored_node_ids and relevant_context
15. Run test, verify it passes
16. Commit

---

## Task 4: Hermes — Per-Turn Proposal + Context Injection

**Why:** The `/llm` endpoint currently generates blind, then forwards raw text. It should build a structured proposal from the conversation, send it to Sophia (which returns relevant context), translate that context back into language for the LLM prompt, then generate.

**Repo:** hermes

**Files:**
- Modify: `hermes/src/hermes/main.py` — `llm_generate()` (line 544) and `_forward_llm_to_sophia()` (line 471)
- Use: `hermes/src/hermes/proposal_builder.py` (from Task 2)
- Test: `hermes/tests/unit/test_context_injection.py` (new)
- Test: `hermes/tests/unit/test_api.py` — update existing `/llm` tests

**Steps:**

1. Write failing test: `/llm` sends proposal to Sophia BEFORE generating, receives context, injects it into messages
2. Run test, verify it fails
3. Refactor the flow in `llm_generate()`:
   - Extract user's latest message text
   - Build proposal via `ProposalBuilder.build(user_text, metadata)`
   - Send proposal to Sophia `/ingest/hermes_proposal`
   - Receive response including `relevant_context`
   - Translate relevant context into a system message (Hermes does the language work: "Relevant knowledge: [node_name] is a [node_type] related to...")
   - Prepend context message to the conversation
   - Call `generate_llm_response()` with enriched messages
   - Return response to Apollo
4. Run test, verify it passes
5. Write test: `/llm` works normally when Sophia is unavailable (graceful degradation — generate without context)
6. Run test, verify it passes
7. Commit

**Note:** This collapses the old separate "forward after generation" and "query before generation" into a single per-turn interaction. The proposal goes to Sophia, context comes back, then Hermes generates.

---

## Task 5: Hermes — Fix Feedback Persistence (hermes#17)

**Why:** `/feedback` at `main.py:807` only logs. With Redis available (Task 1), persist for retrieval.

**Repo:** hermes

**Files:**
- Modify: `hermes/src/hermes/main.py` — `receive_feedback()` (line 807)
- Test: `hermes/tests/unit/test_feedback.py` (new)

**Steps:**

1. Write failing test: `/feedback` stores payload (retrievable by correlation_id)
2. Run test, verify it fails
3. Implement: store feedback in Redis hash keyed by `correlation_id` with configurable TTL
4. Run test, verify it passes
5. Commit

---

## Dependency Graph

```
Task 1 (Redis) ─────────────────────────→ Task 5 (Feedback persistence)

Task 2 (Proposal builder) → Task 3 (Sophia processing) → Task 4 (Per-turn flow)
```

- Tasks 1 and 2 can proceed **in parallel**
- Task 3 depends on Task 2 (proposal contract)
- Task 4 depends on Tasks 2 and 3 (needs builder + Sophia endpoint)
- Task 5 depends on Task 1 (needs Redis)

---

## Out of Scope (Future Work)

| Item | Why Deferred |
|------|-------------|
| sophia#101 — Session boundaries | No code exists; separate concern |
| sophia#20 — Executor implementation | 52-line stub; separate from chat loop |
| logos#403 — planner_stub deprecation | 23 references; mechanical but separate |
| logos#411-414 — Phase 3 memory | Depends on Redis + sessions; will enrich proposals over time |
| logos#416 — Testing sanity | Infrastructure exists, needs live services |
| Sophia→Hermes embedding requests | Not part of this interaction flow |
| JEPA embeddings | Separate media pipeline (sophia#76); nodes get JEPA embeddings when media present |
| Proposal validation/filtering | Start simple (accept all), iterate on acceptance logic |
| Reflection mechanism | Separate from ingestion; starts with ephemeral memory, reads from graph after intake |

---

## Acceptance Criteria

The loop is closed when:

1. User types a message in Apollo chat
2. Apollo sends to Hermes `/llm`
3. Hermes builds a structured proposal (entities + embeddings + proposed relationships) from the conversation turn
4. Hermes sends proposal to Sophia `/ingest/hermes_proposal`
5. Sophia searches HCG using the proposal's embeddings for relevant existing context
6. Sophia decides what from the proposal to ingest, creates nodes in Neo4j, stores embeddings in Milvus
7. Sophia returns relevant context (nodes + scores) alongside stored_node_ids
8. Hermes translates the returned context into natural language (because Sophia is non-linguistic)
9. Hermes injects context into LLM prompt, generates response
10. Hermes returns response to Apollo
11. **On the NEXT message**, step 5 finds context from the nodes created in step 6

Step 11 is the critical test — the loop is only closed when information flows back.
