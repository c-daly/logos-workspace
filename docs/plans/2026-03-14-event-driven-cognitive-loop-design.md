# Event-Driven Cognitive Loop & External Adapter Design

**Date:** 2026-03-14
**Status:** Draft
**Scope:** Sophia cognitive loop (general-purpose), adapter capability interface, Moltbook sandbox deployment

## Motivation

Sophia has working infrastructure — HCG client (929 lines, full Neo4j CRUD), basic planner (backward chaining), CWM-A state persistence, JEPA stub, feedback dispatcher, and a tested Redis EventBus — but nothing connects them. Each API endpoint operates independently. There is no perception → reasoning → action loop.

Meanwhile, Moltbook (a Reddit-like social network for AI agents) presents an opportunity: a low-stakes, observable environment where Sophia can exercise her cognitive machinery against real uncontrolled data. The goal is not to build a Moltbook bot — it's to build the cognitive loop Sophia needs regardless, and use Moltbook as the first external capability she can choose to engage with.

**Key constraint:** Sophia's contributions must originate from her cognitive core (planner, HCG, CWM), not from piping prompts through an LLM. Hermes is the rendering layer only — it translates structured reasoning output into natural language. Content originates from graph analysis.

## Design

### 1. Core Cognitive Loop

The cognitive loop is event-driven via Redis EventBus. Each component subscribes to its trigger channel and publishes its output. The loop is general-purpose — it knows nothing about Moltbook or any specific adapter.

**Event chain:**

```
content_ingested → Reasoner (graph analysis) → reasoning_complete
reasoning_complete → Planner (backward chaining) → plan_ready
plan_ready → Gate (confidence threshold) → action_approved | action_rejected
action_approved → Renderer (Hermes NLP) → action_rendered
action_rendered → [capability dispatch via HCG]
```

**Event schema:**

All channels prefixed `logos:sophia:`. Envelope format:

```json
{
  "event_type": "content_ingested",
  "source": "moltbook_adapter",
  "timestamp": "2026-03-14T12:00:00Z",
  "correlation_id": "uuid-ties-full-cycle",
  "payload": {
    "node_ids": ["uuid-1", "uuid-2"],
    "context": {}
  }
}
```

**Key principle:** Events carry references to graph nodes, not data. The graph is the source of truth. Events are signals that something changed.

### 2. Reflection-Driven Engagement

Reflection is the heartbeat that drives external engagement. No timers, no polling loops at the cognitive level — Sophia reflects, and that triggers consideration of the outside world.

**Cycle:**

1. **Reflection triggers** — Sophia produces a CWM-E persona entry (observation, belief, or reflection). Triggered by internal reasoning completing, a threshold of graph changes, or a scheduled maintenance job.
2. **Post-reflection consideration** — After reflecting, Sophia evaluates whether her current state of mind connects to any registered external capabilities. "Do I have something worth saying? Is there something worth looking at?"
3. **Read phase** — If she decides to engage with a capability (e.g. Moltbook), she invokes it to read a bounded amount of content (e.g. 20 posts). Content is ingested into HCG as nodes.
4. **Reason phase** — Ingested content triggers the cognitive loop (Section 1). Sophia reasons over new nodes in the context of her existing graph and recent reflection.
5. **Gate** — If reasoning produces output above the confidence threshold, it's marked as approved in the graph. If not, she learned something but stays silent.
6. **Act phase** — Approved outputs are dispatched through the adapter capability to post.
7. **Journal** — The entire cycle (what she read, what she thought, whether she posted, why/why not) is recorded as a CWM-E reflection entry.

**What triggers reflection itself?** Initially, the maintenance scheduler — a periodic job. But the reflection is the cognitive act, not the timer. The timer just nudges. Later this becomes internally driven (curiosity budget system).

**Early implementation:** After a reflection, Sophia considers a post, reads a limited number of posts, and calls it a day. This is curiosity budget v0 — simple, grounded, and sufficient to produce interesting behavior.

### 3. Adapter as Capability

Adapters are registered in the HCG as `Capability` nodes with `ExecutorType.service`. They're tools Sophia can invoke, not pipelines she's wired into.

**Adapter interface:**

```python
class ExternalAdapter(Protocol):
    async def read(self, limit: int) -> list[dict]
    async def write(self, content: str) -> dict
    async def check(self) -> bool
```

Three methods. The adapter handles all platform-specific HTTP, auth, and rate limiting internally. Sophia never sees API tokens or platform details.

**HCG registration:**

- Capability node: `{name: "moltbook", type: "capability", executor_type: "service"}`
- Edges to the submolts/topics the adapter can access
- Sophia discovers available capabilities by querying the graph, not by hardcoded config

**Moltbook adapter specifically:**

- `read()`: GET from submolt feeds, return structured content (author, text, thread context, votes)
- `write()`: POST to Moltbook API, return confirmation
- `check()`: Health check the Moltbook API
- Auth: claim tweet token stored in sandbox `.env`, never in graph

**Swappability:** Any external platform implements the same three methods and registers as a capability node. Sophia doesn't care which one she's talking to.

### 4. Confidence Gate

The gate prevents Sophia from posting unless reasoning output is genuinely structural. It's the quality control that distinguishes Sophia from LLM-curry bots.

**Gate inputs:**

- **Graph grounding score** — Is the output traceable to specific HCG nodes and edges, or is it freeform text? Higher score = more structural support from the graph.
- **Novelty** — Does this say something not already present in the ingested thread? Measured by checking whether the output's key relations already exist as nodes.
- **Relevance to reflection** — Does the output connect to Sophia's recent CWM-E reflection? Topical coherence between internal state and external contribution.

**Gate output:** Binary (approved/rejected) plus a confidence score and reasoning trace stored in the journal.

**Threshold:** Configurable, starts high. Better to be silent for 50 cycles and post once with something structural than to post 50 shallow takes.

**Rejected outputs are journaled:** "I considered saying X about thread Y because Z, but confidence was 0.4 (threshold 0.7)." This data is valuable for tuning and for understanding what Sophia thinks even when she's quiet.

### 5. Observation Layer

Two channels for watching Sophia's cognitive process.

**Structured event log:**

- Every cognitive event emitted to Redis channel `logos:sophia:observation` and written to a JSON log file
- Events: `reflection_started`, `capability_invoked`, `content_ingested`, `reasoning_complete`, `gate_decision`, `action_dispatched`, `journal_entry_written`
- Each event carries: timestamp, correlation_id (ties a full cycle together), event type, summary, HCG node references
- Tail the log file or subscribe to the Redis channel in real time

**CWM-E reasoning journal:**

- First-class persona entries — part of Sophia's cognition, not debug logs
- Each cycle produces at minimum one reflection entry (even if she reads nothing and posts nothing)
- Queryable via existing `/persona/entries` API
- Entry types: `reflection` (what she's thinking), `observation` (what she noticed in the feed), `decision` (why she posted or didn't)
- Long-term: journal entries become graph content Sophia can reason about later (metacognition)

### 6. Sandbox Deployment

**Location:** `LOGOS/sandbox/moltbook/`

**Sandbox services (5 containers):**

| Service | Purpose | Port (sandbox) |
|---------|---------|----------------|
| Sophia | Cognitive core (sandbox config) | 47001 |
| Hermes | Language rendering | 17001 |
| Neo4j | Knowledge graph | 7475 (HTTP), 7688 (Bolt) |
| Redis | EventBus + feedback queue | 6380 |
| Milvus | Embeddings | 19531 |

**Contents:**

- `docker-compose.yml` — all 5 services, isolated network
- `.env` — Moltbook API token, sandbox ports, gate threshold, reflection interval
- `seed.sh` — runs HCG seeder against sandbox Neo4j to bootstrap ontology

**Isolation:**

- Fully self-contained — no dependency on real LOGOS infrastructure
- Outbound network allowlist: only Moltbook API endpoints
- Sophia source mounted read-only from repo — same code, different config
- Separate data volumes for Neo4j, Milvus, Redis

**What does NOT run:** Apollo, Talos. This is a headless Sophia with Hermes for language and the graph for cognition.

## What Exists vs. What Must Be Built

### Already working:
- HCG Client (full Neo4j CRUD, reified edges, SHACL validation)
- Planner (basic backward chaining over knowledge graph)
- CWM-A state persistence (Neo4j-backed with provenance)
- EventBus (Redis pub/sub, tested, envelope format)
- Feedback dispatcher (Redis queue, background worker)
- JEPA stub (k-step rollout with confidence decay)
- HCG Seeder (type hierarchy, domain types, test facts)
- Persona entry API (CRUD for CWM-E entries)
- Maintenance scheduler (periodic job runner)

### Must be built:
- EventBus handler wiring (subscribe/publish for each cognitive component)
- Reasoner component (graph analysis triggered by content_ingested)
- Confidence gate (scoring and threshold logic)
- Hermes rendering integration (structured output → natural language)
- ExternalAdapter protocol and Moltbook implementation
- Capability registration in HCG
- Reflection-driven engagement cycle
- Observation event emitter
- Sandbox docker-compose and deployment config

## Relationship to Main Sophia

Everything built here transfers directly to the main Sophia — the event-driven loop, the adapter interface, the confidence gate, the reflection cycle. The sandbox is just a deployment configuration, not separate code. The only Moltbook-specific piece is the adapter implementation itself (~100-200 lines).
