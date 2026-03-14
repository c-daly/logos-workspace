# V-JEPA Grounding Integration Design

**Date**: 2026-03-11
**Status**: Draft
**Author**: Claude Code
**Branch**: full-jepa-token-grid

---

## Overview

Integrate the trained V-JEPA→CLIP translator (from the `PoCs/jepa_clip_translator` experiment run) into the LOGOS architecture so that HCG nodes can carry grounded embeddings alongside their existing text embeddings. This enables Sophia to query her knowledge graph using visual observations — from ingested video, from planning simulations, or eventually from real sensors — through the same proposal pipeline that text-based learning already uses.

The central thesis this design supports: **cognition in LOGOS is not grounded in language**. Grounded perception is a first-class input pathway, structurally identical to text extraction. A proposal derived from watching a video and one derived from reading text are indistinguishable to the HCG.

---

## Goals

1. Tag HCG nodes with 768-dim CLIP-space embeddings derived from V-JEPA video observations
2. Enable bidirectional cross-modal querying: text queries find visually-grounded nodes; visual states find matching nodes and text descriptions
3. Feed visual observations through the existing `ProposalProcessor` pipeline — same schema, same endpoint, same validation as text proposals
4. Support simulation/planning states as the first source of grounded perception (no embodiment yet)
5. Be forward-compatible with Talos sensor input when embodiment arrives

## Non-Goals

- Real-time video streaming / live sensor ingestion (deferred to embodiment phase)
- Replacing or modifying the existing text embedding pipeline or Sophia's type-named Milvus collections
- Object detection or frame-level entity tracking (initial scope: video-level and clip-level embeddings only)
- Fine-tuning the projector on new data during runtime

---

## Architecture

### Components

| Component | Repo | Responsibility |
|-----------|------|----------------|
| `jepa_translator` | hermes | Loads the trained projector checkpoint, serves `/embed/vjepa` endpoint |
| `grounded_embeddings` | Milvus (shared) | 768-dim CLIP-space embeddings from V-JEPA |
| `grounded_embedding_id` | Neo4j node property | FK into `grounded_embeddings` collection |
| `ingestion` pipeline | sophia | Extracts V-JEPA tokens, calls Hermes, submits proposal |
| `CWM-G` | sophia | One layer of the unified CWM; emits `CWMState(model_type="CWM_G")` via `CWMPersistence`; during planning episodes, queries grounded store without persisting |
| `JEPARunner` | sophia | Produces `SimulationResult` → fed to CWM-G (see Open Question 5) |

### Why Hermes

Hermes already owns all embedding operations (CLIP text, OpenAI text, Milvus client). Adding V-JEPA→CLIP translation is a natural extension: Hermes becomes the single service responsible for mapping any input — text, image, or video — into a queryable embedding space. Sophia remains a cognitive layer that never loads model weights.

---

## Embedding Space Design

The system maintains **two parallel embedding stores**:

| Collection | Dimensions | Model | Purpose |
|------------|-----------|-------|---------|
| `Entity`, `Concept`, `State`, `Process`, `Edge` (Sophia-side) | 1536 | OpenAI text-embedding-3-small | Text-derived node embeddings (existing, routed by node type) |
| `grounded_embeddings` (new) | 768 | CLIP ViT-L/14 + V-JEPA projector | Visual-derived node embeddings |

Note: Sophia's `ProposalProcessor` currently routes embeddings to type-named Milvus collections (`Entity`, `Concept`, etc.) keyed by semantic node type — not to a single `hermes_embeddings` collection. The `hermes_embeddings` collection exists in Hermes's own `milvus_client.py` for Hermes-internal use and is separate. `grounded_embeddings` is a new Sophia-side collection added alongside the existing type-named ones.

These are complementary, not competing. A node may have entries in both a type-named collection and `grounded_embeddings`. Cross-modal search (text ↔ visual) happens within `grounded_embeddings`, where both CLIP text and CLIP-translated V-JEPA embeddings share the same 768-dim space. Text-only search continues to use the existing type-named collections.

**Embedding consistency rule**: All embeddings in `grounded_embeddings` — whether from text (CLIP text encoder) or video (V-JEPA → projector) — must use the same 768-dim CLIP ViT-L/14 space. This is what makes both directions of cross-modal search work.

---

## Data Flows

### Ingestion (permanent node tagging)

```
Video file
  → sophia ingestion pipeline
    → V-JEPA encoder (facebook/vjepa2-vitl-fpc64-256)
    → mean-pool tokens → (B, 1024)
    → POST /embed/vjepa  →  Hermes jepa_translator
        → projector forward pass
        → L2-normalize → (B, 768)
        → store in Milvus grounded_embeddings → embedding_id
    → Assemble proposal (proposed_nodes, proposed_edges, document_embedding)
    → POST /ingest/hermes_proposal  →  Sophia ProposalProcessor
        → context search (document_embedding)
        → node dedup (L2 < 0.5 in grounded_embeddings)
        → create/update Neo4j nodes with grounded_embedding_id
        → create edges
```

### Query — text → visual nodes

```
Text query
  → POST /embed/text  →  Hermes (CLIP text encoder, 768-dim)
  → ANN search: grounded_embeddings (768-dim)
  → top-k Neo4j node IDs → Sophia reasons
```

### Query — visual/planning state → descriptions

```
Sophia planner (planning episode active)
  → JEPARunner → SimulationResult (see Open Question 5 re: token extraction)
  → CWM-G → POST /embed/vjepa [query mode]
    → Hermes projector → 768-dim (not stored)
    → ANN search: grounded_embeddings → nearest known nodes
    → ANN search: grounded_embeddings (CLIP text side) → nearest descriptions
  → results inform planner: "imagined state resembles node X"
  → planning episode ends → embedding discarded
```

No state is held between planning episodes. CWM-G is active only within a planning episode.

---

## Hermes `jepa_translator` Module

### New endpoint

```
POST /embed/vjepa
```

**Request**:
```json
{
  "embeddings": [[...]],        // V-JEPA tokens, shape (B, T, 1024) or (B, 1024) mean-pooled
  "mode": "tag" | "query",      // "tag" stores to Milvus; "query" returns embedding only
  "node_id": "uuid"             // Required for "tag" mode; used as Milvus primary key
}
```

**Response**:
```json
{
  "embeddings": [[...]],        // CLIP-space embeddings, shape (B, 768)
  "embedding_ids": ["uuid"]     // Set if mode == "tag"; null if mode == "query"
}
```

### Model loading

The trained projector checkpoint (see Open Question 1 for naming) is loaded at Hermes startup. The projector is a `ResidualTranslator` as defined in `PoCs/jepa_clip_translator/translator.py`: input projection → N× `ResidualBlock` → output projection Linear(hidden_dim→768) → L2-normalize. Exact architecture hyperparameters (hidden_dim, num_blocks) are determined by the selected checkpoint's saved `config` dict.

The model runs CPU-only for the initial integration. GPU inference can be added when throughput demands it.

### Token handling

The endpoint accepts both:
- Mean-pooled input: `(B, 1024)` — used when pre-pooled outside Hermes
- Token-level input: `(B, T, 1024)` — Hermes mean-pools internally before projector

This matches the `_prepare_batch` logic already in the PoC's `search.py`.

---

## Milvus `grounded_embeddings` Collection

```python
collection_name = "grounded_embeddings"
schema = {
    "embedding_id": VarChar(pk=True),   # UUID
    "node_uuid": VarChar,               # FK to Neo4j node
    "embedding": FloatVector(dim=768),  # CLIP-space
    "modality": VarChar,                # "visual" | "text"
    "model": VarChar,                   # "clip-vit-l14-projector-v7" | "clip-vit-l14-text"
    "source": VarChar,                  # "video_ingestion" | "simulation" | "sensor"
    "created_at": Int64,                # Unix timestamp
}
index = IVF_FLAT, metric=IP  # Inner product on L2-normalized vectors = cosine similarity
```

The existing Sophia-side type-named collections (`Entity`, `Concept`, etc.) are unchanged.

---

## Neo4j Node Schema Addition

New optional property on all node types:

```
grounded_embedding_id: String   # UUID → grounded_embeddings in Milvus
```

Nodes without grounded embeddings remain valid. Grounded embedding is an enrichment, not a requirement.

---

## Proposal Format Compatibility

Proposals from the grounded perception pipeline use the **identical schema** as text-based proposals from Hermes. The only differences are in field values, not structure:

| Field | Text proposal | Grounded proposal |
|-------|--------------|-------------------|
| `source_service` | `"hermes"` | `"hermes"` |
| `model` | `"text-embedding-3-small"` | `"clip-vit-l14-projector-v7"` |
| `dimension` | `1536` | `768` |
| `raw_text` | input text | `""` (empty) |
| `metadata.pipeline.embedding_provider` | `"openai"` | `"clip-jepa-translator"` |

`ProposalProcessor` in Sophia currently routes to Milvus collections by node type (e.g. `"entity"` → `Entity`). To store grounded embeddings in `grounded_embeddings`, a new routing branch is needed: when `dimension=768` and `metadata.pipeline.embedding_provider == "clip-jepa-translator"`, route to `grounded_embeddings` instead. See Open Question 2 for the precise routing strategy to use.

This is the only change required to `ProposalProcessor` for the initial integration.

---

## Reflection and Graph Mutation

When a video observation or planning state produces a grounded embedding and retrieves similar nodes, Sophia has the ingredients for reflection: known nodes, their relationships, and a new observation that relates to them. The reflection step:

1. **Retrieve context**: `document_embedding` in proposal triggers context search — top-k similar nodes returned by `ProposalProcessor`
2. **Reason**: Sophia's orchestrator passes retrieved nodes + visual observation metadata to an LLM (via Hermes LLM gateway) to identify potential new relationships
3. **Propose**: LLM output is assembled into `proposed_edges` (and optionally `proposed_nodes`) in standard proposal format
4. **Ingest**: submitted to `POST /ingest/hermes_proposal` — same path as any other proposal

The reflection step uses language as a tool (LLM for relation labelling) but is triggered by and grounded in a non-linguistic observation. The resulting graph mutations are indistinguishable from text-derived ones — which is intentional.

**Trigger**: reflection is invoked by the Sophia orchestrator after a grounded proposal is processed, when `relevant_context` returned by `ProposalProcessor` contains ≥ 2 nodes with L2 < 0.4 (high-confidence visual neighbours). Below this threshold, the observation is stored but no reflection is triggered.

---

## CWM-G Integration

CWM-G is **not a separate system** — it is one layer of a unified working memory alongside CWM-A (abstract) and CWM-E (emotional). All three layers use the same `CWMState` envelope and write through `CWMPersistence` to the same Neo4j-backed cognitive state log. Sophia's API can retrieve any layer's history with `GET /cwm/states?model_type=cwm_g`.

Grounded observations from video ingestion are persisted:

```python
state = CWMState(
    model_type="CWM_G",
    data={
        "source": "cwm_g_service",
        "derivation": "observed",
        "confidence": 0.85,
        "grounded_embedding_id": embedding_id,
        "node_uuid": node_uuid,
        "modality": "visual",
        "tags": ["cwm", "subsystem:cwm_g"],
    }
)
await cwm_persistence.persist(state)
```

Planning states are **ephemeral** — they are not persisted. During a planning episode CWM-G translates the planning state and queries the grounded store in-memory only:

```python
class ContinuousWorkingMemoryGenerative:
    async def ground_planning_state(
        self,
        vjepa_tokens: Tensor,          # (T, 1024) from JEPARunner
        hermes_client: HermesClient,
    ) -> GroundedPlanningState:
        """Translate a planning state to CLIP space and find similar known nodes.
        Result is ephemeral — not persisted to CWMPersistence.
        """
        embedding = await hermes_client.embed_vjepa(vjepa_tokens, mode="query")
        similar_nodes = await hermes_client.search_grounded(embedding, top_k=5)
        return GroundedPlanningState(embedding=embedding, similar_nodes=similar_nodes)
```

The returned `GroundedPlanningState` lives in the planner's call stack only. When the planning episode ends, it is discarded — no CWMState is emitted.

---

## Forward Compatibility (Embodiment)

When Talos sensors arrive, the only change needed is a new input source: sensor frame → V-JEPA encoder → `POST /embed/vjepa`. The Hermes endpoint, Milvus collection, Neo4j schema, and proposal pipeline are all source-agnostic. Sensor-derived observations will be persisted as `CWMState(model_type="CWM_G")` entries just like video-ingestion observations — they join the same cognitive state log. CWM-G's `ground_planning_state` generalises to `ground_perception_state` with the same interface.

---

## Implementation Phases

### Phase 1 — Hermes translator module
- Add `jepa_translator.py` to Hermes
- Load trained projector checkpoint at startup (resolve naming per Open Question 1)
- Expose `POST /embed/vjepa` (tag + query modes)
- Create `grounded_embeddings` Milvus collection
- Unit tests: projector loads, endpoint returns 768-dim L2-normalized output

### Phase 2 — ProposalProcessor routing
- Add `grounded_embeddings` routing to `ProposalProcessor` (per Open Question 2)
- Integration test: submit grounded proposal, verify node created with `grounded_embedding_id`

### Phase 3 — CWM-G `ground_planning_state`
- Implement `ContinuousWorkingMemoryGenerative.ground_planning_state`
- Wire to JEPARunner output in the planning loop (pending Open Question 5)
- Test: simulation produces tokens → CWM-G returns similar known nodes

### Phase 4 — Ingestion pipeline
- Add V-JEPA extraction to Sophia's ingestion pipeline for video inputs
- Submit grounded proposals on video ingestion
- End-to-end test: ingest MSR-VTT sample → node appears in HCG with grounded embedding

### Phase 5 — Reflection
- Add reflection trigger to orchestrator (post-proposal, ≥ 2 high-confidence neighbours)
- LLM-based relation proposal via Hermes LLM gateway
- Test: ingest two related videos → reflection proposes an edge between their nodes

---

## Open Questions

1. **Checkpoint naming**: The PoC produced checkpoints named `best_translator_exp_NNN_*.pt` (e.g. `best_translator_exp_027_residual_mse_then_tiny_cosine_finish.pt`). The spec references `checkpoints_v7/best_vljepa_v7.pt` and `projector_v7.py` as aspirational names for the production-ready artifact. These need to be reconciled before Phase 1: either rename the best PoC checkpoint into a versioned path, or update the spec to use the actual filename. Checkpoint is currently on RunPod and must be downloaded or stored in a model registry.

2. **ProposalProcessor collection routing**: The current `ProposalProcessor` routes embeddings to type-named Milvus collections (`Entity`, `Concept`, etc.) by node type — it does not use embedding dimension as a routing key. The proposed change (route by `dimension=768` + provider tag to `grounded_embeddings`) needs to be designed carefully to avoid breaking the existing node-type routing path. One clean option: add a `grounded` boolean flag to the proposal node schema and check it before falling through to type-based routing.

3. **Token-level vs mean-pooled**: initial integration uses mean-pooled for simplicity; token-level (better for temporal queries) can be added in a follow-up.

4. **Relation labels for visual edges**: canonical labels in `relation_extractor.py` are text-centric (`WORKS_AT`, `DEVELOPED_BY`, etc.). Visual observations may warrant new canonical labels (`LOCATED_NEAR`, `VISUALLY_SIMILAR_TO`, `CO_OCCURS_WITH`). Confirm with ontology before Phase 4.

5. **`SimulationResult` token extraction**: The current `SimulationResult` model (`sophia/src/sophia/jepa/models.py`) carries structured state data (`imagined_states: List[ImaginedState]`) — it does not expose raw V-JEPA token tensors `(T, 1024)`. For Phase 3 (CWM-G integration), either (a) `SimulationResult` needs a new `vjepa_tokens` field populated by the real V-JEPA backend, or (b) CWM-G must derive a proxy representation from `state_data` dicts. This is gated on the real V-JEPA backend (`JEPA_BACKEND=poc`) being wired through `JEPARunner` — the stub backend produces no tensors.
