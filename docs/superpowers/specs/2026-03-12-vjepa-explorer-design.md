# V-JEPA Explorer Design

**Date**: 2026-03-12
**Status**: Draft
**Related**: `2026-03-11-vjepa-grounding-integration-design.md`

---

## Overview

A developer tool in Apollo for interactive exploration of the V-JEPA→CLIP embedding space. The goal is to build intuition for how the translator model behaves — what it retrieves, where things land in the shared space, where it breaks — to inform model evaluation and build confidence before production integration.

The explorer routes through the real Hermes endpoints from day one. Hermes stubs return plausible mock data initially; the real projector and Milvus search drop in without frontend changes when ready.

---

## Goals

1. Query grounded embeddings by text and see what comes back (text→visual direction)
2. Upload a video/image, translate it, and see what text descriptions match (visual→text direction)
3. Visualise the embedding space as a 2D scatter: see whether text and visual embeddings cluster together (alignment) or separate (failure)
4. Project a live query point into the scatter in real time
5. Establish the real Apollo→Hermes integration path before the model is trusted

## Non-Goals

- Running the V-JEPA encoder locally (initial scope: pre-computed MSR-VTT embeddings only)
- Production-quality search (this is exploratory tooling, not a retrieval service)
- User-facing UI (dev tool only)

---

## Page Structure

New dedicated page at `/vjepa-explorer`, added to Apollo's nav alongside the HCG Explorer. Three panels:

```
┌─────────────────┬─────────────────────┬──────────────────────────┐
│   Query Panel   │   Results Panel     │   Embedding Scatter      │
│                 │                     │                          │
│  Text input     │  Ranked results:    │  UMAP projection of      │
│  ─────────────  │  - similarity score │  MSR-VTT embeddings      │
│  File upload    │  - caption / desc   │  · text (blue)           │
│                 │  - video ID         │  · visual (orange)       │
│                 │                     │  ★ query point (live)    │
│                 │                     │  ● top-k results         │
└─────────────────┴─────────────────────┴──────────────────────────┘
```

**Query panel** handles both directions:
- Text input → `search/text` → nearest grounded nodes
- File upload → `translate` → `search/visual` → nearest text descriptions

**Results panel** renders the ranked list regardless of direction. Shape differs:
- Text query results: video ID, top caption, cosine similarity
- Visual query results: text description, cosine similarity

**Embedding scatter** is always visible. Pre-computed 2D UMAP coordinates for a representative subset (~2000 points) of MSR-VTT embeddings, coloured by modality. When a query runs, the translated embedding is projected live via UMAP `.transform()` and appears as a marked point. Top-k results are highlighted. The alignment of text and visual points is the primary signal — interleaved clusters = working, separated clusters = not.

---

## API Contracts

### Apollo frontend → Apollo Python API

| Route | Method | Input | Output |
|-------|--------|-------|--------|
| `/api/vjepa/search/text` | POST | `{ query: string }` | `{ results: Result[], embedding: float[] }` |
| `/api/vjepa/search/visual` | POST | `{ embedding: float[] }` | `{ results: Result[], embedding: float[] }` |
| `/api/vjepa/translate` | POST | `{ tokens: float[][] }` | `{ embedding: float[] }` |
| `/api/vjepa/umap/project` | POST | `{ embedding: float[] }` | `{ x: number, y: number }` |

`Result` shape:
```typescript
{
  id: string           // video ID (text query) or caption ID (visual query)
  text: string         // video title (text query) or caption text (visual query)
  score: number        // cosine similarity
  modality: "text" | "visual"
  // modality="visual": result is a visually-grounded node (returned by text queries)
  // modality="text": result is a text description (returned by visual queries)
}
```

The `embedding` returned in search responses is the CLIP-space query embedding (CLIP text encoding for text queries; translated CLIP embedding for visual queries). The frontend passes this to `/api/vjepa/umap/project` to obtain the live scatter point.

### Apollo Python API → Hermes (via HermesClient)

Four new methods on `HermesClient`:

```python
translate_jepa_to_clip(tokens: list[list[float]]) -> list[float]
    # POST /translate/jepa
    # Input: V-JEPA tokens, shape (T, 1024) or (1, 1024) mean-pooled
    # Output: 768-dim L2-normalized CLIP-space embedding

search_text_to_visual(text: str) -> SearchResponse
    # POST /search/text-to-visual
    # Input: raw text query (Hermes encodes internally via CLIP text encoder)
    # Output: ranked results from grounded_embeddings

search_visual_to_text(embedding: list[float]) -> SearchResponse
    # POST /search/visual-to-text
    # Input: 768-dim CLIP-space embedding
    # Output: ranked text descriptions

umap_project(embedding: list[float]) -> tuple[float, float]
    # POST /umap/project
    # Input: 768-dim CLIP-space embedding
    # Output: (x, y) coordinates in the pre-computed UMAP layout space
```

### Hermes endpoints (new)

| Endpoint | Stub behaviour | Real behaviour (later) |
|----------|---------------|------------------------|
| `POST /translate/jepa` | Return random L2-normalized 768-dim vector | Run projector_v7 forward pass |
| `POST /search/text-to-visual` | Return 5 random MSR-VTT entries | CLIP text encode internally → ANN search `grounded_embeddings` |
| `POST /search/visual-to-text` | Return 5 random MSR-VTT captions | ANN search CLIP text side of `grounded_embeddings` |
| `POST /umap/project` | Return random `(x, y)` within pre-computed layout bounds | `umap_model.transform([embedding])` → `[x, y]` |

---

## Embedding Scatter

**Pre-computed layout**: A representative subset (~2000 points) of MSR-VTT embeddings — both CLIP text and V-JEPA-translated — projected to 2D via UMAP `fit_transform()`. Stored as a JSON file served by Apollo's static assets or the Apollo API.

**Live projection**: When a query runs, the returned 768-dim embedding is sent to Hermes `POST /umap/project` which calls `umap_model.transform([embedding])` and returns `[x, y]`. The UMAP model is loaded at Hermes startup alongside the projector. For a single point against a ~2000-point reference set, `.transform()` is fast enough for real-time use (sub-100ms on CPU). This assumption holds at the stated ~2000-point scale; reconsider if the layout is expanded significantly.

**Fourth new Hermes endpoint**: `POST /umap/project`
- Input: `{ embedding: float[] }` (768-dim)
- Output: `{ x: float, y: float }`
- Stub: return random `(x, y)` within the pre-computed layout bounds

**Scatter rendering**: D3 (already available in Apollo from HCG Explorer). Points sized by modality, query point marked distinctly, top-k results highlighted with connecting lines to the result list.

---

## Implementation Phases

### Phase 1 — Hermes stubs
- `POST /translate/jepa` → random 768-dim L2-normalized vector
- `POST /search/text-to-visual` → mock MSR-VTT results
- `POST /search/visual-to-text` → mock MSR-VTT captions
- `POST /umap/project` → random `(x, y)` in bounds
- Pre-compute UMAP layout from MSR-VTT embeddings, store as JSON

### Phase 2 — Apollo API + HermesClient
- Four new `HermesClient` methods (including `umap_project`)
- Four new Apollo API routes proxying to Hermes (including `/api/vjepa/umap/project`)
- Unit tests: routes return expected shapes
- Note: if Open Question 1 resolves to the CLIP-image-proxy path, the `/api/vjepa/translate` contract may need a new route or overloaded input — revisit before Phase 3

### Phase 3 — Apollo frontend
- `VJepaExplorer.tsx` page with three panels
- Text query flow end-to-end
- File upload → translate → visual search flow
- Scatter with pre-computed layout + live projection point
- Acceptance criteria: text query returns ranked results and places a point on the scatter; file upload flow returns text descriptions; scatter live point appears within the pre-computed layout bounds; all three panels are visible simultaneously without layout overflow

### Phase 4 — Wire real Hermes (when model trusted)
- Load `projector_v7.py` + checkpoint in Hermes
- Load fitted UMAP model (pre-computed on MSR-VTT) — requires Open Question 3 resolved (joblib serialisation path in config)
- Replace stubs with real implementations
- No frontend changes required
- Acceptance criteria: text query returns MSR-VTT results ranked by actual cosine similarity; live scatter point lands near the top-k result cluster

---

## Open Questions

1. **File upload to tokens**: The visual query path requires V-JEPA tokens as input to `/api/vjepa/translate`. For the initial scope (no V-JEPA encoder running locally), file uploads can use CLIP image encoder as a proxy — send frames to `POST /embed/image` in Hermes, get 768-dim embedding, skip `/translate/jepa` entirely. This loses the V-JEPA representation but lets the visual→text direction work before the encoder is available. Decision affects the Phase 2 API contract: the CLIP-proxy path requires either a new Apollo route (`/api/vjepa/embed/image`) or a modified `translate` endpoint that accepts image frames in addition to tokens. Decide before Phase 3 so Phase 2 routes are correct.
2. **Pre-computed layout data**: The UMAP layout JSON needs to be generated from `msrvtt_embeddings.h5` (available locally). This is a one-off script, not part of the service. Should live in `PoCs/jepa_clip_translator/scripts/` and be committed as a static asset.
3. **UMAP model serialisation**: Fitted UMAP model needs to be persisted (joblib) and loaded by Hermes. Same lifecycle as the projector checkpoint — loaded at startup, path from config.
