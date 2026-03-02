# Session Handoff - 2025-12-29

## Current State
- **Phase:** Complete
- **Graph size:** ~1430 nodes, ~1100 edges
- **Blockers:** None

## This Session's Work

### 1. Performance Optimizations
Addressed slowness as graph grew to 1400+ nodes.

| Optimization | Impact |
|--------------|--------|
| Trigram index for fuzzy lookup | O(n) → O(k), 0.001s lookups |
| Prefix blocking for deduplication | O(n²) → O(n×k), 13K vs 1M pairs |
| Persistent embedding cache | No API calls on repeat `/revise` |
| Batch embedding API calls | 100 embeddings per API call |

**Files modified:**
- `substrate/graph.py` - Added `_nodes_by_prefix`, `_nodes_by_trigram`, `find_similar_nodes()`
- `extraction/extractor.py` - `_find_similar_node()` uses trigram index
- `revision/reviser.py` - `embedding_cache.json`, `_get_embeddings_batch()`, `_precompute_embeddings()`, blocking in `_deduplicate()`

### 2. Visualization Tooltips Fix
Raw HTML wasn't rendering in pyvis tooltips. Rewrote to plain text format.

**Before:** `<div class="tt"><span class="tt-title">...` (raw HTML shown)
**After:**
```
━━━ Node Name ━━━
Confidence: 80%  |  Connections: 5  |  Cluster: 3
Definition:
  Word-wrapped text...
```

**File:** `conversation/mind.py` - `build_tooltip()` rewritten

### 3. Bug Fixes

| Bug | Location | Fix |
|-----|----------|-----|
| "unhashable type: 'list'" on `/wonder` | `curiosity/drive.py:160` | Handle `is_a` as list |
| "unhashable type: 'list'" on `/wonder` | `curiosity/structural.py:138,148,180` | Handle `is_a` as list |
| Repeated merge/undo cycle | `revision/reviser.py:_undo_bad_merges()` | Track processed undos |

### 4. Previous Session (for context)
- Topic clustering via Louvain community detection
- Duplicate prevention in extraction/revision
- Research command and HTML visualization

---

## Architecture

```
tiny_mind/
├── substrate/           # Knowledge representation
│   ├── graph.py        # KnowledgeGraph + trigram indexes
│   ├── node.py         # Node class
│   └── edge.py         # Edge class
├── extraction/          # LLM-based learning  
│   └── extractor.py    # Extract + fuzzy matching
├── revision/            # Knowledge maintenance
│   └── reviser.py      # Dedupe + embedding cache
├── curiosity/           # Active learning
│   ├── drive.py        # Goal generation
│   └── structural.py   # Graph analysis
├── conversation/        # Interface
│   ├── mind.py         # TinyMind main class
│   └── clustering.py   # Louvain clustering
└── __main__.py          # CLI
```

## Key Files for Performance

| File | What it does |
|------|--------------|
| `substrate/graph.py` | `find_similar_nodes()` - trigram-based fuzzy search |
| `revision/reviser.py` | `_deduplicate()` - blocking, batch embeddings |
| `embedding_cache.json` | Persistent embedding storage |

## CLI Commands

```bash
python -m tiny_mind chat          # Interactive
python -m tiny_mind know          # Query knowledge
python -m tiny_mind revise        # Maintenance
python -m tiny_mind wonder        # Curiosity goals
python -m tiny_mind viz --html    # HTML visualization
python -m tiny_mind research X    # Web research on topic
```

## Next Steps (potential)

1. Add search/filter to visualization (find node by name)
2. Export to GraphML/JSON-LD
3. Neo4j persistence
4. Add legend to HTML visualization