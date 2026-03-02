# Session Handoff - 2025-12-29

## Current Task
Improving TinyMind's knowledge revision system to properly clean up and maintain the knowledge graph.

## Status
- Phase: Implement (complete for current scope)
- Progress: Ephemeral node removal integrated, bad merge detection improved
- Blockers: None

## Key Files Modified

- **tiny_mind/revision/reviser.py** - Core revision logic
  - Added `ephemeral_removed` field to `RevisionResult`
  - Added `_remove_ephemeral_nodes()` to revise flow (step 2/5)
  - Added `EPHEMERAL_PATTERNS` and `BOILERPLATE_TERMS` constants
  - Expanded `OPPOSITE_WORDS` with matrix/linear algebra distinctions
  - Raised default `similarity_threshold` from 0.85 to 0.92

- **tiny_mind/__main__.py** - CLI
  - Changed revise to save by default (was opt-in `--save`, now opt-out `--no-save`)

## Revision Flow (5 steps)
1. Undo bad merges from previous runs
2. Remove ephemeral/boilerplate nodes (new)
3. Deduplication
4. Contradiction detection
5. Prune low-confidence orphans

## Key Decisions Made

1. **Similarity threshold raised to 0.92** - Was 0.85, caused false merges like "regular matrix" → "real matrix"

2. **Ephemeral patterns to remove** (regardless of connection count):
   - Variable placeholders: `matrix A`, `vector x`, single letters
   - Function variables: `function f`, `f(x)`
   - Example equations: `y = sin 2x`
   - Boilerplate: Cengage Learning, Copyright, electronic rights

3. **Opposite word pairs expanded** - Added matrix/linear algebra terms:
   - `(regular, real)`, `(symmetric, antisymmetric)`, `(row, column)`, etc.

4. **Domain tracking discussion** (not yet implemented):
   - User asked about polysemy ("function" in math vs programming vs social)
   - Recommendation: Separate nodes with qualified names ("mathematical function")
   - Domain as single value per node (each node = one sense)
   - Optional `shares_name_with` edges for homonyms
   - Extraction prompts could qualify ambiguous terms with domain

## Open Questions / Future Work

1. **Domain-aware extraction** - Update prompts to qualify ambiguous terms
2. **Domain-aware deduplication** - Only merge if domains match
3. **Neighborhood similarity for merging** - Use connections to infer if two nodes are same concept
4. **Add `domain` field or use `properties['domain']`** - User preference TBD

## Test Commands
```bash
# Run revision (now saves by default)
python -m tiny_mind revise

# Check graph state
python -m tiny_mind viz

# Verify ephemeral nodes removed
python -c "
import json
with open('tiny_mind.json') as f:
    data = json.load(f)
nodes = list(data['graph']['nodes'].keys())
print(f'Total nodes: {len(nodes)}')
"
```

## Context
- TinyMind is a "baby intelligence" PoC with curiosity-driven learning
- Knowledge graph has ~560 nodes from calculus textbook + MML
- Revision system performs deduplication, contradiction resolution, pruning
- Previous session implemented curiosity system (wonder, explore, ponder)
