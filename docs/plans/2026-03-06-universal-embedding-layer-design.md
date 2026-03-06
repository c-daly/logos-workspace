# Universal Embedding Layer: Holistic Knowledge Representation

**Date**: 2026-03-06
**Status**: Design approved
**Relates to**: JEPA-CLIP Translator PoC, CWM-G/CWM-A/CWM-E unification, logos #496

## Goal

Give every node on the HCG a single embedding in a shared representational space, regardless of how the knowledge was acquired -- text, video, audio, or inference. Sophia reasons over all knowledge uniformly. A memory of watching a dog run, the word "dog," and the sound of barking are all first-class knowledge in the same space.

This is the concrete realization of what LOGOS has always claimed: CWM-A, CWM-G, and CWM-E are aspects of the same graph, not separate systems.

## Context

### Prior Work

The JEPA-CLIP Translator PoC (`PoCs/jepa_clip_translator/`) demonstrated that V-JEPA embeddings (1024-dim) can be mapped toward CLIP image embeddings (768-dim) with 0.85+ cosine similarity using a residual translator trained on 7K MSR-VTT videos with Claude-guided architecture search.

### Key Insight

The translator's value isn't its final output (a CLIP approximation). It's the internal representation learned by the shared trunk. By training bidirectionally -- JEPA in, reconstruct CLIP; CLIP in, reconstruct JEPA -- the trunk is forced to learn what's *common* between modalities. That internal space becomes the universal embedding. CLIP is the training signal, not the product.

Text must also go through the encoder to get a universal embedding. Raw CLIP text embeddings and translated JEPA embeddings are in different spaces -- only the trunk's internal representation is truly shared.

## Architecture

### Multi-Head Autoencoder

```
JEPA input (1024) ---> [JEPA head] ---\
                                       +--> [Shared Trunk] --> internal repr (universal embedding)
CLIP text input (768) -> [Text head] --/                           |
                                                              [Reconstruction heads]
                                                             /                      \
                                                   CLIP target                  JEPA target
```

**Input heads**: One per modality. Each projects its native embedding dimension into the trunk's internal dimension.

**Shared trunk**: Residual blocks that learn the universal representation. The second-to-last layer is the universal embedding.

**Reconstruction heads**: Training-only. Each reconstructs the *other* modality's embedding. Discarded at inference.

### Training Strategy

**Phase 1 (current PoC evolution):**
- JEPA head + Text head, bidirectional reconstruction
- Train on MSR-VTT: video-caption pairs provide paired JEPA and CLIP text embeddings
- Loss: reconstruction of the opposite modality from trunk representation
- Validate with retrieval metrics (R@1, R@5, R@10) -- the real test of whether the space works

**Phase 2 (audio):**
- Add audio input head
- Train on video-with-audio data: audio embeddings paired with existing JEPA/text pairs
- Freeze or lightly fine-tune trunk; primarily train the new head
- No dependency on CLIP for audio -- trains directly against the learned universal space

**Phase 3+ (new modalities):**
- Same pattern: add input head, train against the universal space
- Each new modality expands what Sophia can perceive without changing the representation

### Inference (after training)

1. Input arrives (text, video, audio, or raw embedding)
2. Input passes through its modality-specific head
3. Extract second-to-last trunk layer = universal embedding
4. Store in Milvus / query against existing embeddings
5. Sophia retrieves matched nodes from Neo4j for reasoning

## Integration

### Hermes (embedding service)
- New endpoint: accepts input + modality tag, returns universal embedding
- Hosts the trained multi-head autoencoder
- Replaces current embedding approach -- all embeddings go through the universal encoder

### Sophia (cognitive core)
- `HCGClient` gains `store_embedding(node_uuid, vector, provenance)` and `query_by_embedding(vector, top_k)`
- CWM-G nodes are regular HCG nodes whose embeddings came from the JEPA head
- No special handling per modality -- Sophia sees one embedding space

### Milvus
- Single collection, keyed by node UUID
- Vector dimension = trunk internal dimension (TBD based on experiments)
- Provenance metadata per embedding (model version, source modality, timestamp)

### Model Migration
- When the encoder improves, create a new Milvus collection
- New nodes get embeddings from the new model
- Old nodes re-embedded opportunistically (on access or via maintenance scheduler)
- Provenance metadata tracks which model produced each embedding

## What Must Be Proven First

1. **Retrieval works**: R@K metrics on the current best checkpoint. Cosine similarity is encouraging but retrieval is the real test.
2. **Bidirectional training converges**: The current PoC is unidirectional (JEPA -> CLIP). Adding the reverse direction and extracting trunk representations is untested.
3. **Trunk representations are discriminative**: The internal space must preserve fine-grained distinctions, not just global structure.

## Implementation Path

Build from both ends toward a handshake:

**Notebook side:**
1. Run retrieval evaluation on current best checkpoint (blocked on power restoration)
2. Refactor translator into multi-head autoencoder with bidirectional training
3. Validate trunk embeddings work for cross-modal retrieval
4. Export trained encoder as a loadable module with a clean interface

**Sophia side:**
1. Add universal embedding storage to HCGClient (store/query against Milvus)
2. Add provenance metadata to embedding records
3. Wire up Hermes endpoint for embedding requests

**Handshake:** Sophia sends raw input to Hermes, Hermes runs it through the universal encoder, returns the embedding, Sophia stores it and queries against it.

## Non-Goals (for now)

- **Multi-modal fusion per node**: nodes with multiple sources (text + video + audio) get one embedding for now. Fusion strategy is an empirical question for later.
- **Collection versioning infrastructure**: provenance metadata supports this, but the migration machinery isn't designed yet. Start with one collection.
- **Real-time encoding**: batch is fine. Optimize for throughput later.

## Future Implications

If this works, LOGOS achieves something fundamental: a knowledge graph where every piece of knowledge -- regardless of origin modality -- is represented in a single space that Sophia can reason over uniformly. Perception, language, and abstraction become aspects of the same representation, not separate systems bolted together.
