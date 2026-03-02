# Cognitive Learning Architecture Proposals for Sophia

**Research Summary**: Architectural options for bootstrapping action learning and knowledge graph emergence without hardcoded schemas, based on 2024-2025 state-of-the-art research.

---

## Executive Summary

Three viable architectural paths emerged from research:

1. **Vision-Language-Action (VLA) Unified Space** - Use emerging VLA models as unified embedding foundation
2. **Agentic Graph Reasoning** - LLM-guided iterative graph expansion with topology-driven exploration
3. **Hybrid: VLA Embeddings + Curiosity-Driven Topology** - Combine unified embeddings with graph emergence mechanisms

Each avoids the V-JEPA→CLIP alignment problem while enabling the core learning vision.

---

## Proposal 1: Vision-Language-Action (VLA) Models as Unified Space

### Overview
Replace separate V-JEPA + CLIP embeddings with a single **Vision-Language-Action model** that natively unifies all three modalities in shared latent space.

### Key Research Foundation
- **100+ VLA architectures** surveyed in 2024-2025 systematic reviews
- Models like **RT-2, OpenVLA, Octo, EmbodiedOneVision** demonstrate unified representations
- Hierarchical/late fusion achieves highest manipulation success and generalization
- Transformer-based with cross-modal attention directly generating control commands

### How It Works for Sophia

**Unified Embedding Space:**
- Visual observations → VLA encoder → shared latent vector
- Language descriptions (via Hermes) → VLA encoder → same latent space
- Action capabilities → VLA decoder → executable commands
- **No alignment needed** - vision, language, action share native representation

**Learning Loop:**
1. Talos provides sensor observations → VLA encodes to embedding
2. Sophia executes capability → observes state transition in embedding space
3. Hermes requests semantic context → queries VLA for language grounding
4. Graph edges form connecting embeddings: `state_before --capability--> state_after`
5. Confidence scores track prediction accuracy

**Mental Simulation:**
- Sophia imagines: "grasp cup" (language) → VLA retrieves visual embedding trajectory
- Plans by simulating embedding paths: current_state → [capabilities] → goal_state
- Geometric patterns emerge: parallel trajectories, inverse actions, compositional sequences

### Feasibility Assessment

**Pros:**
- ✅ Solves unified space problem natively (no forced alignment)
- ✅ Active research area with multiple open implementations
- ✅ Designed for embodied robotics (fits Talos integration)
- ✅ Supports imagination: language ↔ visual dynamics
- ✅ Hierarchical fusion proven effective for generalization

**Cons:**
- ⚠️ Large model size (memory constraints similar to V-JEPA)
- ⚠️ May require fine-tuning for specific sensor modalities
- ⚠️ Most models focus on manipulation, not full cognitive architecture
- ⚠️ Availability varies (check which have open weights)

**Implementation Path:**
1. Survey available VLA models with open weights (RT-2, OpenVLA, etc.)
2. Test memory footprint with LoRA fine-tuning on cloud GPU
3. Integrate VLA embeddings into HCG as node representations
4. Implement curiosity-driven edge formation (see Proposal 3)

**Estimated Viability:** **High** - Direct solution to unified space problem with proven robotics applications

---

## Proposal 2: Agentic Deep Graph Reasoning (LLM-Guided Topology)

### Overview
Use **iterative LLM-driven expansion** where Hermes/LLM generates concepts/relationships based on current graph topology, creating self-organizing structure without hardcoded schemas.

### Key Research Foundation
- **Agentic Deep Graph Reasoning** (2025): LLM recursively expands graph using current structure to guide next prompts
- **AutoSchemaKG**: Schema induction from 50M+ documents - categories emerge from statistical patterns
- **Retrieval-Augmented Curiosity (RAC)**: Autonomous question generation drives edge discovery

### How It Works for Sophia

**Bootstrap Phase:**
```
Initial graph: [self, thing, concept] + capability nodes from Talos
```

**Expansion Loop:**
1. **Curiosity Trigger**: Low-confidence edges, disconnected nodes, prediction errors
2. **Query Formation**: Sophia asks Hermes: "What relationships exist between [node_A] and [node_B]?"
3. **LLM Reasoning**: Hermes analyzes embedding similarity, generates relationship hypotheses
4. **Graph Integration**: New edges with confidence scores; new nodes if concepts emerge
5. **Topology Feedback**: Next queries informed by graph structure (explore gaps, bridge clusters)

**Result**: Scale-free networks with:
- Hub nodes (frequently referenced concepts like "grasp", "reach")
- Modular communities (semantic domains cluster)
- Bridge nodes (concepts linking domains)
- Confidence-weighted edges (prunable low-confidence connections)

### Feasibility Assessment

**Pros:**
- ✅ No unified embedding required - works with separate V-JEPA + CLIP
- ✅ TinyMind already demonstrated this pattern at small scale
- ✅ Statistical emergence at scale prevents trivia accumulation
- ✅ Topology-driven exploration focuses on meaningful gaps
- ✅ Confidence scores enable continuous refinement

**Cons:**
- ⚠️ Requires massive interaction scale (millions) to see emergence
- ⚠️ LLM inference cost per edge formation (optimization needed)
- ⚠️ May not support "imagination" as smoothly (no visual generation)
- ⚠️ Depends heavily on Hermes quality for semantic grounding

**Implementation Path:**
1. Extend TinyMind's edge formation mechanism to handle V-JEPA + CLIP embeddings
2. Implement curiosity triggers: DRND (novelty detection), gap analysis, contradiction detection
3. Optimize LLM queries: batch edge formation, use smaller models for routine queries
4. Add temporal confidence decay and pruning mechanisms
5. Monitor emergence: track hub formation, modularity, bridge nodes

**Estimated Viability:** **Medium-High** - Proven concept (TinyMind) but needs scale and optimization

---

## Proposal 3: Hybrid - VLA Embeddings + Curiosity-Driven Graph Topology

### Overview
Combine the **unified embedding space** of VLA models with **curiosity-driven graph emergence** mechanisms, getting benefits of both approaches.

### How It Works for Sophia

**Foundation Layer: VLA Unified Embeddings**
- All observations/concepts/actions encoded in shared VLA latent space
- Enables mental simulation and geometric reasoning

**Learning Layer: Curiosity-Driven Topology**
- **Uncertainty-based exploration**: DRND measures novelty in embedding space
- **Edge formation**: When Sophia observes state transition, queries VLA for relationship type
- **Confidence inference**: Embedding similarity predicts confidence for unobserved edges
- **Topology refinement**: Low-confidence edges pruned; hub nodes reinforced

**Integration:**
```
1. Observe transition: state_A --[capability]--> state_B (VLA embeddings)
2. Measure uncertainty: cosine distance between predicted vs actual state_B
3. If high uncertainty → curiosity trigger → form exploration plan
4. Edge creation: relationship type from VLA decoder, confidence from embedding similarity
5. Graph structure guides next exploration (bridge gaps, test hypotheses)
```

### Key Mechanisms from Research

**From Graph Topology Research:**
- Scale-free structure emerges from frequency (hubs = often-referenced concepts)
- Community detection via embedding clustering
- Negative sampling prevents over-connection
- Critical state dynamics: information propagates efficiently without fragmentation

**From VLA Research:**
- Cross-modal attention aligns observations with language naturally
- Late fusion enables multi-level integration (visual features + semantic concepts)
- Action generation grounded in unified space

**From Action Learning Research:**
- ConditionNET learns preconditions/effects in data-driven manner
- Consistency in feature representations enables anomaly detection
- Task-oriented active learning focuses on regions where model is uncertain

### Feasibility Assessment

**Pros:**
- ✅ Best of both worlds: unified space + meaningful topology
- ✅ Curiosity focuses learning on informative regions (efficient)
- ✅ Supports imagination (VLA) and abstract reasoning (graph structure)
- ✅ Confidence-based refinement prevents trivia accumulation
- ✅ Embedding space naturally supports geometric pattern discovery

**Cons:**
- ⚠️ Most complex architecture (two major components)
- ⚠️ VLA model size + graph operations may strain resources
- ⚠️ Requires careful tuning of curiosity triggers and confidence thresholds
- ⚠️ Integration complexity: how do VLA predictions inform graph structure?

**Implementation Path:**
1. Deploy VLA model with confirmed memory budget (cloud GPU testing)
2. Implement curiosity metrics: DRND for novelty, embedding distance for uncertainty
3. Edge formation protocol: VLA generates relationship, embedding similarity sets confidence
4. Add topology monitoring: track hub formation, modularity, emergence metrics
5. Refinement loop: prune low-confidence, reinforce high-traffic paths
6. Test mental simulation: language → VLA → predicted embedding trajectory

**Estimated Viability:** **High** - Addresses all core requirements, though highest implementation complexity

---

## Comparison Matrix

| Criterion | VLA Only | Agentic Graph | Hybrid VLA + Graph |
|-----------|----------|---------------|-------------------|
| **Unified Space** | ✅ Native | ⚠️ Separate embeddings | ✅ Native |
| **Memory Req** | High | Medium | Highest |
| **Scale to Emerge** | Medium | Very High | Medium |
| **Mental Simulation** | ✅ Direct | ⚠️ Indirect | ✅ Direct |
| **Topology Learning** | Limited | ✅ Core strength | ✅ Core strength |
| **LLM Dependency** | Low | High | Medium |
| **Proven Research** | ✅ 100+ models | ✅ Scale-free nets | ⚠️ Novel combo |
| **Avoids Trivia** | Via training | Via scale/topology | Via confidence + topology |
| **Implementation** | Medium | Medium | High |

---

## Recommended Path Forward

### Short-term (While waiting for VL-JEPA or selecting VLA model):
**Option A: Extend TinyMind** (Proposal 2 foundation)
- Implement confidence-based pruning
- Add curiosity triggers (DRND, gap detection)
- Test at larger scale (simulate thousands of observations)
- **Goal**: Validate that topology emergence produces meaningful structure

**Option B: VLA Model Survey** (Prepare for Proposal 1 or 3)
- Identify available VLA models with open weights
- Test memory footprint and inference speed
- Evaluate fit for Talos sensor modalities
- **Goal**: Confirm VLA viability before committing architecture

### Medium-term (Once VLA access secured):
**Implement Proposal 3 (Hybrid)** in phases:
1. **Phase 1**: VLA integration - replace separate embeddings, test mental simulation
2. **Phase 2**: Basic curiosity - uncertainty-driven exploration, simple edge formation
3. **Phase 3**: Topology emergence - confidence pruning, hub detection, community analysis
4. **Phase 4**: Scale testing - accumulate thousands of observations, monitor emergence

### Long-term (Full system):
- Hierarchical memory integration (ephemeral → short-term → long-term)
- Persona diary + reflection using graph patterns
- Planning via backward chaining through learned topology
- Continuous refinement: confidence updates, pruning, schema-less evolution

---

## Key Insights from Research

### What Makes Structure "Meaningful" (Not Trivia)

1. **Statistical Emergence Over Enumeration**: Processing massive scale (millions of observations) surfaces recurring patterns; isolated trivia lacks co-occurrence to form strong edges

2. **Topology-Guided Exploration**: Next queries informed by current graph → reinforces coherent clusters, discourages random sprawl

3. **Confidence as Selection Pressure**: Low-confidence edges pruned; high-confidence hubs reinforced → evolutionary pressure toward utility

4. **Semantic Grounding in Embeddings**: Edge weights from vector similarity ensure connections reflect meaning, not arbitrary linkage

5. **Critical State Dynamics**: Self-organizing systems reach criticality where information propagates efficiently without fragmentation

### How to Avoid TinyMind's Scale Problem

- **Hierarchical fusion** (VLA research): Multi-level integration prevents flat trivia accumulation
- **Negative sampling** (graph research): Explicitly model non-edges to sharpen boundaries
- **Task-oriented active learning** (action research): Focus on regions relevant to goals, not exhaustive coverage
- **Temporal confidence decay**: Facts have time-limited validity; outdated edges naturally pruned

### Addressing the Genetic Programming "Nonsense Composition" Issue

**Why GP failed**: Primitives composed syntactically without semantic constraints
**How proposed architectures avoid it:**
- **VLA**: Actions generated in unified semantic space (not unconstrained syntax)
- **Graph topology**: Relationships emerge from observed co-occurrence (not free combination)
- **Confidence scores**: Invalid compositions have low confidence, get pruned
- **Embedding similarity**: Only semantically close concepts link; arbitrary compositions have high distance

---

## Critical Open Questions

1. **Scale threshold for emergence**: How many observations before meaningful topology appears? (Research suggests 10K-1M+ depending on domain complexity)

2. **VLA fine-tuning for non-standard sensors**: Can VLA models adapt to specialized Talos sensors, or need retraining?

3. **Curiosity trigger tuning**: What confidence thresholds and novelty metrics produce useful exploration vs noise?

4. **Graph-VLA integration**: How exactly do VLA predictions inform edge confidence and topology? (Implementation detail needing experimentation)

5. **Computational budget**: Can Sophia run VLA inference + graph operations + curiosity evaluation in real-time during interaction?

---

## Sources

### Graph Topology Emergence
- [Agentic Deep Graph Reasoning Yields Self-Organizing Knowledge Networks](https://arxiv.org/abs/2502.13025)
- [AutoSchemaKG: Autonomous Knowledge Graph Construction](https://arxiv.org/html/2505.23628v1)
- [Retrieval Augmented Curiosity: An Autonomous Approach to Knowledge Expansion](https://medium.com/@ryanbgoldberg/retrieval-augmented-curiosity-an-autonomous-approach-to-knowledge-expansion-2d3dc374e08f)
- [CosUKG: A Representation Learning Framework for Uncertain Knowledge Graphs](https://www.mdpi.com/2227-7390/12/10/1419)
- [Uncertainty Management in the Construction of Knowledge Graphs: A Survey](https://arxiv.org/html/2405.16929v2)

### Vision-Language-Action Models
- [Multimodal fusion with vision-language-action models for robotic manipulation: A systematic review](https://www.sciencedirect.com/science/article/pii/S1566253525011248)
- [Vision-Language-Action Models: Concepts, Progress, Applications and Challenges](https://arxiv.org/html/2505.04769v1)
- [Vision Language Action Models in Robotic Manipulation: A Systematic Review](https://arxiv.org/html/2507.10672v1)
- [EmbodiedOneVision: Interleaved Vision-Text-Action Pretraining for General Robot Control](https://arxiv.org/html/2508.21112v1)
- [A review of embodied intelligence systems: a three-layer framework](https://www.frontiersin.org/journals/robotics-and-ai/articles/10.3389/frobt.2025.1668910/full)

### Action Learning from Observation
- [ConditionNET: Learning Preconditions and Effects for Execution Monitoring](https://arxiv.org/abs/2502.01167)
- [Safe Learning of PDDL Domains with Conditional Effects](https://arxiv.org/html/2403.15251v1)
- [Task-Oriented Active Learning of Model Preconditions](https://www.ri.cmu.edu/publications/task-oriented-active-learning-of-model-preconditions-for-inaccurate-dynamics-models/)
- [A Survey of Embodied Learning for Object-Centric Robotic Manipulation](https://arxiv.org/html/2408.11537v1)

---

**Generated**: 2026-01-07
**Research Coverage**: 2024-2025 state-of-the-art
**Focus**: Feasible architectural options for Sophia's learning bootstrap problem
