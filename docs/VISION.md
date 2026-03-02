# LOGOS Vision

## What LOGOS Is

LOGOS is a non-linguistic cognitive architecture built on a core assertion: **text is a poor substrate for thought.** Language models trained on text lack the grounded common sense that comes from non-linguistic experience. LOGOS addresses this by separating cognition from language entirely.

Sophia (the cognitive core) reasons over a knowledge graph (HCG) where nodes, edges, and traversals are the primitives of thought — not words. Language is handled by Hermes as an I/O utility: a translation layer between human language and graph structures, never the medium of reasoning itself.

Grounded understanding comes from JEPA (Joint Embedding Predictive Architecture) models that learn physical/sensory representations without text. These form the Grounded working memory (CWM-G) — a layer of common sense that language models fundamentally cannot provide. The Abstract working memory (CWM-A) captures conceptual and relational knowledge. The Emotional working memory (CWM-E) tracks emotional and persona states. All three are aspects of the same graph, not separate systems.

Planning is a core capability. The HCGPlanner performs backward-chaining over REQUIRES/CAUSES edges to produce executable plans, and Talos provides the embodiment layer — abstracting hardware (or simulated hardware) so that Sophia's plans can drive real-world interaction. The system is designed to be situated: perceiving, reasoning, planning, and acting in a physical environment.

## Domains

*Practice areas that organize work across the project. Updated from commits, PRs, and design docs. Reviewed against goals at regular intervals.*

- **Cognition** — graph-native reasoning over HCG; the core of Sophia
- **Language** — translation between human language and graph structures (Hermes)
- **Perception** — grounded understanding via JEPA, sensory input processing (CWM-G)
- **Planning** — backward-chaining over REQUIRES/CAUSES edges, Process nodes, plan execution
- **Embodiment** — hardware abstraction, simulation, real-world actuation (Talos)
- **Memory** — hierarchical storage, reflection, episodic learning
- **Ontology** — type system, IS_A edges, reified model, graph-native schema
- **Infrastructure** — shared tooling, CI/CD, observability, testing, developer experience
- **Research** — exploratory experiments, PoCs, and learning architecture proposals that inform the core architecture
- **Tooling** — developer workflow automation, agent orchestration, and productivity tools that support LOGOS development

## Goals

1. **Complete the cognitive loop** — in progress
   The core perception → reasoning → action cycle works end-to-end (Hermes extracts entities/relations, Sophia stores and retrieves from HCG, context enriches LLM responses). Expand with: feedback processing, planning integration, multi-turn memory, context quality improvements.

2. **Grounded perception via JEPA** — in progress
   Give Sophia non-linguistic common sense through JEPA models that learn physical/sensory representations. PoC exists in sophia (#76) with tests, docs, and backend integration. This is what makes LOGOS fundamentally different from text-only systems — cognition grounded in experience, not language.

3. **Flexible ontology** — in progress
   Replace rigid schema-typed nodes with a structure-typed model where meaning comes from IS_A edges and graph position. Core reified model is implemented (PR #490); stale artifacts, downstream queries, and validation need updating.

4. **Memory and learning** — not started
   Transform LOGOS from a stateless system into one that learns from experience. Hierarchical memory (ephemeral → short-term → long-term), event-driven reflection, selective diary entries, episodic learning. Spec exists (#415), prerequisites need completing.

5. **CWM unification** — not started
   CWM-A (abstract), CWM-G (grounded), and CWM-E (emotional) are semantically distinct aspects but all live on the same graph. The current module-level separation (separate packages, raw Cypher) needs to become ontology-level (type definitions, HCG client). The distinction is real; the implementation boundary is wrong (#496).

6. **Planning and execution** — in progress
   Sophia reasons in order to act — whether that's answering a user's question ("how do I get to LA by tomorrow?") or driving a robot arm from point A to point B. The HCGPlanner does backward-chaining over REQUIRES/CAUSES edges to produce executable plans represented as Process nodes. Planner stub still exists alongside the real implementation (#403). As Talos matures, plans connect to real-world actuation.

7. **Embodiment via Talos** — in progress
   Talos abstracts hardware behind a consistent HAL. Current state is a simulation scaffold — `docs/proposed_docs/TALOS_IMPROVEMENTS.md` outlines the path to physics-backed simulation (ROS2/Gazebo/MuJoCo). Connects Sophia's plans to real-world actuation and perception feeds (camera frames, IMU, joint state) back into the cognitive loop.

8. **Observability** — in progress
   OTel instrumentation exists across services via `logos_observability`. Gaps remain in Apollo SDK integration, Hermes/Sophia endpoint-level spans, and cross-service testing.

9. **Documentation** — in progress
   Proposed replacement docs exist in `docs/proposed_docs/`. Need to execute the manifest: move into place, archive stale docs, per-repo cleanup.

10. **Testing and infrastructure** — in progress
    Test suites pass across repos with real infrastructure. Remaining: logos coverage improvement, standardized test conventions, OpenAPI contract tests for Hermes.

## Non-Goals

- **Text-based intelligence** — LOGOS is explicitly not an LLM wrapper. Language is I/O, not cognition
- **Production deployment** — focus is on capability development, not scaling or hardening
- **Multi-user** — single-operator system for now; auth (#311) is deferred
- **Real-time hardware** — Talos abstracts hardware but real-time control is out of scope until the cognitive layer is solid

## Current Priorities

1. Cognitive loop expansion (feedback processing, multi-turn memory)
2. Grounded perception via JEPA (CWM-G maturation)
3. Flexible ontology cleanup (stale artifacts, downstream queries)
4. CWM unification (cleanup — not blocking memory work)
