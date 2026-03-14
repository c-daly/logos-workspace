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
- **Infrastructure** — shared tooling, CI/CD, event bus, observability, developer experience
- **Research** — exploratory experiments, PoCs, and learning architecture proposals that inform the core architecture
- **Tooling** — developer workflow automation, agent orchestration, and productivity tools that support LOGOS development

## Goals

1. **Complete the cognitive loop** — in progress
   The core perception → reasoning → action cycle works end-to-end (Hermes extracts entities/relations, Sophia stores and retrieves from HCG, context enriches LLM responses). Centralized Redis event bus (`logos_events`) provides the backbone. Ontology pub/sub distribution keeps Sophia and Hermes type-synced at runtime. The maintenance scheduler gives Sophia autonomous graph-reasoning triggers. KG maintenance work (entity resolution, type correction, relationship inference, ontology evolution) is how this loop becomes self-correcting. Expand with: entity resolution (#503), feedback processing, planning integration, multi-turn memory.

2. **Grounding and physical knowledge** — research active (integration deferred)
   Give Sophia intuitive physical common sense — the ability to recognize when something is physically ridiculous, or to anticipate what happens when a robot takes a corner too fast. This is what makes LOGOS fundamentally different from text-only systems: cognition grounded in experience, not language. The implementation approach is JEPA (Joint Embedding Predictive Architecture) models that learn physical/sensory representations without text, feeding into CWM-G. PoC exists in sophia (#76) with pluggable backend, tests, docs, and API shape validation. **Research is active:** the V-JEPA token-grid PoC (logos-workspace PR #4) ran 80+ experiments translating V-JEPA temporal tokens into CLIP space, achieving txt_R@1 = 0.371 (target 0.42). A universal embedding layer design (`docs/plans/2026-03-06-universal-embedding-layer-design.md`) defines the multi-head autoencoder architecture. **Integration into Sophia remains deferred** until the cognitive loop matures, but the research track is validating feasibility now.

3. **Flexible ontology** — in progress
   Replace rigid schema-typed nodes with a structure-typed model where meaning comes from IS_A edges and graph position. Core reified model is implemented (PR #490); ontology hierarchy restructured (#510). CWM-A, CWM-G, and CWM-E are semantically distinct aspects of the same graph — the current module-level separation (separate packages, raw Cypher) needs to become ontology-level (type definitions, HCG client, #496). Remaining: downstream repo updates (#460, #461), type_definition UUID migration (#515), capability catalog (#465).

4. **Memory and learning** — not started
   Transform LOGOS from a stateless system into one that learns from experience. Hierarchical memory (ephemeral → short-term → long-term), event-driven reflection, selective diary entries, episodic learning. Spec exists (#415), prerequisites need completing. The Redis event bus and maintenance scheduler landed as foundational infrastructure. Testing sanity (#416, priority:critical) must happen first.

5. **Planning and execution** — in progress
   Sophia reasons in order to act — whether that's answering a user's question ("how do I get to LA by tomorrow?") or driving a robot arm from point A to point B. The HCGPlanner does backward-chaining over REQUIRES/CAUSES edges to produce executable plans represented as Process nodes. Planner stub still exists alongside the real implementation (#403). As Talos matures, plans connect to real-world actuation. Blocked on flexible ontology downstream updates (#460).

6. **Embodiment via Talos** — paused
   Talos abstracts hardware behind a consistent HAL. Current state is a simulation scaffold — `docs/proposed_docs/TALOS_IMPROVEMENTS.md` outlines the path to physics-backed simulation (ROS2/Gazebo/MuJoCo). Connects Sophia's plans to real-world actuation and perception feeds (camera frames, IMU, joint state) back into the cognitive loop. Correctly deprioritized until the cognitive layer is solid.

7. **Infrastructure and observability** — in progress
   CI discipline tooling (branch naming, issue linkage) landed across all repos. Reusable workflows pinned to ci/v2. Centralized Redis and `logos_events` package provide the event backbone. Port standardization completed. Docker stacks aligned. OTel instrumentation exists across services via `logos_observability`; Apollo OTel integration complete (PR #156, closing #340, #341, #342). Gaps remain in endpoint-level spans (#335, #338), cross-service testing (#321), and Hermes OTel documentation (#339). Remaining: standardize repos (#433), developer scripts (#409), test data seeder (#481).

8. **Documentation and testing** — in progress
   Documentation: 13 duplicate ecosystem docs removed, CLAUDE.md consolidated across all 6 repos, SPEC.md updated with Redis/logos_events, READMEs corrected. Testing: suites pass across repos with real infrastructure. Remaining: developer onboarding guide (#135), proposed doc execution (#447), logos coverage improvement, OpenAPI contract tests (#91), standardized test conventions (#420).

## Non-Goals

- **Text-based intelligence** — LOGOS is explicitly not an LLM wrapper. Language is I/O, not cognition
- **Production deployment** — focus is on capability development, not scaling or hardening
- **Multi-user** — single-operator system for now; auth (#311) is deferred
- **Real-time hardware** — Talos abstracts hardware but real-time control is out of scope until the cognitive layer is solid

## Current Priorities

1. Cognitive loop expansion (entity resolution, feedback processing, KG maintenance execution)
2. KG maintenance stories (#503, #504, #505, #506) — #503 has full design + implementation plan ready
3. Flexible ontology downstream propagation (#460, #461, #515)
4. Infrastructure hardening (remaining standardization, coverage)
