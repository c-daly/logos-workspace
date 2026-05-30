# LOGOS Vision

> **Reading this against the code.** This document states the *full* vision — an
> open-ended research program. For where the code actually stands today (roughly
> 35–40% of the near-term vision; foundation strong, cognitive core largely spec),
> read `docs/STATUS.md`. The concrete near-term milestone is **Minimum Thinking
> Sophia (MTS)** — the point where LOGOS becomes a small mind that runs: it grows
> and maintains its graph, reasons/plans over it, learns from feedback, runs as a
> persistent loop, and surfaces its state in Apollo. MTS is six sprints out; the
> plan is `docs/plans/2026-05-29-roadmap-to-mts.md`. Everything past MTS
> (grounding, embodiment, the situated agent, the papers) is **Horizon 2**.
> Aspirational capabilities below are marked; goal status lines reflect the
> 2026-05-29 code-grounded audit.

## What LOGOS Is

LOGOS is a non-linguistic cognitive architecture built on a core assertion: **text is a poor substrate for thought.** Language models trained on text lack the grounded common sense that comes from non-linguistic experience. LOGOS addresses this by separating cognition from language entirely.

Sophia (the cognitive core) reasons over a knowledge graph (HCG) where nodes, edges, and traversals are the primitives of thought — not words. Language is handled by Hermes as an I/O utility: a translation layer between human language and graph structures, never the medium of reasoning itself.

Grounded understanding comes from JEPA (Joint Embedding Predictive Architecture) models that learn physical/sensory representations without text. These form the Grounded working memory (CWM-G) — a layer of common sense that language models fundamentally cannot provide. The Abstract working memory (CWM-A) captures conceptual and relational knowledge. The Emotional working memory (CWM-E) tracks emotional and persona states. All three are aspects of the same graph, not separate systems.

Planning is a core capability *by design*. An `HCGPlanner` exists in the foundry that performs backward-chaining over REQUIRES/CAUSES edges to produce executable plans — but it is **not yet invoked anywhere in the running loop**, and a planner stub still co-exists with it (logos #403). Talos is intended as the embodiment layer — abstracting hardware (or simulated hardware) so that Sophia's plans can drive real-world interaction — and currently stands at a simulation scaffold. The system is *designed* to be situated: perceiving, reasoning, planning, and acting in a physical environment. (Aspirational; situated operation is Horizon 2, Goal 9.)

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

1. **Complete the cognitive loop** — in progress (the spine of MTS)
   The *ingestion* arc works end-to-end (Hermes extracts entities/relations, Sophia stores to HCG and retrieves context that enriches LLM responses); the centralized Redis event bus (`logos_events`) and runtime ontology pub/sub between Sophia and Hermes are live on main, and the maintenance scheduler is wired. **But the loop is not yet self-correcting or autonomous:** embeddings silently fail to persist at ingestion (the keystone bug, sophia#146 / logos#528), so the type classifier degrades to a fallback and emergence starves; feedback is a stub that logs and discards; the scheduler has essentially no reasoning jobs; and Sophia runs request/response, not as a persistent event-driven process. Closing this is the through-line of the six MTS sprints: stabilize the spine (Sprint 1) → feedback mutates the graph (Sprint 2) → orchestrator + event-driven loop (Sprint 3) → K-lines/curiosity/gain (Sprint 4) → memory + planning-in-loop (Sprint 5) → demonstrate (Sprint 6). See `docs/plans/2026-05-29-roadmap-to-mts.md`.

2. **Grounding and physical knowledge** — research active (integration deferred)
   Give Sophia intuitive physical common sense — the ability to recognize when something is physically ridiculous, or to anticipate what happens when a robot takes a corner too fast. This is what makes LOGOS fundamentally different from text-only systems: cognition grounded in experience, not language. The implementation approach is JEPA (Joint Embedding Predictive Architecture) models that learn physical/sensory representations without text, feeding into CWM-G. PoC exists in sophia (#76) with pluggable backend, tests, docs, and API shape validation. **Research is active:** the V-JEPA token-grid PoC (logos-workspace PR #4) ran 80+ experiments translating V-JEPA temporal tokens into CLIP space, achieving txt_R@1 = 0.371 (target 0.42). The design and plan are `docs/plans/2026-03-05-jepa-clip-translator-design.md` / `-plan.md`. **Integration into Sophia is Horizon 2** — deferred until MTS is reached and the cognitive loop matures — but the research track is validating feasibility now.

3. **Flexible ontology** — in progress
   Replace rigid schema-typed nodes with a structure-typed model where meaning comes from IS_A edges and graph position. Core reified model is implemented (PR #490); ontology hierarchy restructured (#510). CWM-A, CWM-G, and CWM-E are semantically distinct aspects of the same graph — the current module-level separation (separate packages, raw Cypher) needs to become ontology-level (type definitions, HCG client, #496). Remaining: downstream repo updates (#460, #461), type_definition UUID migration (#515), capability catalog (#465).

4. **Memory and learning** — not started
   Transform LOGOS from a stateless system into one that learns from experience. Hierarchical memory (ephemeral → short-term → long-term), event-driven reflection, selective diary entries, episodic learning. Spec exists (#415), prerequisites need completing. The Redis event bus and maintenance scheduler landed as foundational infrastructure. Testing sanity (#416, priority:critical) must happen first.

5. **Planning and execution** — early / mostly aspirational
   Sophia should reason in order to act — whether that's answering a user's question ("how do I get to LA by tomorrow?") or driving a robot arm from point A to point B. An `HCGPlanner` exists in the foundry that does backward-chaining over REQUIRES/CAUSES edges to produce plans as Process nodes, **but it is not invoked anywhere in the running loop**, and a planner stub still co-exists with it (#403). Planning-in-loop — new knowledge enabling a goal triggers the planner — is **Sprint 5** of the MTS plan, gated on the feedback loop and memory tiers landing first; the planner-vs-stub reconciliation (#403/#464) is a decision deferred to that sprint. As Talos matures (Horizon 2), plans connect to real-world actuation. Downstream flexible-ontology updates (#460, #464) are prerequisites.

6. **Embodiment via Talos** — paused
   Talos abstracts hardware behind a consistent HAL. Current state is a simulation scaffold — `docs/TALOS_IMPROVEMENTS.md` outlines the path to physics-backed simulation (ROS2/Gazebo/MuJoCo). Connects Sophia's plans to real-world actuation and perception feeds (camera frames, IMU, joint state) back into the cognitive loop. Correctly deprioritized until the cognitive layer is solid.

7. **Infrastructure and observability** — in progress
   CI discipline tooling (branch naming, issue linkage) landed across all repos. Reusable workflows pinned to ci/v2. Centralized Redis and `logos_events` package provide the event backbone. Port standardization completed. Docker stacks aligned. OTel instrumentation exists across services via `logos_observability`; Apollo OTel integration complete (PR #156, closing #340, #341, #342). Gaps remain in endpoint-level spans (#335, #338), cross-service testing (#321), and Hermes OTel documentation (#339). Remaining: standardize repos (#433), developer scripts (#409), test data seeder (#481).

8. **Documentation and testing** — in progress
   Documentation: 13 duplicate ecosystem docs removed, CLAUDE.md consolidated across all 6 repos, SPEC.md updated with Redis/logos_events, READMEs corrected. Testing: suites pass across repos with real infrastructure. Remaining: developer onboarding guide (#135), proposed doc execution (#447), logos coverage improvement, OpenAPI contract tests (#91), standardized test conventions (#420).

9. **Situated cognitive agent** — deferred (prerequisites: Goals 1, 4, 5, 6) (#521)
   LOGOS operates as a persistent, situated cognitive agent — perceiving, reasoning, learning, and acting across an ecosystem of communication channels and devices. Slack messages, device commands, sensor feeds, and proactive outreach are all uniform from Sophia's perspective: plan actions routed through actuators (Hermes/Apollo for language channels, Talos for hardware). Responses and actions draw on the full HCG, enriched continuously by autonomous KG maintenance, reflection, inference, and entity resolution. A continuously-learning agent generates unbounded data — local deployment is viable for development, but sustained operation requires personal cloud deployment with elastic storage. The optimization target is search/retrieval efficiency; the graph grows without bound and that's the point. Single-user.

## Non-Goals

- **Text-based intelligence** — LOGOS is explicitly not an LLM wrapper. Language is I/O, not cognition
- **Production deployment** — focus is on capability development, not multi-user scaling or hardening. Personal cloud deployment for sustained single-user operation (Goal 9) is distinct from productization
- **Multi-user** — single-operator system for now; auth (#311) is deferred
- **Real-time hardware** — Talos abstracts hardware but real-time control is out of scope until the cognitive layer is solid

## Current Priorities

Driven by **MTS Sprint 1 — "Make the built spine actually run & be trusted"**
(`docs/plans/2026-05-29-roadmap-to-mts.md`). Critical path: logos#528 → sophia#146 → live-verified #505.

1. **Keystone:** fix embedding persistence (logos#528 upsert-by-uuid + schema, then sophia#146 warn-only swallow) — without this the classifier and emergence both starve.
2. Finish & merge #505 emergent type discovery, live-verified on real infra.
3. De-vacuum integration tests + run them in CI (logos#529); add Redis to CI (logos #526).
4. Cross-repo foundry sync (apollo/talos → v0.7.1, logos#530); FeedbackConfig port fix (sophia #142).

Sprint 2+ then closes the feedback loop, builds the orchestrator/event-driven loop,
and adds the cognitive mechanisms — see the roadmap for the full six-sprint arc.
