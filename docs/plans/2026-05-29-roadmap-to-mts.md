# Roadmap to Completion — "Minimum Thinking Sophia" (MTS)

**Created:** 2026-05-29
**Author:** PM pass (Claude Code), grounded in the 2026-05-29 code-grounded ecosystem audit
**Status:** Proposed — sprint plan + Sprint 1 tickets
**Related:** `docs/VISION.md` (9 goals) · `docs/STATUS.md` · `vault/10-projects/LOGOS/2026-05-29-ecosystem-audit.md` (evidence) · `vault/10-projects/LOGOS/LOGOS_Implementation_Spec.md` (Appendix A priority order)

> All claims here trace to the 2026-05-29 audit (verified `file:line` state), the spec's Appendix A
> dependency order, and the deliberate infra-before-ontology sequencing. Severity/effort are calibrated;
> deliberate documented deferrals are not treated as defects.

---

## Part 1 — What "completion" means

The full vision (sub-20W efficiency, JEPA grounding, multi-device embodiment, situated agent, developmental growth at depth, the 13-paper program) is **an open-ended research program, not a finite backlog.** Scoping "sprints to completion" against it would be dishonest. So completion is defined as a concrete, finite milestone:

> **Minimum Thinking Sophia (MTS):** the point where LOGOS stops being *infrastructure + a thin ingestion loop* and becomes *a small mind that runs* — Sophia, given a stream of experience (text for now), **autonomously grows and maintains** a non-linguistic graph, **reasons/plans** over it, **learns from feedback**, runs as a **persistent event-driven loop** (not request/response), and **surfaces its cognitive state in Apollo**. That demonstrates the core thesis end-to-end.

Everything past MTS = **Horizon 2** (research, deliberately out of this plan's scope): CWM-G/JEPA grounding integration (Goal 2), Talos embodiment (Goal 6), situated multi-device agent (Goal 9), developmental growth at depth, the efficiency program, the papers.

**Current state ≈ 35–40%.** MTS ≈ the ~95% line of the *stated near-term* vision. Six sprints.

---

## Part 2 — The sprint map

| # | Sprint goal | Advances | Exit criteria | Size¹ | ~% after |
|---|-------------|----------|---------------|-------|----------|
| **1** | **Make the built spine actually run & be trusted** | G1, G3, G7, G8 | ingest→embed→classify→emerge works on live infra and CI would catch a regression; #505 merged & live-verified | L | ~45% |
| **2** | **Close the feedback loop (Learning v0)** | G1 | feedback mutates the graph (confidence/dedup/deprecate); maintenance reasoning jobs #503/#504/#506 run | L | ~55% |
| **3** | **Orchestrator + event-driven loop + daemons** | G1; spec §8–9, A·#2/#4 | Sophia runs as a persistent process reacting to bus events; ≥1 daemon transforms a real signal | XL | ~65% |
| **4** | **K-lines + curiosity + CWM-E gain (Cognition v1)** | spec §10.2–10.4, A·#3/#5/#6 | stimulus activates a constellation; non-activation emits curiosity; CWM-E state measurably modulates processing | XL | ~75% |
| **5** | **Memory tiers + planning-in-loop** | G4, G5; spec §10.5, A·#7 | knowledge promotes ephemeral→STM→LTM by criteria; new knowledge enabling a goal triggers HCGPlanner | L | ~85% |
| **6** | **Demonstrate the thesis (MTS integration)** | G1 + Apollo; reconcile #496/#461 | recorded end-to-end session: ingest→grow ontology→reason→plan→reflect, with Apollo rendering goal→plan→diary→cognitive-state | L | **~95% = MTS** |

¹ Solo, spare-time velocity → relative effort (S/M/L/XL), not calendar weeks; calendar time scales with availability. Weight concentrates in Sprints 3–4 (the genuinely novel cognitive mechanisms).

**Ordering rationale.** The audit's central finding: the "built" spine is *silently broken* (embeddings never persist → classifier and #505 both starve) and *under-verified at the seams* (vacuous integration tests, Redis absent from CI). Do not build cognition on a foundation that's secretly a no-op. Sprint 1 = stabilization (and the highest-morale "it works end-to-end" win for a return). After that it's spec Appendix A dependency order: loop → mechanisms → memory → demonstrate. CWM-E gain (spec step 3) is grouped into Sprint 4 with K-lines/curiosity because gain control is only meaningful once there's a loop and constellations to modulate; the CWM-E state models already exist.

---

## Part 3 — Sprint 1: "Make the built spine actually run & be trusted"

**Sprint goal:** every component the audit found *built-but-secretly-broken or untested* is fixed and provably working end-to-end, and #505 (the in-flight frontier) lands live-verified.

**Definition of done (sprint exit):** Seed the demo scenario → run a `/llm` turn mentioning new entities → confirm (a) embeddings land in Milvus, (b) the type classifier assigns non-fallback types, (c) the scheduler's `type_emergence` job mints ≥1 real emergent type, (d) Apollo's dashboard shows the result — and a CI run that **fails** if any of those breaks. `feat/505-*` branches merged; STATUS regenerated.

**Critical path:** `S1-02 → S1-01 → {S1-04, S1-05}`. Others (S1-03/06/07/08) parallelize. Minimum viable sprint = **S1-02 + S1-01 + S1-05** (turns "#505 built but starved" into "the cognitive frontier demonstrably runs").

Tickets use the project template (Context / Objective / Acceptance Criteria / Notes) and `docs/PROJECT_TRACKING.md` labels.

---

### S1-01 — Sophia: embedding persistence silently fails at ingestion  🔴 KEYSTONE
`component:sophia` · `type:bug` · `priority:critical` · `domain:hcg` · **size: L** · **depends-on: S1-02**

**Context.** Live flow has 22 entity nodes in Neo4j but `hcg_*_embeddings` Milvus collections read **0** (vault: `sophia/.memory/2026-05-26-sophia-milvus-embeddings-empty-blocks-emergence.md`). `proposal_processor.py:520` calls `batch_upsert_embeddings` inside a warn-only `try/except` (`:523-528`) that swallows failure. Result: the classifier degrades to fallback `'entity'` for everything, and #505 emergence loads 0 members → "no qualifying clusters." This single bug neuters the entire maintenance/emergence arm.

**Objective.** Embeddings written at ingestion are durably persisted and retrievable; failures surface instead of being swallowed.

**Acceptance criteria.**
- Ingesting N entities via `/ingest/hermes_proposal` yields N retrievable vectors in the matching Milvus collection (asserted by an integration test against live Milvus, not a mock).
- `get_embedding(uuid)` returns a vector for every just-ingested node.
- An embedding-persistence failure raises or is surfaced (logged-and-counted with a non-zero failure signal), not silently `warning`-and-continue.
- Root cause documented in the ticket (auto-id-PK insert from S1-02, connection/flush issue, or missing `flush()`/`load()`).

**Notes.** Depends on S1-02 (upsert-by-uuid schema must be correct first). Keystone of the sprint.

---

### S1-02 — logos: HCGMilvusSync can't upsert-by-uuid (auto_id PK) + schema divergence  🔴
`component:logos` · `type:bug` · `priority:critical` · `domain:hcg` · **size: M**

**Context.** Two contradictory Milvus schemas ship in one repo: `infra/init_milvus_collections.py:65-69` makes `uuid` the PK; runtime `logos_hcg/sync.py:155-156` uses an auto-id INT64 PK with `uuid` a plain field. `upsert_embedding`/`batch_upsert_embeddings` call `collection.insert()` (`sync.py:315`), so re-syncing a uuid **appends a duplicate** rather than replacing — despite comments claiming upsert semantics (`:305`,`:314`). The guard test (`tests/infra/test_milvus_collections.py`) encodes the init-script schema (4 collections, no `id`), so it passes only in CI and masks the divergence.

**Objective.** One canonical collection schema across init + runtime, with working replace-by-uuid.

**Acceptance criteria.**
- Single source of truth for collection schema (runtime + init agree on PK, field set, `uuid` length).
- `upsert_embedding` replaces by uuid (true `collection.upsert()` on a uuid-PK schema, or delete-then-insert); re-syncing the same uuid does not create duplicate rows (asserted against live Milvus).
- `test_uuid_is_primary_key` passes against a **runtime-created** collection; `EXPECTED_COLLECTIONS`/`EXPECTED_FIELDS` reconciled to the real 6-collection schema.
- Foundry version bumped + tagged; downstream pins updated (coordinate with S1-07).

**Notes.** Foundry half of S1-01; do it first.

---

### S1-03 — infra: Redis missing from CI test stacks (pub/sub path never exercised)  🟠
`component:infrastructure` · `type:bug` · `priority:high` · **size: M** · refs logos #526; vault `logos-test-stack-render-redis-gap`

**Context.** Sophia CI starts Neo4j+Milvus but **not Redis**, so `test_pubsub_flow.py` self-skips and the #501 inter-service ontology-sync path has tests with **zero automated execution**. Vault note: the render template strips Redis and a re-render drops it everywhere — "THIS is the real bug."

**Objective.** Redis is a first-class service in the rendered test stack and per-repo CI; pub/sub tests run, not skip.

**Acceptance criteria.**
- Test-stack source template includes a Redis service; `copy_test_stacks.py` distributes it to downstream repos; re-render is idempotent and **keeps** Redis (regression-guarded).
- Sophia `ci.yml` waits on Redis health; `test_pubsub_flow.py` executes (not skipped) in CI and passes.
- A render test asserts Redis present in every generated `docker-compose.test.yml`.

---

### S1-04 — De-vacuum integration tests & actually run them in CI  🟠
`component:sophia` · `component:apollo` · `component:logos` · `type:refactor` · `priority:high` · **size: M** · **depends-on: S1-01, S1-03**

**Context.** The foundation broke silently *because the tests can't catch it*: apollo integration tests assert only `isinstance(data, list)` (pass on empty — `test_hcg_integration.py:39/48/86`); the sophia ingestion test queries `{id:}` (nodes stored with `uuid`) with **no assertion** (`test_hermes_ingestion_integration.py:139`); apollo CI excludes integration entirely (`ci.yml:105 -m "not integration"`); the logos M3 gate prints "GATE PASSED" while running a fully-skipped module (`m3-planning.yml:70-72`).

**Objective.** Integration tests fail when the data path is actually broken, and CI runs them.

**Acceptance criteria.**
- The three named vacuous tests assert real, non-empty, schema-correct outcomes (ingest seeded data → assert specific nodes/edges return by `uuid`).
- Apollo CI runs the integration tier against live infra (remove/replace the blanket exclusion).
- The logos M3 gate no longer reports success over skipped tests — it runs real assertions or fails loudly. *(Fully un-skipping the flexible-ontology planning tests is #464, blocked; minimum here: "the gate cannot green over a skipped module.")*

**Notes.** The trust layer — without it, S1-01/S1-02 can regress unnoticed.

---

### S1-05 — Finish & merge #505 Emergent Type Discovery to live-verified  🟢
`component:sophia` · `component:hermes` · `type:story` · `priority:high` · `domain:hcg` · **size: M** · **depends-on: S1-01** · refs #505

**Context.** #505 is real and unit-tested but lives on unmerged `feat/505-emergent-type-discovery` (sophia) and `feat/505-name-cluster` (hermes); it was live-validated only up to the embedding boundary and cannot run live until S1-01 lands.

**Objective.** Emergence runs end-to-end on live infra and merges to main.

**Acceptance criteria.**
- With embeddings persisting (S1-01), a seeded run of `type_emergence` mints ≥1 emergent type node + centroid and retypes its members (verified in Neo4j + Milvus).
- Hermes `/name-cluster` verified in the live path (not just unit mock).
- Both branches merged to main; untracked local `containers/` test-stack scaffolding committed or gitignored.
- STATUS regenerated to reflect #505 landed (feeds S1-08).

---

### S1-06 — Sophia: FeedbackConfig default Hermes port 18000 → 17000  🟡
`component:sophia` · `type:bug` · `priority:medium` · **size: S**

**Context.** `feedback/config.py:21` defaults `hermes_url` to `http://localhost:18000`; Hermes API is `17000`. Without an explicit `SOPHIA_FEEDBACK_HERMES_URL`, feedback delivery silently fails (retried 5×, dead-lettered).

**Objective.** Correct out-of-the-box default.

**Acceptance criteria.** Default resolves to the real Hermes port (prefer deriving from `logos_config` ports, not a literal); a unit test asserts the default; feedback delivers without an env override. *(Making feedback actually learn is Sprint 2; this just stops the silent misroute.)*

---

### S1-07 — Cross-repo foundry sync + workspace hygiene  🟡
`component:infrastructure` · `type:chore` · `priority:medium` · **size: S/M**

**Context.** sophia/hermes pin foundry `v0.7.1`; apollo/talos still `v0.5.0` (two minors behind) in `pyproject.toml` + `Dockerfile` — `check_foundry_alignment` is intra-repo so CI never flags the skew. Housekeeping: local `apollo/` main is one commit behind merged #168; the `apollo-wt-166` worktree is stale; Talos `Dockerfile` `CMD python -m talos` fails (no `__main__`) so the published image exits immediately.

**Objective.** All repos on a single foundry version; workspace clean.

**Acceptance criteria.**
- apollo + talos pinned to current foundry (post-S1-02 tag) in `pyproject.toml` + `Dockerfile`; `bump-downstream.sh` run; CI green.
- `git -C apollo pull` (merged #168 present locally); `apollo-wt-166` worktree removed.
- Talos `Dockerfile` gains a `__main__`/console entrypoint or the CMD is corrected; `docker run` no longer exits immediately.

---

### S1-08 — Doc refresh (return moment; keep it cheap)  🟢
`component:logos` · `type:refactor` · `priority:low` · **size: S**

**Context.** Docs trail code (mostly underselling): STATUS.md predates the #505 burst & headlines v0.7.0; vault `LOGOS.md` landing page lists **old ports (18000/28000…) and dead monorepo paths** (the first doc a returning-you reads); `COGNITIVE_LOOP.md` says ProposalBuilder has "zero tests" (now tested) and that the bus carries no inter-service traffic (it does, #501); VISION Goal 5 oversells "HCGPlanner" and cites a nonexistent design doc.

**Objective.** The map matches the territory at the load-bearing points.

**Acceptance criteria.** STATUS.md regenerated (via the PM agent) reflecting #505 + foundry v0.7.1; vault `LOGOS.md` ports + codebase paths corrected; named stale `COGNITIVE_LOOP.md` claims fixed; VISION Goal 5 wording reconciled with the actual planner (seeds the Sprint 5 planner decision).

---

## Appendix — keys

**Labels** (from `docs/PROJECT_TRACKING.md`): `component:{logos,sophia,hermes,talos,apollo,infrastructure}` · `priority:{critical,high,medium,low}` · `type:{bug,story,feature,refactor,chore,epic}` · `domain:hcg` · `status:{todo,in-progress,blocked,done}`.

**Size:** S ≈ a focused sitting · M ≈ a few sessions · L ≈ a sprint's main chunk · XL ≈ spans a sprint.

**Existing-issue mapping:** S1-04↔#464 (blocked, flexible-ontology planning tests) · S1-05↔#505 · S1-03↔#526 · downstream of Sprint 2: #503/#504/#506 (maintenance reasoning jobs); Sprint 6: #496/#461 (CWM→ontology consolidation + apollo reader reconciliation).

*Recorded 2026-05-29. Working-tree only until committed.*
