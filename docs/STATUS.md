# LOGOS Project Status

**Generated:** 2026-05-25
**Previous update:** 2026-04-05
**Foundry Version:** v0.7.0 (all repos at v0.7.0; no new release this period)

---

## What Changed Since April 5

After the new-job pause, work resumed in late May with a focused **environment + test-stack infrastructure push**, plus a **diagnostic session** that surfaced a previously-undocumented runtime gap (see "Key Finding" below). No new service features landed; the period is infrastructure, developer-experience, and reconstruction.

### Recent Work (merged PRs since April 5)
- **logos-workspace #8** (2026-05-24) — Idempotent env bootstrap for the LOGOS workspace (`bootstrap.sh`): toolchain (uv/Poetry/fnm), Python 3.12 pinning, per-repo extras, vendored webapp SDK build, config distribution.
- **logos #524** (2026-05-24) — `copy_test_stacks.py`: distribute rendered test stacks from `infra/<repo>/` into downstream repos' `containers/`.
- **sophia #143** (2026-05-24) — Add `otel` extra (`poetry install -E otel`).
- **apollo #165** (2026-05-25) — Fix `run_apollo.sh` cold-start: fail loud + build vendored SDKs before webapp install.
- **hermes #103** (~2026-04-05) — Restructure ML extras, harden publish-ml build (PyTorch CPU index).

### In Flight (open PRs)
- **logos #525** — `fix(infra): sync test-stack render template with hand-maintained outputs` (Redis service, offset host ports, `${NEO4J_PASSWORD}` bridge, neo4j 5.11.0 pin). **CI green, all review threads resolved — ready to merge.**
- **logos-workspace #9** — `feat(bootstrap): add --check-config switch` (render by default, drift-check on demand). CI green.

### Blocked
- **logos #464** — [Sophia] Update M3 planning tests for flexible ontology (`status:blocked`). Gates Goal 5 (planning) progress; blocked on flexible-ontology downstream updates.

---

## Key Finding: Apollo HCG/persona UI is empty (no ticket yet)

A live-debug this session found apollo's dashboard shows **no graph and no persona entries, with no errors**. Root cause is **not** empty data or a connection fault:

- Sophia starts (via `run_apollo.sh`), connects to Neo4j (`bolt://localhost:7687`), and **seeds successfully** on boot (pick-and-place, plan, persona).
- apollo connects to the same Neo4j but reads labels Sophia **no longer writes**: `hcg_client.py` queries `:Entity`/`:Process`/`:State`/`:Plan`/`:StateHistory`, and `persona_store.py` reads `:PersonaEntry`. Grep confirms Sophia writes **none** of these as labels post-#490; persona is persisted as `cwm_e` CWM-state nodes via `CWMPersistence`.
- Empty result sets aren't errors → blank UI, clean logs.

This is the **documented-but-unstarted work of logos #496** ("Consolidate CWM modules into HCG ontology types and retire `logos_cwm_e`"), whose problem list literally includes *"Apollo's `PersonaEntry` schema drift"* and whose acceptance criteria include *"Apollo's `PersonaEntry` model reconciled with ontology definition"* — all unchecked. It got stranded when the CWM-consolidation track was deferred behind the KG-maintenance track. **Recommend filing an apollo-reader-reconciliation issue under #496** (draft prepared this session).

---

## Progress Against Vision Goals

1. **Complete the cognitive loop** — *in progress.* Maintenance **scaffolding** done (Redis/pub-sub #500, ontology pub/sub #501, scheduler #508), but the four reasoning jobs (#503–507) are all `todo`. The scheduler runs with essentially no jobs yet.
2. **Grounding / physical knowledge** — *research active, integration deferred.* Unchanged this period (V-JEPA→CLIP PoC stands at txt_R@1 ≈ 0.371 vs 0.42 target).
3. **Flexible ontology** — *in progress, partially stalled.* Reified model (#490) merged is the foundation. The CWM→ontology consolidation (#496) is **not started** and is the direct cause of the Apollo finding above. Also open: type_definition UUID migration (#515), capability catalog (#465), downstream propagation (#460/#461).
4. **Memory and learning** — *not started.* Spec #415; depends on cognitive-loop maturity + testing sanity (#416).
5. **Planning and execution** — *in progress / blocked.* Planner stub still co-exists with HCGPlanner (#403); blocked on flexible-ontology downstream (#460, #464).
6. **Embodiment via Talos** — *paused* (deliberately deprioritized until cognition is solid).
7. **Infrastructure & observability** — *in progress; active this period.* This session: env bootstrap (#8), test-stack render/copy pipeline (#524, #525), run_apollo cold-start (#165), sophia otel extra (#143). Remaining: test data seeder centralization (#481), developer scripts (#409), endpoint spans (#335/#338), cross-service OTel testing (#321).
8. **Documentation & testing** — *in progress.* PM agent + planning docs rediscovered and STATUS refreshed (this doc). Remaining: onboarding guide (#135), OpenAPI contract tests (#91).
9. **Situated cognitive agent** — *deferred* (#521; prereqs Goals 1/4/5/6).

---

## Drift / Reconciliation

`scripts/reconcile-issues.sh` flags older open issues (mostly 2026-02/03) not reflected in the project board — e.g. logos #412 (event-driven reflection), #411 (hierarchical memory), #403 (deprecate planner_stub), #321/#335/#338/#339 (OTel gaps), #246/#264/#265/#267 (persona diary stack — legacy `logos_cwm_e`, subsumed by #496), and the maintenance stories #503–507. Worth a board/label pass at next vision review.

---

## Current Priorities (carried from VISION + this session)

1. **Apollo reader reconciliation** (slice of #496) — get the dashboard surfacing seeded data; unblocks day-to-day use and validates the cognitive loop end-to-end.
2. Cognitive-loop expansion — first KG-maintenance reasoning job (#503 entity resolution; design + plan ready).
3. Flexible-ontology consolidation (#496 in full) + downstream propagation (#460/#461/#515).
4. Merge the in-flight infra PRs (logos #525, logos-workspace #9); infra hardening (seeder #481, remaining standardization).
