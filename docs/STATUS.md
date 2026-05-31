# LOGOS Project Status

**Generated:** 2026-05-30
**Previous update:** 2026-05-25
**Foundry Version:** v0.7.1 (sophia/hermes pinned; apollo/talos still v0.5.0 — see S1-07)

> **Source of truth.** This status is reconciled against the code on `main` plus
> unmerged frontier branches, not against prose docs. Evidence trail:
> `vault/10-projects/LOGOS/2026-05-29-ecosystem-audit.md` (code-grounded audit,
> verified `file:line`) and `docs/plans/2026-05-29-roadmap-to-mts.md` (the MTS
> roadmap + Sprint-1 tickets). The 2026-05-30 issue realignment
> (`docs/plans/2026-05-30-issue-realignment.md`) carved the work into six MTS
> sprints on GitHub.

---

## The one-paragraph picture

**The foundation is strong; the cognitive core is still largely spec.** What is
built and solid: the service topology (5 repos, shared `logos_*` packages,
ports, config), the reified HCG ontology model, the ingest pipeline
(Hermes NER/embeddings → Sophia proposal processing → Neo4j + Milvus), the Redis
event bus with live inter-service ontology sync, the maintenance scheduler, and
the CI/test-stack/env tooling. What is **not** yet built: the autonomous
reasoning that turns that pipeline into a mind — feedback that mutates the graph,
the maintenance reasoning jobs, K-lines/curiosity/CWM-E gain, memory tiers, and
the persistent event-driven loop. Crucially, the audit found the *built* spine is
**silently broken at one load-bearing seam**: embeddings never durably persist,
which starves the type classifier and emergence. That bug is the keystone of the
near-term plan.

**Where the milestone line sits.** Against the full vision (grounding,
embodiment, the 13-paper program) this is an open-ended research program, not a
finite backlog. Against the concrete near-term milestone — **Minimum Thinking
Sophia (MTS)**, defined below — the project is roughly **35–40%**, and MTS is six
sprints out (see `docs/plans/2026-05-29-roadmap-to-mts.md`).

---

## Keystone finding: embeddings silently fail to persist

The audit's central finding, re-verified in current code:

- Live flow has entity nodes in Neo4j but the `hcg_*_embeddings` Milvus
  collections read **0**. Vector persistence is a no-op.
- **Sophia side** (`sophia/src/sophia/ingestion/proposal_processor.py:519-524`):
  the `batch_upsert_embeddings` call sits inside a warn-only `try/except` that
  swallows failure — the pipeline keeps going and logs a `warning`.
- **Logos side** (`logos/logos_hcg/sync.py:155`): the runtime Milvus schema uses
  an `auto_id` INT64 primary key with `uuid` as a plain field, while
  `upsert_embedding`/`batch_upsert_embeddings` call `collection.insert()`
  (`sync.py:315,365`) — so a re-synced `uuid` **appends a duplicate** instead of
  replacing, despite comments claiming upsert semantics. The init script
  (`infra/init_milvus_collections.py`) ships a *contradictory* `uuid`-PK schema,
  and the guard test encodes the init schema, so the divergence passes in CI.

**Consequence:** with no retrievable vectors, the type classifier degrades to the
fallback `'entity'` type for everything, and #505 emergent-type discovery loads 0
cluster members → "no qualifying clusters." This single seam neuters the whole
maintenance/emergence arm.

**Tracked as the Sprint-1 keystone:** **sophia#146** (warn-only swallow at
ingestion) depends on **logos#528** (upsert-by-uuid + schema reconciliation).
Fix logos#528 first, then sophia#146.

---

## What's built and solid

| Area | State | Evidence |
|------|-------|----------|
| Service topology + shared packages | Solid | `logos_config`, `logos_hcg`, `logos_events`, ports, 5 repos wired |
| Reified HCG ontology model | Solid | logos #490 merged on main (flexible-ontology migration) |
| Ingest pipeline (Hermes → Sophia → Neo4j/Milvus) | Built, runs | `ProposalBuilder` (now tested), `ProposalProcessor`; **but** embeddings don't persist (keystone) |
| Redis event bus + inter-service ontology sync | Built, on main | `logos_events.EventBus`; sophia #501 pub/sub publisher merged |
| Maintenance scheduler | Built, on main | sophia #508 merged — runs, but the reasoning jobs it would dispatch (#503/#504/#506) are unbuilt |
| Type classification via centroids | Built | sophia #129 / logos #494 — degrades to fallback while embeddings are empty (keystone) |
| CI / test-stack render-and-copy / env bootstrap | Active, recent | logos #524, logos-workspace #8/#9, apollo #165, sophia #143 |

## What's spec / in-flight / not started

| Area | State | Tracking |
|------|-------|----------|
| Emergent type discovery (#505) | Built + unit-tested, **unmerged**, can't run live until keystone lands | `feat/505-emergent-type-discovery` (sophia), `feat/505-name-cluster` (hermes) — S1-05 |
| Feedback that mutates the graph | Not started — `/feedback` is a stub that logs and returns | Sprint 2 (#499 epic, #503/#504/#506/#507) |
| Maintenance reasoning jobs | Not started — scheduler has ~no jobs | Sprint 2 |
| Orchestrator + persistent event-driven loop + daemons | Not started — Sophia is request/response | Sprint 3 (under-decomposed by design) |
| K-lines / curiosity / CWM-E gain | Not started — CWM-E state models exist, mechanisms don't | Sprint 4 |
| Memory tiers + planning-in-loop | Not started | Sprint 5 (#415 epic, #460/#464 et al.) |
| CWM→ontology consolidation (retire `logos_cwm_e`) | Not started — root cause of the empty Apollo dashboard | logos #496 (Sprint 6 reconcile) |
| Grounding / JEPA (CWM-G) | Research active, integration deferred (Horizon 2) | V-JEPA→CLIP PoC, txt_R@1 ≈ 0.371 vs 0.42 target |
| Embodiment via Talos | Paused (Horizon 2) | deliberate |

---

## Minimum Thinking Sophia (MTS) — the near-term milestone

MTS is the point where LOGOS stops being *infrastructure + a thin ingestion loop*
and becomes *a small mind that runs*: given a stream of experience (text for now),
Sophia **autonomously grows and maintains** a non-linguistic graph,
**reasons/plans** over it, **learns from feedback**, runs as a **persistent
event-driven loop**, and **surfaces its cognitive state in Apollo**. Everything
past MTS — JEPA grounding, Talos embodiment, the situated multi-device agent, the
efficiency program, the papers — is **Horizon 2**, deliberately out of scope here.

### The six MTS sprints

| # | Goal | Exit (abbrev.) | ~% after |
|---|------|----------------|----------|
| **1** | **Make the built spine actually run & be trusted** | ingest→embed→classify→emerge works on live infra; CI would catch a regression; #505 merged & live-verified | ~45% |
| **2** | Close the feedback loop (Learning v0) | feedback mutates the graph; maintenance jobs run | ~55% |
| **3** | Orchestrator + event-driven loop + daemons | Sophia runs as a persistent process reacting to bus events | ~65% |
| **4** | K-lines + curiosity + CWM-E gain (Cognition v1) | stimulus activates a constellation; non-activation emits curiosity; gain modulates processing | ~75% |
| **5** | Memory tiers + planning-in-loop | knowledge promotes ephemeral→STM→LTM; new knowledge triggers HCGPlanner | ~85% |
| **6** | Demonstrate the thesis (MTS integration) | recorded end-to-end session, Apollo rendering goal→plan→diary→cognitive-state | **~95% = MTS** |

Weight concentrates in Sprints 3–4 (the genuinely novel cognitive mechanisms).
Sprints S3/S4 are intentionally under-decomposed on the board — decompose when
they approach. Full detail and per-ticket ACs in
`docs/plans/2026-05-29-roadmap-to-mts.md`.

---

## Sprint 1 — "Make the built spine actually run & be trusted"

The active sprint. Goal: every component the audit found *built-but-secretly-broken
or untested* is fixed and provably working end-to-end, and #505 lands
live-verified. Critical path: **logos#528 → sophia#146 → {S1-04, S1-05}**.

| Ticket | What | Issue | State |
|--------|------|-------|-------|
| S1-01 🔴 keystone | Embedding persistence silently fails at ingestion | **sophia#146** | open |
| S1-02 🔴 | HCGMilvusSync can't upsert-by-uuid (auto_id PK) + schema divergence | **logos#528** | open |
| S1-03 🟠 | Redis missing from CI test stacks (pub/sub never exercised) | folded into logos #526 | open |
| S1-04 🟠 | De-vacuum integration tests + run them in CI | **logos#529** | open (5 prune PRs linked) |
| S1-05 🟢 | Finish & merge #505 emergent type discovery, live-verified | logos #505 | in-progress |
| S1-06 🟡 | FeedbackConfig default Hermes port 18000→17000 | folded into sophia #142 | open |
| S1-07 🟡 | Cross-repo foundry sync (apollo/talos still v0.5.0) + workspace hygiene | **logos#530** | open |
| S1-08 🟢 | Doc refresh (this document, VISION, COGNITIVE_LOOP, vault landing) | **logos#531** | in-progress |

**Sprint exit (DoD):** seed the demo scenario → run a `/llm` turn mentioning new
entities → confirm (a) embeddings land in Milvus, (b) the classifier assigns
non-fallback types, (c) `type_emergence` mints ≥1 real emergent type, (d) Apollo's
dashboard shows the result — and a CI run that **fails** if any of those breaks.

---

## Known live-data gotcha: empty Apollo dashboard

Apollo's dashboard shows **no graph and no persona entries, with no errors**. Root
cause is schema drift, not missing data: Apollo's `hcg_client.py` queries
`:Entity`/`:Process`/`:State`/`:Plan` labels and `persona_store.py` reads
`:PersonaEntry`, but Sophia no longer writes those as labels post-#490 (persona is
persisted as `cwm_e` CWM-state nodes). Empty result sets aren't errors → blank UI,
clean logs. This is the unstarted work of **logos #496** (consolidate CWM modules
into HCG ontology types; reconcile Apollo's reader), slated for **Sprint 6**.

---

## Deferred manual step (not in this PR)

The vault LOGOS landing/narrative still lists **old ports (18000/28000…) and dead
monorepo paths** and should be re-pointed at the audit
(`2026-05-29-ecosystem-audit.md`) and the MTS roadmap. The vault is edited through
a separate tool, not from this repo, so per S1-08 that update is left as a manual
follow-up for the maintainer. This PR refreshes only the in-repo workspace docs
(`docs/STATUS.md`, `docs/VISION.md`, `docs/COGNITIVE_LOOP.md`).
