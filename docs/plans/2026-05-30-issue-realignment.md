# Issue Realignment Plan — to the MTS goal

**Created:** 2026-05-30
**Status:** ✅ **EXECUTED 2026-05-30** (§A–§F applied; board grouping deferred — see Execution Record at bottom). Pre-execution adversarial verification flipped 4 dispositions; details below.
**Goal anchor:** `docs/plans/2026-05-29-roadmap-to-mts.md` (MTS, 6 sprints) · evidence: `vault/10-projects/LOGOS/2026-05-29-ecosystem-audit.md`

---

## ⏯ RESUME STATE / NEXT ACTION (read this first)

We assessed GitHub issue structure vs the refined MTS goal, produced this realignment plan, and were about to execute it. Two paths to resume:

- **Full pass (incl. GitHub Project board):** the token lacks the Projects-v2 scope (currently `repo, read:org, gist, admin:*`). User must first run **`gh auth refresh -s project`** (read+write; the read-only `read:project` won't let me set board fields). Then execute §A milestones + §B–§F + board grouping in one pass.
- **Non-board pass (no token change needed):** execute the 7 milestones + §B–§F now — all only need `repo` scope, which is present. Board grouping deferred.

User was deciding between these two when we paused for a reboot. On resume, ask which path (or proceed per their answer).

**Related open state:** the 5 test-prune PRs are **open, not merged**, and currently link **no issue** (logos #527, sophia #145, hermes #104, talos #59, apollo #169). §B's new test issue (S1-04) is meant to track them — the `Check issue linkage` CI gate may flag them until then.

---

## A. Structure layer

**Milestones** — create in `logos` (46/53 issues live there); retire Phase-1:
- **Create:** `S1 · Stabilize Spine` · `S2 · Feedback Loop` · `S3 · Orchestrator & Event Loop` · `S4 · K-lines & CWM-E Gain` · `S5 · Memory & Planning` · `S6 · Demonstrate (MTS)` · `Horizon 2 (deferred)`
- **Retire/close:** Phase-1 `M1 HCG` · `M2 SHACL` · `M3 Planning` · `M4 Pick-and-Place`. Re-milestone their stragglers (#91 off M2, #135 off M3).
- Milestones are per-repo → for the ~6 non-logos issues (sophia #142/#101/#20, hermes #100, talos #31), create matching milestones only where needed, OR group on the board.

**Board (Projects v2)** — the proper cross-repo home for sprint grouping per PROJECT_TRACKING.md. **Needs `gh auth refresh -s project`.** Once available: inspect the existing LOGOS board (does it exist? fields? items?), then add a Sprint/Iteration field and assign issues per §F.

## B. CREATE — Sprint-1 gap (the current goal's immediate work is unticketed)

| New | Title | Repo | Labels | Milestone | Note |
|-----|-------|------|--------|-----------|------|
| **S1-01** | fix: embedding persistence silently fails at ingestion | **sophia** | component:sophia, type:bug, priority:critical, domain:hcg | S1 | 🔑 KEYSTONE — blocks live #505 |
| **S1-02** | fix: HCGMilvusSync can't upsert-by-uuid (auto_id PK) + schema divergence | logos | component:logos, type:bug, priority:critical, domain:hcg | S1 | blocks S1-01 |
| **S1-04** | test: de-vacuum integration tests + run them in CI | logos | type:testing, priority:high, component:infrastructure | S1 | **link the 5 prune PRs as `Part of`** |
| **S1-07** | chore: cross-repo foundry version sync + workspace hygiene | logos | type:chore, component:infrastructure | S1 | apollo/talos still pin foundry v0.5.0 |
| **S1-08** | docs: refresh STATUS / vault landing / COGNITIVE_LOOP / VISION | logos | type:documentation, component:logos | S1 | or fold into #447 |

**Fold into existing (don't create):** S1-03 (Redis-in-CI) → extend **#526**; S1-05 (finish #505) → **is #505** (re-milestone S1, fix status/priority per §E); S1-06 (FeedbackConfig port) → inside sophia **#142**.
**Also missing:** apollo has 0 issues despite Sprint-6 reconciliation work (#461 apollo slice) + the webapp-UI 0%-coverage gap — create under S6 later.

## C. CLOSE

**Done-but-open** (verify merged PR first): logos **#508** (scheduler on main), logos **#501** (pub/sub on main), workspace **#11** (bootstrap shipped), hermes **#103** (ML build merged ~04-05).
**Legacy, subsumed by #496** (close as superseded, comment → #496): logos **#246**, **#264**, **#265**, **#267** (cwm_e persona-diary stack).

## D. REFRAME / RE-MILESTONE

- **#464** → retitle `Rebuild HCGPlanner tests for flexible ontology (replaces removed M3 gate)`; → S5; unblock (the M3 workflow was removed in logos PR #527).
- **#463** → S6 (validate end-to-end demo).
- **#135** → drop M3 milestone, keep docs backlog.
- **#91** → drop M2 milestone; → S1/test (hermes OpenAPI documents only 5/13 endpoints — real).

## E. RELABEL / REPRIORITIZE

- **#505**: status:todo→in-progress; priority:medium→high (active frontier on feat/505 branches).
- **#311** (apollo auth): priority:high→low; → Horizon 2 (VISION lists auth as a non-goal).
- **#416**: reconcile priority:critical (act in S1 or downgrade — untouched since Feb).
- **#498**: add component:logos, type:automation (unlabeled).
- **talos #31**: add component:talos, type:testing; consider close (mostly satisfied by the prune work).
- **hermes #103**: type:story→type:chore (then close per §C).

## F. ASSIGN existing keepers to sprints

| Sprint | Issues |
|--------|--------|
| **S2** | #499 (epic) · #503 · #504 · #506 · #507 · sophia #142 |
| **S4** | #412 |
| **S5** | #415 (epic) · #411 · #414 · #413 · #403 · #460 · sophia #101 · sophia #20 · #465 · #462 |
| **S6** | #496 · #461 · #463 |
| **Horizon 2** | #521 · hermes #100 · #317 · #311 · #458 · #515 |
| **Infra/observability backlog** (not MTS-blocking) | #526(+S1-03) · #522 · #523 · #481 · #433 · #420 · #409 · #447 · #321 · #335 · #338 · #339 |

> Surfaces an imbalance: S2 is well-formed (the #499 epic); **S3/S4 have ~no issues** — the deep cognitive machinery (orchestrator/daemons/K-lines/curiosity/gain) isn't decomposed. That decomposition is a planning task for when those sprints approach — don't over-ticket now.

## G. Execution sequence

1. *(if board)* `gh auth refresh -s project`; inspect the LOGOS board.
2. Create 7 milestones (§A); retire Phase-1.
3. Create Sprint-1 issues (§B) — **keystone S1-01 first**; link the 5 prune PRs to S1-04.
4. Verify merged PRs, then close §C.
5. Apply §D/§E retitles + label edits.
6. Assign milestones per §F (+ board field if available).
7. Run `scripts/reconcile-issues.sh` to confirm no residual drift.

**Executable now with `repo` scope:** all of §B–§F + milestones. **Needs token:** the board grouping only.

---
*Saved 2026-05-30 ahead of a reboot. Working-tree only — `git`/`vault sync` to persist.*

---

## Execution Record — 2026-05-30

Before mutating, a 14-agent adversarial verification pass checked every destructive/judgment disposition against ground truth (code on main, VISION, the M3-removal PR, #496 scope). It **flipped 4 calls** that would have discarded un-built work — applied the corrected versions.

### Verification-driven deviations from the plan as written
- **hermes #103 — NOT closed.** Work never landed on main (Dockerfile still hardcodes `torch==2.3.1`/`--extras ml-gpu`; no `ml-core` extra). Kept open; relabeled `type:story → type:refactor` (no canonical `type:chore` exists — mapped to `refactor`); tracking comment added.
- **logos #264 — NOT closed.** #496 subsumes only the CRUD plumbing; the headline *auto-create PersonaEntry on `/plan`* behavior is absent from code and from #496's ACs. Kept open, `status:blocked` (blocked-by #496), requirement carried forward in a comment.
- **logos #265 — NOT closed.** Reflection worker is genuinely un-built. Kept open; retitled *"Rebuild CWM-E reflection worker on HCG ontology (post-#496)"*; `status:blocked` (blocked-by #496).
- **logos #267 — NOT closed.** Apollo diary UI delivered by neither #496 nor anything else. Kept open; re-homed to **S6**; `status:blocked` (blocked-by #496). (Resolves the §C-line-47 vs §B-line-42 contradiction the verifier flagged — #267 is *not* subsumed by #496.)
- **`type:chore` → `type:refactor`** everywhere (S1-07, #103): `chore` is not in the canonical `type:*` taxonomy (PROJECT_TRACKING.md); the authority wins, no new label created.
- **#508 closed *with caveats*:** core scheduler is done/on-main/wired, but observability ACs were unmet → split to new issue **#532** before closing; threshold + relationship_discovery deferrals named (#506).
- **#463 keeps `status:blocked`** (blocked on S5 deps #460/#462, *not* S1) — corrected the task's wrong "blocked on S1" framing.
- **#416 stays open** as the testing epic, downgraded `critical → high` (not folded/closed).
- **talos #31 stays open** (PR #59 unmerged; coverage-reporting/skip-audit ACs untouched) — labels added only.

### §A Milestones created
- **logos #6–#12:** S1 Stabilize Spine · S2 Feedback Loop · S3 Orchestrator & Event Loop · S4 K-lines & CWM-E Gain · S5 Memory & Planning · S6 Demonstrate (MTS) · Horizon 2 (deferred)
- **sophia #1–#3:** S1, S2, S5 · **hermes #1:** Horizon 2
- **Retired (closed):** logos M1–M4 (all 0 open after moving #91→S1, clearing #135).

### §B Issues created
| New | Repo#Num | Milestone |
|-----|----------|-----------|
| S1-01 (🔑 keystone) | **sophia#146** | S1 |
| S1-02 (HCG upsert) | **logos#528** | S1 |
| S1-04 (de-vacuum tests) | **logos#529** | S1 |
| S1-07 (foundry sync) | **logos#530** | S1 |
| S1-08 (docs refresh) | **logos#531** | S1 |
| #508 observability follow-up | **logos#532** | — (infra backlog) |

5 prune PRs (logos#527, sophia#145, hermes#104, talos#59, apollo#169) linked to **#529** ("Part of"). CI "Check issue linkage" is green on all (warn-only gate).

### §C–§F applied
- **Closed:** logos #501, #508, #246; workspace #11.
- **Reframed/relabeled:** #464 (retitle+unblock→S5), #463 (→S6, keep blocked), #91 (→S1), #135 (drop M3), #505 (→S1, in-progress+high), #311 (→Horizon 2, low), #416 (crit→high), #498 (labeled), talos#31 (labeled).
- **Sprint assignment (§F):** S2 ×6, S4 ×1, S5 ×10, S6 ×4, Horizon 2 ×5 — all succeeded.

### Sprint distribution (open issues, logos)
S1=6 · S2=5 · S3=0 · S4=1 · S5=9 · S6=4 · Horizon 2=5 · (none / infra backlog)=18.
**S3 empty, S4=1** is the known, intentional under-decomposition — the deep cognitive sprints (orchestrator/daemons/K-lines/gain) aren't ticketed yet by design; decompose when they approach.

### ⏳ Still pending (path a / board)
GitHub Projects-v2 board grouping was **not** done — token still lacks `project` scope (`repo, read:org, gist, admin:*`). To finish: `gh auth refresh -s project`, then add a Sprint/Iteration field on the LOGOS board and group by the milestones above. Sprint grouping is fully usable via milestones in the meantime.
