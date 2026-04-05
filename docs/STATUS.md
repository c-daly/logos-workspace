# LOGOS Project Status

**Generated:** 2026-04-05
**Previous update:** 2026-03-18
**Foundry Version:** v0.7.0 (all repos at v0.7.0)

---

## What Changed Since March 18

**Short answer: almost nothing.** The project has been on pause since ~March 18 due to the user starting a new job. Three issues were filed (two on March 18, one on March 23), one issue was closed (hermes #101, same day as last status), and logos-workspace #5 was closed. No PRs opened, no PRs merged, no commits landed in any service repo. The workspace repo received two documentation commits (vision doc, gitignore updates).

### Issues Created Since March 18
- **logos #522** (2026-03-18): fix: robust ML container builds for downstream services (PyTorch CPU index) -- unlabeled
- **logos #523** (2026-03-23): Update LOGOS ticket templates to be compatible with experiment goal.yaml format -- component:logos
- **hermes #103** (2026-03-18): fix: restructure ML extras and harden publish-ml build -- unlabeled

### Issues Closed Since March 18
- **hermes #101** (closed 2026-03-18): [hermes] Add V-JEPA inference and VisualEmbeddingProvider -- closed via PR #102 (already tracked in previous status)
- **logos-workspace #5** (closed 2026-03-18): Autonomous experiment loop -- closed after engine moved to agent-swarm

### PRs Merged Since March 18
None.

### Open PRs
None across any repository.

### Commits Since March 18
- **logos-workspace** (2 commits): `d741ca8` docs and gitignore, `9b8b084` vision doc
- **logos, sophia, hermes, talos, apollo**: No commits since March 18.

---

## Open Issue Summary

| Repo | Open Issues | Change from Mar 18 |
|------|-------------|---------------------|
| logos | 44 | +2 (new: #522, #523; hermes issues tracked in logos) |
| sophia | 3 | -- |
| hermes | 2 | +1 (new: #103; #101 closed) |
| talos | 1 | -- |
| apollo | 0 | -- |
| logos-workspace | 0 | -- (was already 0; #5 closed Mar 18) |
| **Total** | **50** | **+2 net** |

---

## In Flight

Nothing. No open PRs, no branches with uncommitted work, no active implementation across any repo.

---

## Blocked

| Issue | Repo | Title | Blocked On |
|-------|------|-------|------------|
| #464 | logos | Update M3 planning tests for flexible ontology | #460, #461 (upstream ontology work) |
| #463 | logos | Validate M4 demo end-to-end with flexible ontology | #460, #461 (upstream ontology work) |

No change from previous status.

---

## Open Issues by Repo

### logos -- 44 open issues

**KG Maintenance epic (#499, status:in-progress, priority:high):**
- #501: Ontology Pub/Sub Distribution (priority:high, status:in-progress) -- infrastructure landed (logos #512, sophia #136, hermes #95), issue still open for remaining work
- #503: Entity Resolution -- Alias Detection and Merging (priority:high, status:todo) -- full design + 7-task TDD plan ready, not yet started
- #504: Type Correction -- Centroid-Based Reclassification (priority:medium, status:todo)
- #505: Ontology Evolution -- Emergent Type Discovery (priority:medium, status:todo) -- design doc exists
- #506: Relationship Inference -- Taxonomic Scaffolding and Missing Edges (priority:medium, status:todo)
- #507: Competing Edges & Confidence Model (priority:medium, status:todo)
- #508: Maintenance Scheduler -- When and How Jobs Run (priority:medium, status:in-progress) -- scheduler framework landed in sophia #137

**Learning & Memory epic (#415, priority:high):**
- #411: Hierarchical Memory Infrastructure (priority:high)
- #412: Event-driven Reflection System (priority:high)
- #413: Selective Diary Entry Creation (priority:medium)
- #414: Episodic Memory & Learning (priority:high)
- #416: Testing Sanity -- prerequisite to learning & memory work (priority:critical)

**Flexible Ontology cleanup (#458-465):**
- #458: Update ontology with lessons learned from TinyMind (priority:medium)
- #460: Update sophia planner for flexible ontology (priority:medium)
- #461: Update downstream repos for flexible ontology (priority:medium)
- #462: Update pick-and-place test data for flexible ontology (priority:medium)
- #463: Validate M4 demo end-to-end (priority:medium, status:blocked)
- #464: Update M3 planning tests (priority:medium, status:blocked)
- #465: Implement capability catalog in flexible ontology (priority:medium)

**OTel/Observability:**
- #321: Cross-service coverage gaps
- #335: [Sophia] Instrument API Endpoints with OTel Spans (priority:high)
- #338: [Hermes] Instrument API Endpoints with OTel Spans (priority:high)
- #339: [Hermes] Add OTel Testing & Documentation (priority:medium)

**Persona Diary features:**
- #246: CWM-E persona diary storage & API exposure (priority:medium)
- #264: Wire persona diary into Sophia (priority:medium)
- #265: CWM-E reflection worker (priority:medium)
- #267: Persona diary UI components (priority:medium)

**Infrastructure & Build (NEW since last status):**
- #522: Robust ML container builds for downstream services (PyTorch CPU index) -- NEW, unlabeled
- #523: Update LOGOS ticket templates for experiment goal.yaml format -- NEW, component:logos

**Infrastructure & Other (existing):**
- #521: Situated Cognitive Agent -- Persistent Operation, Communication Channels, Device Ecosystem (priority:low)
- #515: Migrate type_definition nodes from fabricated type_* IDs to real UUIDs
- #498: PM agent: detect undocumented new functionality during status updates
- #496: Consolidate CWM modules into HCG ontology types (priority:high)
- #481: Centralize test data seeder script
- #447: Documentation consolidation (priority:medium)
- #433: Standardize LOGOS repos
- #420: Standardize testing infrastructure
- #409: Standardize developer scripts
- #403: Deprecate planner_stub in favor of HCGPlanner
- #311: [Apollo] Implement Authentication and Authorization (priority:high, deferred per non-goals)
- #317: [Apollo] Advanced Graph Layouts and Visualization Options (priority:low)
- #135: Developer onboarding guide (priority:medium)
- #91: Add OpenAPI validation tests (priority:high)

### sophia -- 3 open issues
- #142: Fix /feedback delivery and implement minimal confidence update (priority:high)
- #101: Define session boundaries and ephemeral node lifecycle (priority:medium)
- #20: Extend HCG + Executor to support general tool actions (priority:medium)

### hermes -- 2 open issues
- #103: Restructure ML extras and harden publish-ml build -- NEW, unlabeled
- #100: Deploy interim JEPA translator as /embed_visual endpoint (priority:high)

### talos -- 1 open issue
- #31: Add coverage reporting and audit skip conditions

### apollo -- 0 open issues

### logos-workspace -- 0 open issues

---

## Progress Against Vision Goals

### 1. Complete the cognitive loop -- STALLED (was: infrastructure ready)

No implementation work since March 18. Entity resolution (#503) still has its full design + 7-task TDD plan ready to execute but nobody has picked it up. The KG maintenance epic (#499) and its sub-stories (#503-508) are all in the same state they were 18 days ago. Feedback processing (sophia #142) untouched. This was the declared top priority.

**Ready to execute:** Entity resolution (#503) -- design doc + implementation plan exist
**Designed but not planned:** Ontology evolution (#505)
**Still needs design:** Feedback processing, type correction (#504), relationship inference (#506)

### 2. Grounded perception via JEPA -- STALLED (was: building)

Hermes #101 (V-JEPA inference + VisualEmbeddingProvider) merged March 18 -- the last code change to land anywhere. Hermes #100 (deploy interim JEPA translator as `/translate_visual`) remains the next pick but has not been started. Two new build-related issues were filed (logos #522, hermes #103) for ML container/packaging problems discovered during the #101 work, suggesting there are unresolved build issues that would need attention before #100 can ship cleanly.

### 3. Flexible ontology -- STALLED (no change since Mar 2)

No movement. Downstream cleanup issues (#458-465) remain open. #464 and #463 still blocked on #460/#461. Type_definition UUID migration (#515) untouched. The reified model is implemented but downstream propagation has stalled for 34 days.

### 4. Memory and learning -- NOT STARTED

No change. Epic #415 and stories (#411-414) still waiting. Testing sanity (#416, priority:critical) remains the prerequisite.

### 5. Planning and execution -- PAUSED

No change. Planner stub deprecation (#403) still open. Still blocked on flexible ontology downstream updates (#460).

### 6. Embodiment via Talos -- PAUSED

No change. Correctly deprioritized.

### 7. Infrastructure and observability -- STABLE, new build issues surfaced

No new infrastructure work. Two new issues surfaced around ML container builds: logos #522 (PyTorch CPU index for downstream services) and hermes #103 (ML extras restructuring). These were discovered during the hermes PR #102 work and filed March 18 but not yet addressed. They may affect the ability to ship hermes #100 (JEPA translator deployment).

OTel gaps unchanged: endpoint-level spans for Sophia (#335) and Hermes (#338), cross-service testing (#321), Hermes OTel docs (#339).

### 8. Documentation and testing -- MINOR UPDATES

Two workspace commits landed (vision doc updates, gitignore). logos #523 filed to align ticket templates with experiment goal.yaml format. Otherwise no change. Testing gaps (#416, #420, #91) still open.

---

## Stale / Drift

**Stale issues (>30 days no activity as of April 5):**

Nearly everything is stale. With the project paused since March 18, all issues that were not updated in the last session are now 30+ days without activity. The most concerning:

| Issue | Repo | Last Activity | Days Stale |
|-------|------|---------------|------------|
| #20 | sophia | 2026-02-28 | 36 |
| #31 | talos | 2026-02-28 | 36 |
| #91 | logos | 2026-02-28 | 36 |
| #135 | logos | 2026-02-28 | 36 |
| #246-267 | logos | 2026-02-28 | 36 |
| #311-339 | logos | 2026-02-28 | 36 |
| #403-420 | logos | 2026-02-28 | 36 |
| #433-465 | logos | 2026-02-28 | 36 |
| #481 | logos | 2026-02-28 | 36 |
| #496-498 | logos | 2026-02-28 - 03-04 | 32-36 |
| #504-508 | logos | 2026-03-01 - 03-14 | 22-35 |
| #515 | logos | 2026-03-04 | 32 |
| #101 | sophia | 2026-03-01 | 35 |

This is expected given the pause. No action needed unless the user wants to triage/close some of the older issues.

**Potential closures (review with user):**
- logos #501 (Ontology Pub/Sub Distribution): Significant implementation landed (sophia #136, hermes #95). Still marked in-progress -- review remaining scope.
- logos #508 (Maintenance Scheduler): Core scheduler framework landed in sophia #137. Still marked in-progress -- review remaining scope.
- logos #498 (PM agent: detect undocumented new functionality): This meta-tooling request may have been superseded by the PM agent skill in agent-swarm.

**New issues needing labels:**
- logos #522: No labels. Should have component, priority, type labels.
- logos #515: No labels. Should have component, domain, type labels.
- hermes #103: No labels. Should have component, priority, type labels.

**Status label drift:**
- logos #501 and #508 are marked `status:in-progress` but have had no activity in 22-35 days. These should either be actively worked or moved to `status:todo`.

---

## Observations

**The project is paused, not abandoned.** Everything is exactly where it was left on March 18. No regressions, no drift in the codebase itself. The backlog is intact, designs are still valid, and the implementation plans have not rotted. This is a clean pause point.

**Resumption priorities are clear.** When work resumes, the previous priorities still hold:
1. **Entity resolution (#503)** -- full design + 7-task TDD plan ready to execute. This is the most shovel-ready work item.
2. **Hermes #100 (JEPA translator deployment)** -- but check logos #522 and hermes #103 first. ML container build issues may need resolution before this can ship.
3. **Flexible ontology downstream (#460, #461)** -- still blocking the planning track and two other issues.

**Three unlabeled issues need triage.** logos #522, logos #515, and hermes #103 were filed without full labels. A quick labeling pass would keep the backlog clean.

**The 18-day gap has pushed most issues past the 30-day stale threshold.** This is cosmetic -- the issues are not actually stale in the sense of being forgotten. But if the pause continues significantly longer, it would be worth reviewing whether any priorities have shifted.
