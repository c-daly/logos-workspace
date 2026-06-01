# S1 ("Stabilize Spine") — Resolution Plan

**Date:** 2026-06-01
**Author:** Claude Code (autonomous session)
**Status:** living plan — execution in progress

## Purpose

Plan to resolve the remaining open issues in the **S1 · Stabilize Spine** milestone,
and a record of what has already been done this session. This plan was written under an
explicit **no-merge regime**: every change lands as a feature branch + PR for human review;
nothing is merged by the agent.

## Operating constraints (this session)

- **Feature branches + PRs only. No merging — ever.** Every PR is left for human review.
- **Verify for real, read the output.** Beyond TDD: run the actual unit/integration/e2e
  suites and Playwright where the surface is UI; rely on **CI as the canonical build/test
  verification** (read its result, don't assume).
- **Adversarial method** — try to refute each fix/finding before claiming it.
- **Outside review via `codex`** — independent (non-Claude) review of diffs/plans.
- **Don't originate architecture** — for architectural issues, produce options + a
  recommendation and flag for the owner; do not unilaterally refactor.
- **Surface friction, don't work around it.**

### Mechanism note (why not big fan-out workflows)

A fan-out analysis workflow with per-agent `StructuredOutput` schemas returned only 1 of 8
structured results (~350k subagent tokens, mostly discarded). There is a known disconnect
between the Workflow tool's StructuredOutput mechanism and how agent-swarm subagents
communicate. **Decision:** do tool-using work (edits/tests/git/codex) in the main loop with
the routed `native__*` MCP tools (reliable here); use workflows only for parallel *text*
analysis, without `schema`.

---

## The six remaining issues — classification

| Issue | Title | Priority | Classification |
|-------|-------|----------|----------------|
| logos#530 | cross-repo foundry pin sync (talos+apollo → v0.7.2) | medium | **In progress** — PRs open |
| logos#535 | embedding-dim coherence (hermes persist_embedding) | critical | **Split**: regression test = safe; fix = needs live Milvus |
| apollo#171 | distinguish Hermes timeout from unreachable | high | **Safe to implement** |
| logos#91 | OpenAPI conformance tests (expand contract) | high | **Safe to implement** (additive) |
| sophia#148 | duplicate nodes — entity identity not resolved | high | **Blocked** — investigate the block first |
| logos#539 | decouple dev-time wiring from release-time pinning | medium | **Gated on owner decision** (architectural) |

---

## Per-issue plans

### logos#530 — cross-repo foundry pin sync  *(in progress)*

**State (grounded):** talos pinned `logos-foundry` at v0.5.0; apollo pinned
`logos-foundry`/`logos-sophia-sdk`/`logos-hermes-sdk` all at v0.5.0; sophia & hermes on
v0.7.2. The embedding-dim epic that gated this bump landed in **v0.7.2**. The drift-check
script + worktree gitignore were already merged (workspace PR #13). The prior bump PRs
(talos#60, apollo#170) were closed-not-merged; owner since clarified they "shouldn't have
been closed".

**Done this session:** bumped both to **v0.7.2** → **talos#65** (Closes talos#64) +
**apollo#179** (Closes apollo#178), on convention-conformant branches
`chore/<repo><ticket>-bump-foundry` (per GIT_PROJECT_STANDARDS.md). Verified locally
(`poetry lock` resolves, `poetry check` valid, lockfiles at v0.7.2, no v0.5.0 left) and in
CI (Python lint & tests pass on both; apollo Playwright E2E + Node + JS pass; branch-naming
+ issue-linkage + sync-status pass). Per-repo sub-tickets talos#64 / apollo#178 track the
bump under the foundry-sync epic logos#530.

**Resolved:** the initial branches violated the naming convention (`chore/530-…`, then
`chore/logos530-…`) — both mis-referenced a logos issue on talos/apollo branches and tripped
`sync-status` (issue-sync looked up a nonexistent same-repo #530 → 404). Fixed by following
GIT_PROJECT_STANDARDS.md strictly: per-repo sub-tickets (talos#64, apollo#178) and conformant
branches `chore/talos64-bump-foundry` / `chore/apollo178-bump-foundry` with `Closes #N`. All
checks pass. Lesson: the documented standard is authoritative for conventions — do not derive
them from the (drifted) existing branch history.

### apollo#171 — distinguish Hermes timeout from unreachable  *(safe to implement)*

**State (grounded):** `apollo/src/apollo/api/server.py:423` sets `httpx.Timeout(30.0)`;
`apollo/src/apollo/sdk/__init__.py:157` raises `"Cannot reach Hermes at {url} while {action}: {exc}"`,
conflating a read **timeout** with a **connection failure**.

**Approach:** in the SDK call path, catch `httpx.TimeoutException` (and subclasses)
distinctly from `httpx.ConnectError`/`ConnectTimeout`, and surface a different message
(e.g. "Hermes timed out after Ns" vs "Cannot reach Hermes"). Align/raise the 30s ceiling or
make it configurable. Keep the public error contract backward-compatible where possible.

**Tests:** (TDD) unit tests that monkeypatch the httpx client to raise `ReadTimeout` vs
`ConnectError` and assert the distinct messages/handling. (Playwright) drive the chat UI
against a stubbed slow/again-down Hermes and assert the user-visible error differs. (CI)
apollo Python lint & tests + Playwright E2E.

**Risk:** low-medium (bounded; the chat error path is well-isolated). No owner decision
required. Verify there is no closed PR / owner note deferring it before implementing.

### logos#91 — OpenAPI conformance tests  *(safe to implement)*

**State (grounded):** `hermes/tests/unit/test_openapi_conformance.py` exists and runs in
hermes CI; it has a **strict xfail** (`test_canonical_contract_covers_all_public_routes`)
recording that `contracts/hermes.openapi.yaml` documents only 5 of N public routes — an
explicit follow-up to logos#91. Acceptance also asks for request positive/negative cases +
JSON schema validation.

**Approach:** expand `logos/contracts/hermes.openapi.yaml` to document the remaining served
public routes (so the strict xfail flips to xpass, then remove the marker), and add
request-payload positive/negative + response JSON-schema-validation tests.

**Tests:** the existing conformance suite (flipping xfail→xpass is itself the signal) +
new request/response validation cases. (CI) hermes `pytest tests/` + logos
`validate-artifacts` OpenAPI validation.

**Risk:** medium — touches the canonical contract artifact that downstream SDKs depend on;
documenting already-served routes is low-risk, but regenerated SDKs should be sanity-checked.
No owner decision required.

### logos#535 — embedding-dim coherence (hermes side)  *(split)*

**State (grounded):** logos-side dim-resolution API landed in v0.7.2 (#541/#542 closed).
The issue's own acceptance is **hermes-side**: `hermes/src/hermes/milvus_client.py`
`persist_embedding` (insert at ~:284 wraps `[embedding]`) must insert with the correct
shape (no `num_rows` mismatch); a multi-entity doc must yield `entities == embeddings`; a
**live Milvus** regression test must assert row count grows by N (none exists today);
re-verify the cosmology repro live.

**Approach (split):**
1. **Safe now:** add the regression test scaffold (parametrize against an ephemeral/live
   Milvus; assert row count grows by N). It can land as xfail if the shape bug is still
   present, pinning the expected behavior.
2. **Needs care:** the shape fix touches ingestion-critical code and must be verified
   against **live Milvus** — only do this with infra up and full verification; otherwise
   leave the failing/xfail test + a precise root-cause note for review.

**Tests:** live-Milvus integration test (not mock), per the acceptance. (CI) hermes tests.

**Risk:** medium-high for the fix (ingestion path; needs live infra). Test-first is safe.

### sophia#148 — duplicate nodes / entity identity  *(blocked)*

**State (grounded):** `status:blocked`, priority:high. A new UUID is minted per entity
mention, so the same entity appears as duplicate nodes (observed live this session).

**Approach:** first **investigate what it is blocked on** (read the issue + comments and the
ingestion identity-resolution locus in sophia). Do **not** implement an identity-resolution
scheme autonomously — entity resolution is a design-loaded change and the issue is flagged
blocked. Produce a grounded root-cause + options note for the owner.

**Risk:** high to implement blind. Investigation/plan only this session.

### logos#539 — decouple dev-time wiring from release-time pinning  *(gated)*

**State (grounded):** architectural refactor to stop per-bug foundry tag bumps (dev = path/
workspace wiring; release = tag pins). #530 is the recurring manual chore that #539
eliminates. No merged PR addresses it.

**Approach:** produce 2–3 options (poetry path deps + a dev group; workspace/source
overrides; a sync script) with tradeoffs + a recommendation, as a design doc / draft PR
**flagged for owner review**. Do not unilaterally refactor cross-repo dependency wiring.

**Risk:** architectural — owner decision required.

---

## Execution order

1. **logos#530** — finish (confirm Python CI; resolve sync-status convention). *(in progress)*
2. **apollo#171** — implement (TDD + Playwright) → PR.
3. **logos#91** — expand contract + tests → PR.
4. **logos#535** — add live-Milvus regression test (+ shape fix only if infra up & verified) → PR.
5. **sophia#148** — investigate the block → root-cause/options note (no blind fix).
6. **logos#539** — options + recommendation → design doc / draft PR for review.

## Progress log

- ✅ Reviewed open PR **sophia#150** (edge-drop logging): clean, CI green; codex
  APPROVE-WITH-NITS (created-count logged pre-rollback; relation is LLM-supplied text). No
  defects requiring changes.
- ✅ Reviewed open PR **apollo#174** (faithful HCG views): works, CI green incl. Playwright;
  codex REQUEST-CHANGES — double `buildGraph` per render + filters leaking into the type
  list; un-namespaced reified IDs (collision risk); no `edge` color; thin tests. Findings
  recorded for follow-up.
- ✅ **logos#530** — talos#65 (Closes talos#64) + apollo#179 (Closes apollo#178) opened on
  convention-conformant branches; all CI green (Python tests + Playwright + branch-naming +
  sync-status). Earlier mis-named PRs (talos#61–63, apollo#175–177) closed + branches deleted.
  Awaiting human merge.

## Open decisions for the owner

1. **Scope** — confirm which of #171 / #91 / #535-fix / #148 / #539 to actually implement
   autonomously vs. plan-only.
2. **apollo#174** — apply codex's REQUEST-CHANGES fixes, or leave for manual review?
3. **#530 merge** — talos#65 + apollo#179 are green and ready; merge is yours to make.
