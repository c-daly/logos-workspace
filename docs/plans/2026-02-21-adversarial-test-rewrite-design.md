# Adversarial Test Suite Redesign

**Date:** 2026-02-21
**Status:** Draft
**Scope:** All LOGOS repositories (apollo, sophia, hermes, talos, logos)

## Problem

The LOGOS codebase has gone through significant architectural shifts (flat ontology, proposal processing, reified edges, OTel instrumentation, WebSocket batching, Three.js memoization, async background workers). Test suites have not kept pace. Coverage is uneven, test value is unclear in places, and the overall test architecture may not reflect current risk areas.

## Goals

1. Redesign each repo's test suite from a clean slate
2. Let agents autonomously decide what to test, how to organize tests, and where to invest effort
3. Use adversarial dynamics to ensure test quality, strategy soundness, and aggregate utility
4. Produce test suites that catch real production bugs, not just achieve line coverage

## Non-Goals

- Prescriptive per-module test assignments
- Maintaining existing test organization for its own sake
- 100% coverage (we want smart coverage, not total coverage)

---

## Architecture

### Overview

A ralph loop acts as orchestrator, spawning three subagent processes per iteration. Each agent has a distinct mandate, principles, and artifacts it produces. The loop runs per-repo until exit criteria are met.

```
┌──────────────────────────────────────────────────────────────┐
│  Ralph Loop (Orchestrator)                                    │
│                                                                │
│  Per iteration:                                                │
│                                                                │
│  ┌─────────────┐  ┌──────────────┐                            │
│  │   WRITER    │  │  ADVERSARY   │   (parallel)               │
│  │             │  │              │                             │
│  │ Reads:      │  │ Reads:       │                             │
│  │ - source    │  │ - source     │                             │
│  │ - JUDGMENT  │  │ - tests      │                             │
│  │ - prev plan │  │ - JUDGMENT   │                             │
│  │             │  │ - mutations  │                             │
│  │ Produces:   │  │              │                             │
│  │ - tests     │  │ Produces:    │                             │
│  │ - TEST_PLAN │  │ - CRITIQUE   │                             │
│  │ - RESULTS   │  │ - RISK_MAP   │                             │
│  │ - REBUTTAL  │  │ - EVALUATION │                             │
│  └──────┬──────┘  └──────┬───────┘                            │
│         │                │                                     │
│         └───────┬────────┘                                     │
│                 ▼                                              │
│         ┌──────────────┐                                       │
│         │    JUDGE     │   (after both complete)               │
│         │              │                                       │
│         │ Reads:       │                                       │
│         │ - source     │                                       │
│         │ - tests      │                                       │
│         │ - CRITIQUE   │                                       │
│         │ - RESULTS    │                                       │
│         │ - REBUTTAL   │                                       │
│         │ - EVALUATION │                                       │
│         │ - RISK_MAP   │                                       │
│         │              │                                       │
│         │ Produces:    │                                       │
│         │ - JUDGMENT   │                                       │
│         └──────────────┘                                       │
│                                                                │
│  Exit criteria (ALL must hold):                                │
│  - Judge verdict = PASS                                        │
│  - Coverage >= threshold                                       │
│  - Mutation score >= threshold                                 │
│  - Confidence scores converge (gap < 10%)                      │
│  - No unaddressed consensus overrides                          │
└──────────────────────────────────────────────────────────────┘
```

### Iteration Flow

1. **Writer and Adversary run in parallel** as separate subagent processes (Task tool, general-purpose type). They share no context — only file artifacts.
2. **Judge runs after both complete.** It reads all artifacts and produces the authoritative ruling.
3. **Consensus overrides:** If Writer's REBUTTAL and Adversary's EVALUATION both flag the same Judge ruling as incorrect, it's marked as a consensus override. The Judge must address all consensus overrides in the next iteration.
4. **The orchestrator** (ralph loop session) reads JUDGMENT.md, checks exit criteria, and either emits `<promise>TESTS COMPLETE</promise>` or lets the loop continue.

### Bias Cancellation Mechanism

The Writer has a pro-test bias (defends its work). The Adversary has an anti-test bias (finds fault). When both agree the Judge got something wrong, their biases have canceled out — it's a strong signal the Judge's ruling was incorrect.

| Writer says | Adversary says | Result |
|-------------|---------------|--------|
| Judge was right | Judge was wrong | Normal disagreement. Ruling stands. |
| Judge was wrong | Judge was right | Normal disagreement. Ruling stands. |
| Judge was wrong | Judge was wrong | **Consensus override.** Judge must reconsider. |

---

## Agent Specifications

### Writer Agent

**Mandate:** Design and implement the optimal test suite for the repository. You decide what to test, how to organize tests, and where to invest effort.

**Inputs:**
- Repository source code (full access)
- JUDGMENT.md (from previous iteration, if exists)
- TEST_PLAN.md (own evolving plan from previous iterations)
- Consensus override notes (if any)

**Outputs:**
- `TEST_PLAN.md` — Living document. Explains what is being tested and why. Must be revised when understanding changes. Includes confidence score.
- Test files — Clean-slate rewrites. Organization is writer's choice.
- `RESULTS.md` — Coverage %, mutation score, what changed this iteration, response to previous judgment.
- `REBUTTAL.md` (optional) — Evaluation of previous Judge rulings the writer disagrees with. Must include reasoning.

**Principles:**
1. **Honest self-assessment.** If something is genuinely hard to test, say so and explain why, rather than writing a fake test that gives false confidence.
2. **Risk-based prioritization.** Test high-risk, complex code more thoroughly. Simple code gets simple tests. Justify the allocation in TEST_PLAN.md.
3. **Good-faith response to critique.** When the adversary finds a real gap, acknowledge it and fix it. Don't dismiss valid findings defensively.
4. **Explain trade-offs.** If you chose NOT to test something, document why. "Not worth testing" is valid if the reasoning is sound.
5. **Evolve the plan.** Your TEST_PLAN.md WILL be wrong in early iterations. Revise aggressively based on what you discover about the codebase and what the adversary finds.

**Workflow per iteration:**
1. Read source code (explore as needed — you decide how to navigate the repo)
2. Read JUDGMENT.md and any consensus overrides
3. Revise TEST_PLAN.md based on new understanding
4. Write/rewrite test files
5. Run test suite + coverage
6. Run mutation testing, summarize surviving mutants
7. Write RESULTS.md with confidence score
8. Optionally write REBUTTAL.md if disagreeing with a Judge ruling
9. Commit all artifacts

---

### Adversary Agent

**Mandate:** Find every meaningful weakness in the test suite. Your job is to identify gaps that would let real bugs ship. You are rigorous but fair.

**Inputs:**
- Repository source code (full access)
- Test files (writer's output)
- Mutation testing results (MUTMUT_SUMMARY.md)
- JUDGMENT.md (from previous iteration)
- TEST_PLAN.md (writer's strategy)

**Outputs:**
- `CRITIQUE.md` — Findings organized by severity. Includes confidence score and verdict (PASS/FAIL).
- `RISK_MAP.md` — Assessment of where real risk lives in the codebase vs where testing effort is allocated.
- `EVALUATION.md` (optional) — Assessment of previous Judge rulings the adversary disagrees with.

**Principles:**
1. **Fair play.** Acknowledge good work. If the writer nailed a tricky edge case, say so. Credibility comes from balance, not relentless negativity.
2. **Substantive findings only.** Every critique must explain: what could go wrong in production if this gap isn't addressed? If you can't articulate the real-world risk, it's not a finding.
3. **No goalpost moving.** If a previous critique was addressed adequately, mark it resolved. Don't invent new objections to the same area.
4. **Distinguish severity.** "This will miss a production bug" vs "this test name could be clearer" are not the same. Label accordingly: Critical, Important, Nitpick.
5. **Intellectual honesty.** If you can't find meaningful gaps, say PASS. Your job is to find real problems, not to justify your existence.

**CRITIQUE.md structure:**
```markdown
## Verdict: PASS | FAIL
## Confidence: N%

## Resolved from Previous Round
- [Finding that was adequately addressed]

## Critical Gaps (must fix)
- [Finding + real-world risk + suggested test]

## Important Gaps (should fix)
- [Finding + risk assessment]

## Surviving Mutants Analysis
- [Mutant + why it survived + whether it matters]

## Nitpicks (optional)
- [Lower priority observations]
```

**RISK_MAP.md structure:**
```markdown
## High Risk / Under-tested
- [Module/area + why it's high risk + current test gap]

## High Risk / Well-tested
- [Module/area + what's covered + remaining concerns]

## Low Risk / Over-tested
- [Module/area + suggestion to reduce]

## Recently Changed / Stale Tests
- [Areas where architecture shifted but tests didn't follow]

## Cross-module Boundaries
- [Integration points that no unit test covers]
```

---

### Judge Agent

**Mandate:** Evaluate the adversary's critique for accuracy and fairness. Make the authoritative ruling on what the writer should address. Calibrate confidence scores.

**Inputs:**
- Repository source code (full access)
- Test files
- CRITIQUE.md
- RESULTS.md
- REBUTTAL.md (if writer filed one)
- EVALUATION.md (if adversary filed one)
- RISK_MAP.md
- Previous JUDGMENT.md (own precedent)

**Outputs:**
- `JUDGMENT.md` — Validated findings, dismissed findings with reasoning, calibrated confidence score, final verdict, rebuttal rulings, consensus override responses.

**Principles:**
1. **Consistent standards.** Apply the same bar across iterations. Don't get stricter over time just because the bar "should" keep rising. Overturn precedent only with new evidence.
2. **Show your work.** Every ruling needs reasoning. "DISMISSED" without explanation is not acceptable.
3. **Evaluate the adversary too.** If the adversary is being unfair (violating fair play), call it out. If the adversary is being too lenient, call that out too.
4. **The metric is production risk.** The ultimate question: would this test suite catch the kinds of bugs that would actually ship? Not: is every line covered?
5. **Address consensus overrides.** If both Writer and Adversary flagged a previous ruling, you must engage with their reasoning and either reverse or provide stronger justification for the original ruling.

**JUDGMENT.md structure:**
```markdown
## Final Verdict: PASS | FAIL
## Calibrated Confidence: N%

## Consensus Override Responses
- [Previous ruling + both agents' objections + judge's response]

## Critique Validation
- Finding 1: VALID / DISMISSED — [reasoning]
- Finding 2: VALID / DISMISSED — [reasoning]

## Rebuttal Rulings
- Writer's rebuttal on X: ACCEPTED / REJECTED — [reasoning]

## Adversary Evaluation Rulings
- Adversary's objection on Y: ACCEPTED / REJECTED — [reasoning]

## Adversary Conduct Assessment
- [Was the adversary fair? Substantive? Did it acknowledge resolved issues?]

## Priorities for Next Iteration
1. [Highest priority validated finding]
2. [Second priority]
3. [...]
```

---

## Orchestrator Prompt

The ralph loop session receives this prompt each iteration. Its job is lightweight: dispatch agents and check exit criteria.

```markdown
# Test Rewrite Orchestrator — {REPO}

You are orchestrating an adversarial test suite redesign for the {REPO} repository.

## State Files (in repo root)
- TEST_PLAN.md — Writer's evolving test strategy
- RESULTS.md — Writer's latest results + confidence
- CRITIQUE.md — Adversary's latest critique + confidence
- RISK_MAP.md — Adversary's risk assessment
- JUDGMENT.md — Judge's latest ruling + calibrated confidence
- REBUTTAL.md — Writer's appeal (if any)
- EVALUATION.md — Adversary's appeal (if any)

## Each Iteration

### Step 1: Read current state
- Read JUDGMENT.md (if exists)
- Read RESULTS.md (if exists)
- Check for consensus overrides: did BOTH REBUTTAL.md and EVALUATION.md
  flag the same ruling? If so, note it for the Judge.
- Check git log for changes since last iteration

### Step 2: Spawn Writer + Adversary (parallel)
Use the Task tool to spawn TWO general-purpose subagents simultaneously:
- Writer agent with full writer prompt + current state
- Adversary agent with full adversary prompt + current state
Wait for both to complete.

### Step 3: Spawn Judge
Use the Task tool to spawn a general-purpose subagent with:
- Judge prompt + all artifacts from Steps 1-2
- Consensus override notes (if any)
Wait for completion.

### Step 4: Evaluate exit criteria
Read JUDGMENT.md. Check ALL of:
1. Judge verdict = PASS
2. Coverage >= {COVERAGE_TARGET}%
3. Mutation score >= {MUTATION_TARGET}%
4. Confidence scores converge (writer, adversary, judge within 10% of each other)
5. No unaddressed consensus overrides

If ALL criteria met: <promise>TESTS COMPLETE</promise>
If not: summarize what's still needed. The loop will continue.
```

---

## Per-Repo Configuration

| Repo | Framework | Test Command | Coverage Target | Mutation Target | Notes |
|------|-----------|-------------|-----------------|-----------------|-------|
| apollo (Python) | pytest | `poetry run pytest -v --cov` | 85% | 80% | |
| apollo (webapp) | vitest | `cd webapp && npm test -- --coverage` | 80% | 75% | Use stryker-js for mutations |
| sophia | pytest | `poetry run pytest tests/unit/ -v --cov` | 85% | 80% | Largest Python codebase |
| hermes | pytest | `poetry run pytest -v --cov` | 85% | 80% | ML tests skip without deps |
| talos | pytest | `poetry run pytest -v --cov` | 90% | 85% | Already has 95% CI threshold |
| logos | pytest | `poetry run pytest -v --cov` | 85% | 80% | Shared foundry code |

### Mutation Testing Tools
- **Python repos:** mutmut (`poetry add --group dev mutmut`)
- **TypeScript (webapp):** Stryker (`npx stryker run`)

---

## Invocation

For each repo, the operator runs:

```bash
/ralph-loop "You are the test rewrite orchestrator for {REPO}. Follow the orchestrator prompt in docs/plans/2026-02-21-adversarial-test-rewrite-design.md. The repo root is /Users/cdaly/projects/LOGOS/{REPO}." --max-iterations 20 --completion-promise "TESTS COMPLETE"
```

### Recommended order
1. **hermes** — smallest codebase, highest test density. Good pilot.
2. **talos** — small, already high threshold. Quick win.
3. **apollo (Python)** — moderate size, clear testing patterns.
4. **sophia** — largest, most architectural churn. Benefits from patterns learned above.
5. **logos** — shared foundry, test last since it affects all others.
6. **apollo (webapp)** — different framework (vitest), do separately.

---

## Artifacts Summary

| File | Producer | Purpose |
|------|----------|---------|
| `TEST_PLAN.md` | Writer | Living test strategy with rationale |
| `RESULTS.md` | Writer | Coverage, mutation score, confidence, response to judgment |
| `REBUTTAL.md` | Writer | Appeal of judge rulings (optional) |
| `CRITIQUE.md` | Adversary | Findings with severity, confidence, verdict |
| `RISK_MAP.md` | Adversary | Where risk lives vs where testing effort is |
| `EVALUATION.md` | Adversary | Appeal of judge rulings (optional) |
| `JUDGMENT.md` | Judge | Authoritative ruling, calibrated confidence, verdict |
| `MUTMUT_SUMMARY.md` | Writer (tool output) | Raw mutation testing results |

---

## Exit Criteria (all must hold)

1. **Judge verdict = PASS** — the judge, having considered all evidence, finds the test suite adequate
2. **Coverage >= threshold** — per-repo target met
3. **Mutation score >= threshold** — per-repo target met
4. **Confidence convergence** — writer, adversary, and judge confidence scores within 10% of each other
5. **No unaddressed consensus overrides** — all cases where writer and adversary agreed the judge was wrong have been addressed

## Safety Rails

- **Max iterations:** 20 per repo (prevent infinite loops)
- **Git commits each iteration:** Full rollback capability
- **Clean-slate scope:** Only test files are rewritten. Source code is read-only.
- **Existing tests preserved in git:** Can always recover previous test suite from git history
