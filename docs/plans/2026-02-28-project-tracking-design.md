# LOGOS Project Tracking Redesign

**Date:** 2026-02-28
**Status:** Approved
**Approach:** GitHub-native (Projects board + automation + reconciliation)

## Problem

1. Existing issues are stale and don't reflect the current state of the project
2. Recent work has happened without corresponding issues/PRs
3. No big-picture view of what's planned, in flight, or done across repos
4. Agents can't reliably update board status, causing drift from reality
5. Labels and tags are inconsistent across repos
6. Phase-based language is obsolete and needs removal

## Design

### 1. Cleanup

#### Phase language removal

- Delete labels: `phase:1`, `phase:2`, `phase:3`, `phase 1 closers` from all repos
- Delete labels: `workstream:A`, `workstream:B`, `workstream:C` from all repos
- Delete label: `sprint:apollo-prototype` from logos
- Update `priority:high` description to "High priority — critical path or blocking other work"
- Close or rename milestone "Phase 1 - HCG and Abstract Pipeline" in logos
- Rename issues with phase language in titles (#416, #415, #346)

#### Label standardization

Converge all 5 repos on an identical canonical label set with consistent naming (colon separator, lowercase).

**Canonical labels:**

| Category | Labels |
|----------|--------|
| Component | `component:logos`, `component:sophia`, `component:hermes`, `component:talos`, `component:apollo`, `component:infrastructure` |
| Type | `type:bug`, `type:feature`, `type:refactor`, `type:testing`, `type:documentation`, `type:research`, `type:automation` |
| Priority | `priority:critical`, `priority:high`, `priority:medium`, `priority:low` |
| Status | `status:todo`, `status:in-progress`, `status:in-review`, `status:blocked` |
| Domain | `domain:hcg`, `domain:planner`, `domain:diagnostics` |
| Capability | `capability:perception`, `capability:actuation`, `capability:explainability` |
| Surface | `surface:browser`, `surface:cli`, `surface:llm` |

**Delete from all repos:**

- Phase labels: `phase:1`, `phase:2`, `phase:3`, `phase 1 closers`
- Workstream labels: `workstream:A`, `workstream:B`, `workstream:C`
- Sprint labels: `sprint:apollo-prototype`
- Bare duplicates: `bug`, `documentation`, `enhancement`, `refactor`, `testing`, `infrastructure`, `chore`, `ci`, `codex`, `epic`, `tech-debt`, `sophia`, `hermes`, `apollo`, `talos`, `logos`, `blocked`, `cross-repo`, `observability`
- GitHub default duplicates: `good-first-issue`, `help-wanted` (keep `good first issue`, `help wanted`)
- Malformed labels: `documentation:#EDEDED`, `observability:#EDEDED`, `testing:#EDEDED`
- Separator-inconsistent: `status/planned`, `status/in-progress`, `status/in-review`
- Repo-unique orphans: `area/docs`, `configuration`, `feature` (apollo), `api` (sophia)
- Redundant namespaced: `type:enhancement` (use `type:feature`), `type:infrastructure` (use `component:infrastructure`), `type:tracking`, `type:ui`, `scope:cross-repo`, `priority:medium-high`
- Obsolete status labels: `status:on-hold`, `status:todo` (if not in canonical set — actually `status:todo` IS canonical, keep it)
- Labels to delete: `decision-required`, `dependencies`, `standardization`

**Ensure every repo has:**

- All canonical labels with identical names, descriptions, and colors
- No extra labels beyond canonical + GitHub defaults (`good first issue`, `help wanted`, `duplicate`, `invalid`, `wontfix`, `question`)

#### Stale issue triage

- Review all ~50 open issues in logos, categorize as: still relevant, already done, superseded, needs rewrite
- Same pass on sophia (5), hermes (2), talos (2), apollo (4)
- Close issues that are done or superseded with a note explaining why
- Rewrite issues flagged as needing updates

#### Backfill untracked work

- Audit recent git history across repos for significant changes with no corresponding issue
- Create retroactive issues (closed) so there's a record

#### Cross-repo reconciliation

- Reconcile logos issues with `[Sophia]`, `[Apollo]`, `[Hermes]` prefixes against repo-local issues
- Deduplicate where the same work is tracked in two places
- Decide canonical location: repo-local issue is the source of truth, logos cross-repo references should link to them

#### Orphaned issues

- Add all open issues from sophia, hermes, talos, apollo to the project board
- These were never associated with the project and need to be added during triage

### 2. Project Board Structure

A single cross-repo GitHub Projects (v2) board.

**Views:**

- **Kanban** — columns by status (Todo, In Progress, In Review, Done, Blocked)
- **Roadmap** — timeline view grouped by epic
- **Backlog** — table view, sortable/filterable by priority, component, type

**Custom fields (project-level):**

- **Epic** (single select) — high-level goal grouping
- **Priority** (single select) — mirrors label for board-level filtering
- **Component** (single select) — mirrors label for board-level filtering

**Epics:**

Defined as a project field, not as issues. Likely candidates (finalized during triage):

- Cognitive Loop
- Observability/OTel
- Auth & Security
- Memory & Reflection
- Testing Infrastructure
- Developer Experience

**Issue templates:**

Standardized template in each repo:

```markdown
## Context
<!-- What exists today, why this matters -->

## Objective
<!-- What "done" looks like -->

## Acceptance Criteria
- [ ] ...

## Notes
<!-- Constraints, related issues, links to design docs -->
```

### 3. Automation

The goal: status updates happen as a side effect of normal git/PR workflow, not as an explicit step agents must remember.

#### Minimal agent contract

1. Work on a branch named `type/issue-number-description` (e.g., `feat/123-add-auth`)
2. Include `Closes #N` or `Part of #N` in the PR body
3. That's it

#### GitHub Actions (in each repo)

- **PR opened referencing an issue** → set issue label to `status:in-progress`, update project board to "In Progress"
- **PR marked ready for review** → set issue label to `status:in-review`, update project board to "In Review"
- **PR merged with `Closes #N`** → project board moves to "Done", clean up status labels

#### Claude Code hooks/skills

- PR creation hook: warn if branch/PR doesn't reference an issue
- Skill for creating well-formed issues from design docs

#### Periodic reconciliation script

A `gh`-based script that reports:

- Merged PRs that didn't close an issue
- Issues labeled `status:in-progress` with no open PR
- Issues with no activity in 30+ days
- Issues in repos not added to the project board

Run manually or on a weekly cron via GitHub Actions.

## Execution Order

### Step 1: Foundation

- Add `read:project` and `project` scopes to `gh` token
- Standardize labels across all 5 repos
- Delete obsolete labels (phase, workstream, sprint, duplicates, malformed)
- Fix `priority:high` description

### Step 2: Triage and Board

- Triage all open issues (close stale, update current, backfill untracked)
- Reconcile cross-repo duplicates
- Remove phase language from issue titles
- Close/rename phase milestones
- Set up project board with views and custom fields
- Add all open issues to the board
- Assign epics
- Create issue templates in each repo

### Step 3: Automation

- GitHub Actions workflows for PR → issue status sync
- Claude Code hook for PR creation (enforce issue linking)
- Reconciliation script
- Document conventions (branch naming, PR body format, agent contract)

## Prerequisites

- `gh` token needs `read:project` and `project` scopes
