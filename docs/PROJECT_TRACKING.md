# LOGOS Project Tracking Conventions

## Overview

LOGOS uses GitHub Issues as the source of truth for work tracking, with a GitHub Projects board for cross-repo visibility. Status stays in sync automatically through PR conventions and GitHub Actions.

## Branch Naming

```
type/issue-number-description
```

Examples:
- `feat/123-add-auth`
- `fix/456-null-check-in-planner`
- `chore/789-bump-foundry`
- `refactor/101-simplify-executor`

Valid types: `feat`, `fix`, `chore`, `refactor`, `test`, `docs`, `ci`, `perf`

## PR Body Format

Every PR must reference an issue:

```
Closes #123
```

or for partial work:

```
Part of #123
```

Supported keywords: `Closes`, `Fixes`, `Resolves`, `Part of`

When a PR with `Closes #N` is merged, GitHub automatically closes the issue.

## Labels

All repos use an identical canonical label set. Do not create repo-specific labels.

| Category | Labels | Purpose |
|----------|--------|---------|
| **Component** | `component:logos`, `component:sophia`, `component:hermes`, `component:talos`, `component:apollo`, `component:infrastructure` | Which service |
| **Type** | `type:bug`, `type:feature`, `type:refactor`, `type:testing`, `type:documentation`, `type:research`, `type:automation` | What kind of work |
| **Priority** | `priority:critical`, `priority:high`, `priority:medium`, `priority:low` | How urgent |
| **Status** | `status:todo`, `status:in-progress`, `status:in-review`, `status:blocked` | Current state (auto-updated by PR workflow) |
| **Domain** | `domain:hcg`, `domain:planner`, `domain:diagnostics` | Problem area |
| **Capability** | `capability:perception`, `capability:actuation`, `capability:explainability` | System capability |
| **Surface** | `surface:browser`, `surface:cli`, `surface:llm` | User interaction surface |

**Status labels are updated automatically** when PRs are opened, reviewed, or merged. You generally don't need to set them manually.

## Issue Creation

Use the issue template in each repo. Every issue should have:

- **Context**: What exists today, why this matters
- **Objective**: What "done" looks like
- **Acceptance Criteria**: Specific, verifiable items (checkboxes)
- **Notes** (optional): Constraints, related issues, design doc links

Write issues as if an agent with no project context will pick them up. Be specific about files, APIs, and expected behavior.

## Project Board

The LOGOS GitHub Project board provides three views:

- **Kanban**: Columns by status (Todo, In Progress, In Review, Done, Blocked)
- **Roadmap**: Timeline grouped by epic
- **Backlog**: Table view, sortable by priority/component/type

Issues are grouped by **Epic** (a project-level field, not a label). Epics represent high-level goals like "Cognitive Loop", "Observability", "Auth & Security".

## Automation

### What happens automatically

The `issue-sync.yml` workflow in each repo handles status transitions:

| PR Event | Issue Label Change |
|----------|-------------------|
| Draft PR opened | `status:in-progress` |
| Non-draft PR opened | `status:in-review` |
| PR marked ready for review | `status:in-review` |
| PR merged with `Closes #N` | Status labels cleared, issue closed |

### Minimal agent contract

Agents only need to:

1. Work on a branch named `type/issue-number-description`
2. Include `Closes #N` or `Part of #N` in the PR body
3. That's it — automation handles the rest

## Reconciliation

Run the audit script periodically to catch drift:

```bash
./scripts/reconcile-issues.sh        # default: 30 day stale threshold
./scripts/reconcile-issues.sh 14     # custom: 14 day threshold
```

The script reports:
- Merged PRs that didn't close an issue
- Issues labeled `status:in-progress` with no open PR
- Issues with no activity past the stale threshold
- Issue count summary per repo
