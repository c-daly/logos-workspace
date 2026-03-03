# Project Tracking Redesign — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Clean up stale project tracking, standardize labels across all repos, set up a cross-repo GitHub Projects board with epics, and add automation so status stays in sync with git/PR workflow.

**Architecture:** GitHub-native approach — GitHub Issues as source of truth, GitHub Projects (v2) for cross-repo views and roadmap, GitHub Actions for automated status sync, Claude Code hooks for enforcement.

**Tech Stack:** GitHub CLI (`gh`), GitHub Actions, GitHub Projects v2 GraphQL API

**Repos:** `c-daly/logos`, `c-daly/sophia`, `c-daly/hermes`, `c-daly/talos`, `c-daly/apollo`

**Prerequisite:** The `gh` CLI token must have `read:project` and `project` scopes. Update at https://github.com/settings/tokens before starting. Verify with:
```bash
gh auth status
```

**Design doc:** `docs/plans/2026-02-28-project-tracking-design.md`

---

## Task 1: Delete obsolete labels from all repos

Delete phase, workstream, sprint, and malformed labels from every repo.

**Files:** None (GitHub API operations only)

**Step 1: Delete phase labels**

Run for each repo:
```bash
for repo in c-daly/logos c-daly/sophia c-daly/hermes c-daly/talos c-daly/apollo; do
  echo "=== $repo ==="
  for label in "phase:1" "phase:2" "phase:3" "phase 1 closers"; do
    gh label delete "$label" --repo "$repo" --yes 2>/dev/null && echo "  deleted: $label" || echo "  not found: $label"
  done
done
```

**Step 2: Delete workstream labels**

```bash
for repo in c-daly/logos c-daly/sophia c-daly/hermes c-daly/talos c-daly/apollo; do
  echo "=== $repo ==="
  for label in "workstream:A" "workstream:B" "workstream:C"; do
    gh label delete "$label" --repo "$repo" --yes 2>/dev/null && echo "  deleted: $label" || echo "  not found: $label"
  done
done
```

**Step 3: Delete sprint labels**

```bash
gh label delete "sprint:apollo-prototype" --repo c-daly/logos --yes
```

**Step 4: Delete malformed labels (color leaked into name)**

```bash
for repo in c-daly/sophia c-daly/hermes c-daly/talos c-daly/apollo; do
  echo "=== $repo ==="
  for label in "documentation:#EDEDED" "observability:#EDEDED" "testing:#EDEDED"; do
    gh label delete "$label" --repo "$repo" --yes 2>/dev/null && echo "  deleted: $label" || echo "  not found: $label"
  done
done
```

**Step 5: Delete separator-inconsistent labels (Apollo slash-style)**

```bash
for label in "status/planned" "status/in-progress" "status/in-review"; do
  gh label delete "$label" --repo c-daly/apollo --yes 2>/dev/null && echo "  deleted: $label" || echo "  not found: $label"
done
```

**Step 6: Verify no obsolete labels remain**

```bash
for repo in c-daly/logos c-daly/sophia c-daly/hermes c-daly/talos c-daly/apollo; do
  echo "=== $repo ==="
  gh label list --repo "$repo" --limit 100 --json name --jq '.[].name' | grep -iE "phase|workstream|sprint|#EDEDED|status/" || echo "  clean"
done
```
Expected: all repos print "clean"

**Step 7: Commit**

No file changes — these are GitHub API operations. No commit needed.

---

## Task 2: Delete bare duplicate and orphan labels from all repos

Remove labels that duplicate the namespaced canonical set or are repo-specific orphans.

**Step 1: Delete bare duplicates from all repos**

```bash
BARE_DUPES=(
  "bug" "documentation" "enhancement" "refactor" "testing" "infrastructure"
  "chore" "ci" "codex" "epic" "tech-debt" "observability"
  "sophia" "hermes" "apollo" "talos" "logos"
  "blocked" "cross-repo"
)

for repo in c-daly/logos c-daly/sophia c-daly/hermes c-daly/talos c-daly/apollo; do
  echo "=== $repo ==="
  for label in "${BARE_DUPES[@]}"; do
    gh label delete "$label" --repo "$repo" --yes 2>/dev/null && echo "  deleted: $label" || echo "  skip: $label"
  done
done
```

**Step 2: Delete GitHub default duplicates (keep the space versions)**

```bash
for repo in c-daly/logos c-daly/sophia c-daly/hermes c-daly/talos c-daly/apollo; do
  for label in "good-first-issue" "help-wanted"; do
    gh label delete "$label" --repo "$repo" --yes 2>/dev/null && echo "  deleted: $label from $repo"
  done
done
```

**Step 3: Delete redundant namespaced labels**

```bash
REDUNDANT=(
  "type:enhancement" "type:infrastructure" "type:tracking" "type:ui"
  "scope:cross-repo" "priority:medium-high"
  "decision-required" "dependencies" "standardization"
  "status:on-hold"
)

for repo in c-daly/logos c-daly/sophia c-daly/hermes c-daly/talos c-daly/apollo; do
  echo "=== $repo ==="
  for label in "${REDUNDANT[@]}"; do
    gh label delete "$label" --repo "$repo" --yes 2>/dev/null && echo "  deleted: $label" || echo "  skip: $label"
  done
done
```

**Step 4: Delete repo-specific orphans**

```bash
# Apollo-specific
for label in "area/docs" "configuration" "feature"; do
  gh label delete "$label" --repo c-daly/apollo --yes 2>/dev/null
done

# Sophia-specific
gh label delete "api" --repo c-daly/sophia --yes 2>/dev/null
```

**Step 5: Fix Talos casing**

```bash
gh label edit "component:Talos" --repo c-daly/talos --name "component:talos" --description "Related to Talos hardware abstraction layer (sensors, actuators)"
```

**Step 6: Verify — list remaining labels per repo**

```bash
for repo in c-daly/logos c-daly/sophia c-daly/hermes c-daly/talos c-daly/apollo; do
  echo "=== $repo ==="
  gh label list --repo "$repo" --limit 100 --json name --jq '.[].name' | sort
  echo ""
done
```

Review output manually to confirm only canonical + GitHub default labels remain.

---

## Task 3: Ensure canonical labels exist with correct descriptions in all repos

Create any missing canonical labels and update descriptions/colors to be identical everywhere.

**Step 1: Apply canonical label set**

```bash
# Define canonical labels: "name|description|color"
CANONICAL=(
  "component:logos|Related to the LOGOS meta-repo (specs, ontology, SDKs, shared tooling)|5319e7"
  "component:sophia|Related to Sophia cognitive core (Orchestrator, CWM-A, CWM-G, Planner, Executor)|0e8a16"
  "component:hermes|Related to Hermes language utility (STT, TTS, NLP, embeddings)|1d76db"
  "component:talos|Related to Talos hardware abstraction layer (sensors, actuators)|fbca04"
  "component:apollo|Related to Apollo UI/client (dashboard, visualization, command interface)|d93f0b"
  "component:infrastructure|Related to infrastructure, HCG, ontology, CI/CD, or repository setup|5319e7"
  "type:bug|Bug fix or error correction|d73a4a"
  "type:feature|New feature or capability|0052cc"
  "type:refactor|Code refactoring or improvement|fbca04"
  "type:testing|Test creation or improvement|bfd4f2"
  "type:documentation|Documentation or knowledge management|0075ca"
  "type:research|Research sprint or investigation|8b4789"
  "type:automation|Automation scripts or tooling|ededed"
  "priority:critical|Critical — blocking all progress|b60205"
  "priority:high|High priority — critical path or blocking other work|b60205"
  "priority:medium|Medium priority — important but not blocking|d93f0b"
  "priority:low|Low priority — nice to have or future enhancement|fbca04"
  "status:todo|Planned but not yet started|ededed"
  "status:in-progress|Currently being worked on|0e8a16"
  "status:in-review|Ready for or currently in review|fbca04"
  "status:blocked|Blocked by dependencies or external factors|d93f0b"
  "domain:hcg|Hybrid Cognitive Graph data model, ontology, validation, storage|c5def5"
  "domain:planner|Planning/execution logic, Sophia orchestrator, Talos capability registry|d4c5f9"
  "domain:diagnostics|Observability, metrics, tracing, UX diagnostics panels|fef2c0"
  "capability:perception|Perception, media ingestion, causal world model updates|bfdadc"
  "capability:actuation|Execution via Talos capabilities (simulated or physical)|d4c5f9"
  "capability:explainability|Diagnostics, visualization, explainability tooling|c5def5"
  "surface:browser|Browser-based UI/dashboard experience|1d76db"
  "surface:cli|CLI experience (Apollo CLI, scripts, GH Action summaries)|ededed"
  "surface:llm|Multimodal LLM integration surface using LOGOS as a co-processor|8b4789"
)

for repo in c-daly/logos c-daly/sophia c-daly/hermes c-daly/talos c-daly/apollo; do
  echo "=== $repo ==="
  for entry in "${CANONICAL[@]}"; do
    IFS='|' read -r name desc color <<< "$entry"
    # Try to edit first (updates existing), then create if it doesn't exist
    gh label edit "$name" --repo "$repo" --description "$desc" --color "$color" 2>/dev/null \
      && echo "  updated: $name" \
      || (gh label create "$name" --repo "$repo" --description "$desc" --color "$color" 2>/dev/null \
        && echo "  created: $name" \
        || echo "  FAILED: $name")
  done
done
```

**Step 2: Verify label consistency**

```bash
for repo in c-daly/logos c-daly/sophia c-daly/hermes c-daly/talos c-daly/apollo; do
  echo "=== $repo ==="
  gh label list --repo "$repo" --limit 100 --json name,description --jq '.[] | select(.name | startswith("component:") or startswith("type:") or startswith("priority:") or startswith("status:") or startswith("domain:") or startswith("capability:") or startswith("surface:")) | "\(.name)\t\(.description)"' | sort
  echo ""
done
```

Visually confirm all repos have identical output for the canonical set.

**Step 3: Commit**

No file changes. GitHub API only.

---

## Task 4: Fix priority:high description and clean up milestones

**Step 1: Update priority:high description (already handled in Task 3)**

Verify it says "High priority — critical path or blocking other work" (not "critical for current phase").

**Step 2: Close phase milestone in logos**

```bash
cd /Users/cdaly/projects/LOGOS/logos
# Get milestone number
gh api repos/c-daly/logos/milestones --jq '.[] | select(.title | test("Phase 1")) | .number'
# Close it (replace N with the number)
gh api repos/c-daly/logos/milestones/N -X PATCH -f state=closed
```

**Step 3: Verify**

```bash
gh api repos/c-daly/logos/milestones --jq '.[] | "\(.title) (\(.state))"'
```

---

## Task 5: Remove phase language from issue titles

**Step 1: Rename issues with phase in title**

```bash
cd /Users/cdaly/projects/LOGOS/logos

# #416: "Phase 2.5: Testing Sanity - Prerequisite to Phase 3"
gh issue edit 416 --title "Testing Sanity — prerequisite to learning & memory work"

# #415: "Phase 3: Learning & Memory Systems - Master Tracking"
gh issue edit 415 --title "Learning & Memory Systems — Master Tracking"

# #346: "Phase 3: Evaluate Observability Stack Requirements"
gh issue edit 346 --title "Evaluate Observability Stack Requirements"
```

**Step 2: Verify**

```bash
gh issue list --repo c-daly/logos --state open --json number,title | python3.12 -c "
import sys, json
for i in json.load(sys.stdin):
    if 'phase' in i['title'].lower() or 'Phase' in i['title']:
        print(f'  STILL HAS PHASE: #{i[\"number\"]} {i[\"title\"]}')
" && echo "clean" || true
```

---

## Task 6: Delete phase-related GitHub Actions workflows

These workflows reference the old phase structure and are no longer relevant.

**Files:**
- Delete: `logos/.github/workflows/create-phase1-issues.yml`
- Delete: `logos/.github/workflows/phase1-gate.yml`
- Delete: `logos/.github/workflows/phase2-e2e.yml`
- Delete: `logos/.github/workflows/phase2-otel.yml`
- Delete: `logos/.github/workflows/phase2-perception.yml`

**Step 1: Verify the workflows are not referenced elsewhere**

```bash
cd /Users/cdaly/projects/LOGOS/logos
grep -r "create-phase1-issues\|phase1-gate\|phase2-e2e\|phase2-otel\|phase2-perception" .github/ --include="*.yml" -l
```

Expected: only the files themselves match.

**Step 2: Delete the workflows**

```bash
cd /Users/cdaly/projects/LOGOS/logos
rm .github/workflows/create-phase1-issues.yml
rm .github/workflows/phase1-gate.yml
rm .github/workflows/phase2-e2e.yml
rm .github/workflows/phase2-otel.yml
rm .github/workflows/phase2-perception.yml
```

**Step 3: Commit**

```bash
cd /Users/cdaly/projects/LOGOS/logos
git add .github/workflows/
git commit -m "chore: remove obsolete phase-related GitHub Actions workflows"
```

---

## Task 7: Update issue templates in logos

Replace the existing workstream/phase-referencing templates with a single clean template. The individual component templates can stay but need the workstream dropdown removed.

**Files:**
- Modify: `logos/.github/ISSUE_TEMPLATE/logos-task.yml`
- Modify: `logos/.github/ISSUE_TEMPLATE/sophia-task.yml`
- Modify: `logos/.github/ISSUE_TEMPLATE/hermes-task.yml`
- Modify: `logos/.github/ISSUE_TEMPLATE/talos-task.yml`
- Modify: `logos/.github/ISSUE_TEMPLATE/apollo-task.yml`
- Modify: `logos/.github/ISSUE_TEMPLATE/research-task.yml`
- Modify: `logos/.github/ISSUE_TEMPLATE/documentation-task.yml`

**Step 1: Read all existing templates to understand current structure**

```bash
cd /Users/cdaly/projects/LOGOS/logos
for f in .github/ISSUE_TEMPLATE/*.yml; do
  echo "=== $f ==="
  cat "$f"
  echo ""
done
```

**Step 2: Update each template**

Remove any `workstream` dropdown field. Remove any phase references from descriptions. Ensure each template has these fields:
- Priority dropdown (critical/high/medium/low)
- Context textarea (required)
- Objective textarea (required)
- Acceptance Criteria textarea (required)
- Notes textarea (optional — dependencies, related issues, links)

The exact edits depend on reading the current templates; apply the same pattern to each.

**Step 3: Verify templates are valid YAML**

```bash
cd /Users/cdaly/projects/LOGOS/logos
python3.12 -c "
import yaml, glob
for f in glob.glob('.github/ISSUE_TEMPLATE/*.yml'):
    with open(f) as fh:
        yaml.safe_load(fh)
    print(f'  OK: {f}')
"
```

**Step 4: Commit**

```bash
cd /Users/cdaly/projects/LOGOS/logos
git add .github/ISSUE_TEMPLATE/
git commit -m "chore: update issue templates — remove workstream/phase references, standardize fields"
```

---

## Task 8: Create issue templates in downstream repos

sophia, hermes, talos, and apollo currently have no issue templates. Create a standard template in each.

**Files:**
- Create: `sophia/.github/ISSUE_TEMPLATE/task.yml`
- Create: `sophia/.github/ISSUE_TEMPLATE/config.yml`
- Create: `hermes/.github/ISSUE_TEMPLATE/task.yml`
- Create: `hermes/.github/ISSUE_TEMPLATE/config.yml`
- Create: `talos/.github/ISSUE_TEMPLATE/task.yml`
- Create: `talos/.github/ISSUE_TEMPLATE/config.yml`
- Create: `apollo/.github/ISSUE_TEMPLATE/task.yml`
- Create: `apollo/.github/ISSUE_TEMPLATE/config.yml`

**Step 1: Create template for each repo**

Template content (adapt `component:X` label and title prefix per repo):

```yaml
# task.yml
name: Task
description: Create a new task
title: ""
labels: ["component:SERVICE"]
body:
  - type: dropdown
    id: priority
    attributes:
      label: Priority
      options:
        - priority:critical
        - priority:high
        - priority:medium
        - priority:low
    validations:
      required: true
  - type: textarea
    id: context
    attributes:
      label: Context
      description: What exists today, why this matters
      placeholder: Describe the current state and motivation
    validations:
      required: true
  - type: textarea
    id: objective
    attributes:
      label: Objective
      description: What "done" looks like
      placeholder: Describe the desired end state
    validations:
      required: true
  - type: textarea
    id: acceptance
    attributes:
      label: Acceptance Criteria
      description: Specific, verifiable criteria for completion
      placeholder: |
        - [ ] Criterion 1
        - [ ] Criterion 2
    validations:
      required: true
  - type: textarea
    id: notes
    attributes:
      label: Notes
      description: Constraints, related issues, links to design docs
      placeholder: Optional additional context
    validations:
      required: false
```

```yaml
# config.yml
blank_issues_enabled: true
```

**Step 2: Create directories and files**

```bash
for repo in sophia hermes talos apollo; do
  mkdir -p /Users/cdaly/projects/LOGOS/$repo/.github/ISSUE_TEMPLATE
done
```

Write the `task.yml` and `config.yml` for each repo, replacing `SERVICE` with the repo name.

**Step 3: Commit in each repo**

```bash
for repo in sophia hermes talos apollo; do
  cd /Users/cdaly/projects/LOGOS/$repo
  git add .github/ISSUE_TEMPLATE/
  git commit -m "chore: add standardized issue template"
done
```

---

## Task 9: Triage open issues across all repos

This is a manual/interactive task. For each open issue, decide: keep, close, or rewrite.

**Step 1: Generate triage report**

```bash
for repo in c-daly/logos c-daly/sophia c-daly/hermes c-daly/talos c-daly/apollo; do
  echo "=== $repo ==="
  gh issue list --repo "$repo" --state open --limit 100 --json number,title,labels,updatedAt --jq '.[] | "  #\(.number) \(.title) [updated: \(.updatedAt | split("T")[0])]"'
  echo ""
done
```

**Step 2: For each issue, classify**

Go through each issue and mark it:
- **Keep** — still relevant, update description if needed
- **Close (done)** — work was already completed; close with comment: "Completed in <commit/PR reference>"
- **Close (superseded)** — no longer applicable; close with comment explaining why
- **Rewrite** — keep open but update title/description to reflect current state

**Step 3: Reconcile cross-repo duplicates**

Issues in logos with `[Sophia]`, `[Apollo]`, `[Hermes]`, `[Talos]` prefixes may duplicate repo-local issues. For each:
- If a repo-local issue exists for the same work, close the logos issue with a cross-reference
- If no repo-local issue exists, decide whether the issue belongs in logos or should be transferred

**Step 4: Backfill untracked work**

```bash
for repo in logos sophia hermes talos apollo; do
  echo "=== $repo ==="
  cd /Users/cdaly/projects/LOGOS/$repo
  git log --oneline --since="2026-01-01" --no-merges | head -30
  echo ""
done
```

Compare against closed issues. For significant changes with no issue, create a retroactive closed issue.

---

## Task 10: Set up GitHub Projects board

**Prerequisite:** `gh` token has `read:project` and `project` scopes.

**Step 1: Create the project**

```bash
gh project create --owner c-daly --title "LOGOS" --format "BOARD"
```

Note the project number from the output.

**Step 2: Add custom fields**

Using the GraphQL API (gh project field-create may work, otherwise use gh api graphql):

```bash
PROJECT_NUM=<number from step 1>

# Epic field
gh project field-create $PROJECT_NUM --owner c-daly --name "Epic" --data-type "SINGLE_SELECT" \
  --single-select-options "Cognitive Loop,Observability,Auth & Security,Memory & Reflection,Testing Infrastructure,Developer Experience"

# (Priority and Component fields may not be needed if labels suffice — evaluate during setup)
```

**Step 3: Add all open issues to the project**

```bash
PROJECT_NUM=<number>

for repo in c-daly/logos c-daly/sophia c-daly/hermes c-daly/talos c-daly/apollo; do
  echo "=== $repo ==="
  gh issue list --repo "$repo" --state open --limit 100 --json url --jq '.[].url' | while read url; do
    gh project item-add $PROJECT_NUM --owner c-daly --url "$url" && echo "  added: $url" || echo "  FAILED: $url"
  done
done
```

**Step 4: Set up board views**

This must be done in the GitHub UI (Projects v2 views aren't fully CLI-configurable):

1. **Kanban view** (default) — group by Status field (Todo, In Progress, In Review, Done, Blocked)
2. **Roadmap view** — add a Roadmap layout, group by Epic
3. **Backlog view** — Table layout, all items, sortable columns

**Step 5: Assign epics to issues**

Based on triage results from Task 9, assign each issue to an epic using the project board UI or GraphQL API.

---

## Task 11: Create GitHub Actions workflow for PR-to-issue status sync

Add a workflow to each repo that auto-updates issue status when PR events occur.

**Files:**
- Create: `logos/.github/workflows/issue-sync.yml`
- Create: `sophia/.github/workflows/issue-sync.yml`
- Create: `hermes/.github/workflows/issue-sync.yml`
- Create: `talos/.github/workflows/issue-sync.yml`
- Create: `apollo/.github/workflows/issue-sync.yml`

**Step 1: Write the workflow**

```yaml
# .github/workflows/issue-sync.yml
name: Sync Issue Status from PRs

on:
  pull_request:
    types: [opened, ready_for_review, closed]

jobs:
  sync-status:
    runs-on: ubuntu-latest
    permissions:
      issues: write
      pull-requests: read
    steps:
      - name: Extract linked issue numbers
        id: issues
        uses: actions/github-script@v7
        with:
          script: |
            const body = context.payload.pull_request.body || '';
            const branch = context.payload.pull_request.head.ref || '';

            // Match "Closes #N", "Fixes #N", "Part of #N"
            const bodyMatches = body.match(/(?:closes|fixes|resolves|part of)\s+#(\d+)/gi) || [];
            const issueNums = bodyMatches.map(m => m.match(/#(\d+)/)[1]);

            // Match branch pattern: type/123-description
            const branchMatch = branch.match(/^\w+\/(\d+)/);
            if (branchMatch && !issueNums.includes(branchMatch[1])) {
              issueNums.push(branchMatch[1]);
            }

            core.setOutput('numbers', JSON.stringify([...new Set(issueNums)]));
            core.info(`Found issues: ${issueNums.join(', ')}`);

      - name: Update issue labels
        uses: actions/github-script@v7
        with:
          script: |
            const issueNums = JSON.parse('${{ steps.issues.outputs.numbers }}');
            const pr = context.payload.pull_request;
            const action = context.payload.action;

            for (const num of issueNums) {
              const issueNum = parseInt(num);
              const statusLabels = ['status:todo', 'status:in-progress', 'status:in-review', 'status:blocked'];

              // Remove all existing status labels
              const issue = await github.rest.issues.get({
                owner: context.repo.owner,
                repo: context.repo.repo,
                issue_number: issueNum,
              });
              const currentLabels = issue.data.labels.map(l => l.name);
              const nonStatusLabels = currentLabels.filter(l => !statusLabels.includes(l));

              let newStatus;
              if (action === 'closed' && pr.merged) {
                // PR merged — don't add a status label, issue will be closed by "Closes #N"
                newStatus = null;
              } else if (action === 'ready_for_review' || (action === 'opened' && !pr.draft)) {
                newStatus = 'status:in-review';
              } else if (action === 'opened' && pr.draft) {
                newStatus = 'status:in-progress';
              }

              const updatedLabels = newStatus ? [...nonStatusLabels, newStatus] : nonStatusLabels;

              await github.rest.issues.update({
                owner: context.repo.owner,
                repo: context.repo.repo,
                issue_number: issueNum,
                labels: updatedLabels,
              });

              core.info(`Updated #${issueNum}: status=${newStatus || 'cleared'}`);
            }
```

**Step 2: Write the file to each repo**

Copy the workflow to all 5 repos.

**Step 3: Commit in each repo**

```bash
for repo in logos sophia hermes talos apollo; do
  cd /Users/cdaly/projects/LOGOS/$repo
  git add .github/workflows/issue-sync.yml
  git commit -m "ci: add PR-to-issue status sync workflow"
done
```

---

## Task 12: Create reconciliation script

A script to audit drift between git activity and issue tracking.

**Files:**
- Create: `logos/scripts/reconcile-issues.sh`

**Step 1: Write the script**

```bash
#!/usr/bin/env bash
# reconcile-issues.sh — audit drift between git and issue tracking
# Run from the LOGOS workspace root

set -euo pipefail

REPOS=("c-daly/logos" "c-daly/sophia" "c-daly/hermes" "c-daly/talos" "c-daly/apollo")
DAYS_STALE=30

echo "=== LOGOS Issue Reconciliation Report ==="
echo "Date: $(date +%Y-%m-%d)"
echo ""

echo "## Merged PRs without linked issues (last 30 days)"
for repo in "${REPOS[@]}"; do
  echo "--- $repo ---"
  gh pr list --repo "$repo" --state merged --limit 50 --json number,title,body,mergedAt \
    --jq ".[] | select(.mergedAt > \"$(date -v-30d +%Y-%m-%dT00:00:00Z)\") | select((.body | test(\"(?i)(closes|fixes|resolves|part of)\\\\s+#\") | not) and (.body | length > 0 == false | not)) | \"  PR #\\(.number): \\(.title)\"" 2>/dev/null || true
done
echo ""

echo "## Issues labeled in-progress with no open PR"
for repo in "${REPOS[@]}"; do
  echo "--- $repo ---"
  gh issue list --repo "$repo" --label "status:in-progress" --state open --json number,title \
    --jq '.[] | "  #\(.number) \(.title)"' 2>/dev/null || true
done
echo ""

echo "## Issues with no activity in ${DAYS_STALE}+ days"
STALE_DATE=$(date -v-${DAYS_STALE}d +%Y-%m-%dT00:00:00Z)
for repo in "${REPOS[@]}"; do
  echo "--- $repo ---"
  gh issue list --repo "$repo" --state open --limit 100 --json number,title,updatedAt \
    --jq ".[] | select(.updatedAt < \"$STALE_DATE\") | \"  #\\(.number) \\(.title) [last: \\(.updatedAt | split(\"T\")[0])]\"" 2>/dev/null || true
done
echo ""

echo "=== End Report ==="
```

**Step 2: Make it executable**

```bash
chmod +x /Users/cdaly/projects/LOGOS/logos/scripts/reconcile-issues.sh
```

**Step 3: Test it**

```bash
cd /Users/cdaly/projects/LOGOS/logos
./scripts/reconcile-issues.sh
```

**Step 4: Commit**

```bash
cd /Users/cdaly/projects/LOGOS/logos
git add scripts/reconcile-issues.sh
git commit -m "chore: add issue reconciliation audit script"
```

---

## Task 13: Document conventions

Write a short conventions doc that agents and humans can reference.

**Files:**
- Create: `logos/docs/PROJECT_TRACKING.md`

**Step 1: Write the doc**

Content should cover:
- Branch naming: `type/issue-number-description` (e.g., `feat/123-add-auth`, `fix/456-null-check`)
- PR body: must include `Closes #N` or `Part of #N`
- Labels: link to canonical label set, explain when to use each category
- Project board: where to find it, what the views are, how epics work
- Issue creation: use templates, write for an agent audience (context, objective, acceptance criteria)
- Reconciliation: how and when to run the audit script

**Step 2: Commit**

```bash
cd /Users/cdaly/projects/LOGOS/logos
git add docs/PROJECT_TRACKING.md
git commit -m "docs: add project tracking conventions"
```

---

## Task 14: Push all changes

**Step 1: Push each repo**

```bash
for repo in logos sophia hermes talos apollo; do
  echo "=== $repo ==="
  cd /Users/cdaly/projects/LOGOS/$repo
  git push origin HEAD
done
```

**Step 2: Verify CI passes**

Check each repo's CI status after push.
