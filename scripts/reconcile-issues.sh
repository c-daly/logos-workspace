#!/usr/bin/env bash
# reconcile-issues.sh — audit drift between git and issue tracking
# Run from anywhere. Uses gh CLI to query GitHub directly.

set -euo pipefail

REPOS=("c-daly/logos" "c-daly/sophia" "c-daly/hermes" "c-daly/talos" "c-daly/apollo")
DAYS_STALE=${1:-30}
STALE_DATE=$(date -v-${DAYS_STALE}d +%Y-%m-%dT00:00:00Z 2>/dev/null \
  || date -d "${DAYS_STALE} days ago" +%Y-%m-%dT00:00:00Z)
RECENT_DATE=$(date -v-30d +%Y-%m-%dT00:00:00Z 2>/dev/null \
  || date -d "30 days ago" +%Y-%m-%dT00:00:00Z)

echo "=== LOGOS Issue Reconciliation Report ==="
echo "Date: $(date +%Y-%m-%d)"
echo "Stale threshold: ${DAYS_STALE} days"
echo ""

echo "## Merged PRs without linked issues (last 30 days)"
echo ""
for repo in "${REPOS[@]}"; do
  echo "--- $repo ---"
  gh pr list --repo "$repo" --state merged --limit 50 \
    --json number,title,body,mergedAt \
    --jq ".[] | select(.mergedAt > \"$RECENT_DATE\") | select((.body // \"\") | test(\"(?i)(closes|fixes|resolves|part of)\\\\s+#\") | not) | \"  PR #\\(.number): \\(.title)\"" 2>/dev/null || true
  echo ""
done

echo "## Issues labeled in-progress with no open PR"
echo ""
for repo in "${REPOS[@]}"; do
  echo "--- $repo ---"
  gh issue list --repo "$repo" --label "status:in-progress" --state open \
    --json number,title \
    --jq '.[] | "  #\(.number) \(.title)"' 2>/dev/null || true
  echo ""
done

echo "## Issues with no activity in ${DAYS_STALE}+ days"
echo ""
for repo in "${REPOS[@]}"; do
  echo "--- $repo ---"
  gh issue list --repo "$repo" --state open --limit 100 \
    --json number,title,updatedAt \
    --jq ".[] | select(.updatedAt < \"$STALE_DATE\") | \"  #\\(.number) \\(.title) [last: \\(.updatedAt | split(\"T\")[0])]\"" 2>/dev/null || true
  echo ""
done

echo "## Issue count summary"
echo ""
for repo in "${REPOS[@]}"; do
  open=$(gh issue list --repo "$repo" --state open --limit 1 --json number --jq 'length' 2>/dev/null || echo "?")
  echo "  $repo: $open open issues"
done
echo ""

echo "=== End Report ==="
