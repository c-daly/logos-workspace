#!/usr/bin/env bash
set -euo pipefail

# Reinstall logos-foundry in all downstream repos from the git tag.
# Usage: ./scripts/reinstall-foundry.sh [tag]
#   tag defaults to the version in logos/pyproject.toml

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPOS=(sophia hermes talos apollo)
REPO_URL="git+https://github.com/c-daly/logos.git"

# Resolve tag: argument > pyproject.toml version
if [[ -n "${1:-}" ]]; then
  TAG="$1"
else
  TAG="v$(grep -m1 '^version' "$ROOT_DIR/logos/pyproject.toml" | sed 's/.*"\(.*\)"/\1/')"
  echo "Auto-detected tag: $TAG"
fi

FAILED=()

for repo in "${REPOS[@]}"; do
  repo_dir="$ROOT_DIR/$repo"
  if [[ ! -d "$repo_dir" ]]; then
    echo "[$repo] skipped (directory not found)"
    continue
  fi

  printf '\n=== %s ===\n' "$repo"

  if ! poetry -C "$repo_dir" run pip install --force-reinstall --no-deps "${REPO_URL}@${TAG}" 2>&1; then
    FAILED+=("$repo")
    continue
  fi

  # Verify import
  if poetry -C "$repo_dir" run python -c "from logos_config import RedisConfig; print('  OK: logos_config imports clean')" 2>&1; then
    installed=$(poetry -C "$repo_dir" run pip show logos-foundry 2>/dev/null | grep '^Version:' | awk '{print $2}')
    echo "  Installed: logos-foundry $installed"
  else
    echo "  WARN: import check failed"
    FAILED+=("$repo")
  fi
done

echo ""
if [[ ${#FAILED[@]} -eq 0 ]]; then
  echo "All repos updated to logos-foundry @ $TAG"
else
  echo "FAILED: ${FAILED[*]}"
  exit 1
fi
