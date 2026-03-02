#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

declare -A seen

update_repo() {
  local repo_path="$1"
  local git_dir="$repo_path/.git"
  if [[ ! -d "$git_dir" ]]; then
    return
  fi
  if [[ -n "${seen[$repo_path]:-}" ]]; then
    return
  fi
  seen[$repo_path]=1

  local repo_name branch upstream
  repo_name="$(basename "$repo_path")"
  branch="$(git -C "$repo_path" branch --show-current 2>/dev/null || echo "detached")"

  printf '\n=== %s [%s] ===\n' "$repo_name" "$branch"

  # Fetch all remotes
  git -C "$repo_path" fetch --all --prune 2>&1

  # Fast-forward current branch if it has an upstream
  upstream="$(git -C "$repo_path" rev-parse --abbrev-ref "@{upstream}" 2>/dev/null || true)"
  if [[ -n "$upstream" ]]; then
    if git -C "$repo_path" merge --ff-only "$upstream" 2>/dev/null; then
      echo "Fast-forwarded to $upstream"
    else
      echo "Cannot fast-forward to $upstream (diverged or local changes)"
    fi
  else
    echo "No upstream set — fetched only"
  fi
}

# Scan for git repositories under ROOT_DIR
while IFS= read -r -d '' git_dir; do
  repo_dir="${git_dir%/.git}"
  update_repo "$repo_dir"
done < <(find "$ROOT_DIR" -type d -name .git -print0)

# Sync wiki to logos-workspace
printf '\n=== wiki → logos-workspace sync ===\n'
(cd "$ROOT_DIR/wiki" && git push workspace master 2>&1 || echo "Wiki sync failed (non-fatal)")
