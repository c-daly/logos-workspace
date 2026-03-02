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
  local repo_name
  repo_name="$(basename "$repo_path")"
  printf '\n=== %s ===\n' "$repo_name"
  (cd "$repo_path" && git pull --ff-only)
}

# Scan for git repositories under ROOT_DIR
while IFS= read -r -d '' git_dir; do
  repo_dir="${git_dir%/.git}"
  update_repo "$repo_dir"
done < <(find "$ROOT_DIR" -type d -name .git -print0)
