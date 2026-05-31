#!/usr/bin/env bash
set -euo pipefail

# check-foundry-drift.sh — flag downstream repos whose logos-foundry / SDK pins
# lag the current logos release.
#
# A downstream repo "lags" when any of its logos-* git dependencies in
# pyproject.toml pins a tag older than the latest logos release. This is the
# read-only audit complement to scripts/reinstall-foundry.sh (which performs
# the bump).
#
# Usage:
#   ./scripts/check-foundry-drift.sh [latest_tag]
#
#   latest_tag  Optional. The release every downstream repo should be on.
#               Defaults to "v<version>" from logos/pyproject.toml, falling
#               back to the newest semver git tag in the logos checkout.
#
# Exit codes:
#   0  all downstream pins are at (or ahead of) the latest release
#   1  one or more repos lag — details printed to stdout
#   2  usage / environment error (e.g. cannot determine latest release)
#
# Designed to run locally or in CI (e.g. a workspace cron / GitHub Action).

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DOWNSTREAM_REPOS=(sophia hermes talos apollo)

# Packages published from the logos monorepo that downstream repos pin by tag.
LOGOS_PACKAGES="logos-foundry logos-sophia-sdk logos-hermes-sdk"

# --- Resolve the latest logos release -------------------------------------

resolve_latest_tag() {
  if [[ -n "${1:-}" ]]; then
    printf '%s\n' "$1"
    return 0
  fi

  local pyproject="$ROOT_DIR/logos/pyproject.toml"
  if [[ -f "$pyproject" ]]; then
    local version
    version="$(grep -m1 '^version' "$pyproject" | sed 's/.*"\(.*\)".*/\1/')"
    if [[ -n "$version" ]]; then
      printf 'v%s\n' "$version"
      return 0
    fi
  fi

  # Fallback: newest semver tag in the logos checkout.
  if [[ -d "$ROOT_DIR/logos/.git" ]]; then
    local tag
    tag="$(git -C "$ROOT_DIR/logos" tag --list 'v*' --sort=-v:refname | head -n1)"
    if [[ -n "$tag" ]]; then
      printf '%s\n' "$tag"
      return 0
    fi
  fi

  return 1
}

# Compare two "vMAJOR.MINOR.PATCH" tags. Echoes "older", "equal", or "newer"
# describing how $1 relates to $2 (the latest release).
semver_cmp() {
  local a="${1#v}" b="${2#v}"
  local IFS=.
  # shellcheck disable=SC2206
  local av=($a) bv=($b)
  local i
  for i in 0 1 2; do
    local ai="${av[$i]:-0}" bi="${bv[$i]:-0}"
    # Strip any pre-release/build suffix from the patch component.
    ai="${ai%%[-+]*}"; bi="${bi%%[-+]*}"
    if ((10#${ai:-0} < 10#${bi:-0})); then echo older; return; fi
    if ((10#${ai:-0} > 10#${bi:-0})); then echo newer; return; fi
  done
  echo equal
}

LATEST="$(resolve_latest_tag "${1:-}")" || {
  echo "ERROR: could not determine the latest logos release." >&2
  echo "       Pass it explicitly: $0 v0.7.1" >&2
  exit 2
}

echo "=== Foundry Pin Drift Check ==="
echo "Date:            $(date +%Y-%m-%d)"
echo "Latest release:  $LATEST"
echo ""

LAGGING=()

for repo in "${DOWNSTREAM_REPOS[@]}"; do
  pyproject="$ROOT_DIR/$repo/pyproject.toml"
  if [[ ! -f "$pyproject" ]]; then
    echo "[$repo] skipped (pyproject.toml not found)"
    continue
  fi

  repo_lags=0
  for pkg in $LOGOS_PACKAGES; do
    # Match e.g.  logos-foundry = { git = "...logos.git", tag = "v0.5.0" }
    # Tolerates optional spaces and ordering of git/tag keys.
    while IFS= read -r line; do
      tag="$(printf '%s\n' "$line" | sed -n 's/.*tag *= *"\(v[0-9][^"]*\)".*/\1/p')"
      [[ -z "$tag" ]] && continue
      rel="$(semver_cmp "$tag" "$LATEST")"
      if [[ "$rel" == older ]]; then
        echo "[$repo] LAGS: $pkg pinned at $tag (latest $LATEST)"
        repo_lags=1
      fi
    done < <(grep -E "^[[:space:]]*${pkg}[[:space:]]*=" "$pyproject" || true)
  done

  if [[ "$repo_lags" -eq 0 ]]; then
    echo "[$repo] OK (at $LATEST)"
  else
    LAGGING+=("$repo")
  fi
done

echo ""
if [[ ${#LAGGING[@]} -eq 0 ]]; then
  echo "All downstream repos are current with logos $LATEST."
  exit 0
fi

echo "DRIFT DETECTED in: ${LAGGING[*]}"
echo "Fix with: ./scripts/reinstall-foundry.sh $LATEST  (then bump the pins + relock)"
exit 1
