#!/usr/bin/env bash
#
# bootstrap.sh — idempotent environment setup for the LOGOS workspace.
#
# Takes a bare machine to a ready-to-run state: installs the toolchain
# (uv, Poetry, fnm/Node), pins every repo to Python 3.12, installs deps
# with the correct per-repo extras, builds the vendored webapp SDKs,
# scaffolds .env files, and pulls the infra images.
#
# It does NOT start services — that is run_apollo.sh's job. This script
# is safe to re-run: every phase checks-then-acts and converges.
#
# Usage:  ./bootstrap.sh [--with-ml] [--skip-clone] [--help]
#
# POSIX-friendly bash (no associative arrays) so it runs on macOS's
# stock bash 3.2 as well as Linux/WSL2.
set -euo pipefail

WORKSPACE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Foundry first — the others depend on logos-foundry.
REPOS="logos sophia hermes talos apollo"

# --- logging ---------------------------------------------------------------
RED=$'\033[0;31m'; GREEN=$'\033[0;32m'; YELLOW=$'\033[1;33m'; BLUE=$'\033[0;34m'; NC=$'\033[0m'
log_info() { printf '%s[INFO]%s %s\n' "$BLUE"   "$NC" "$*"; }
log_ok()   { printf '%s[ OK ]%s %s\n' "$GREEN"  "$NC" "$*"; }
log_warn() { printf '%s[WARN]%s %s\n' "$YELLOW" "$NC" "$*" >&2; }
log_err()  { printf '%s[ERR ]%s %s\n' "$RED"    "$NC" "$*" >&2; }
die()      { log_err "$*"; exit 1; }

# --- args ------------------------------------------------------------------
WITH_ML=false
SKIP_CLONE=false

usage() {
  cat <<'EOF'
bootstrap.sh — idempotent environment setup for the LOGOS workspace.

Installs the toolchain (uv, Poetry, fnm/Node), pins every repo to Python
3.12, installs deps with the correct per-repo extras, builds the vendored
webapp SDKs, scaffolds .env files, and pulls infra images. Does NOT start
services (use apollo/scripts/run_apollo.sh). Safe to re-run.

Usage: ./bootstrap.sh [--with-ml] [--skip-clone] [--help]

  --with-ml      include heavy ML deps (torch) for sophia/hermes
  --skip-clone   assume repos are already checked out
  --help         show this help
EOF
}

parse_args() {
  for arg in "$@"; do
    case "$arg" in
      --with-ml)    WITH_ML=true ;;
      --skip-clone) SKIP_CLONE=true ;;
      -h|--help)    usage; exit 0 ;;
      *)            die "unknown argument: $arg (try --help)" ;;
    esac
  done
}

# --- os detection ----------------------------------------------------------
OS="unknown"
detect_os() {
  case "$(uname -s)" in
    Linux*)  OS="linux"; grep -qi microsoft /proc/version 2>/dev/null && OS="wsl" || true ;;
    Darwin*) OS="macos" ;;
    *)       die "unsupported OS: $(uname -s)" ;;
  esac
}

# --- per-repo dependency config (single source of truth) -------------------
# Prints the `poetry install` arguments for a repo. Prevents the class of bug
# where an extra is requested from a repo that doesn't define it.
repo_install_args() {
  local args="--no-interaction"
  case "$1" in
    apollo) args="$args -E otel" ;;
    sophia) args="$args -E otel"; [ "$WITH_ML" = true ] && args="$args --with ml" ;;  # ml = group
    hermes) args="$args -E otel"; [ "$WITH_ML" = true ] && args="$args -E ml" ;;      # ml = extra
    logos|talos) : ;;                                                    # no extras
    *) die "unknown repo: $1" ;;
  esac
  printf '%s' "$args"
}

# --- toolchain -------------------------------------------------------------
# uv/poetry/fnm install to ~/.local/bin or ~/.local/share; ensure on PATH for
# the rest of this run regardless of the user's shell rc.
export PATH="$HOME/.local/bin:$HOME/.local/share/fnm:$PATH"

ensure_uv() {
  if command -v uv >/dev/null 2>&1; then log_ok "uv present ($(uv --version))"; return; fi
  log_info "installing uv…"
  curl -LsSf https://astral.sh/uv/install.sh | sh
  command -v uv >/dev/null 2>&1 || die "uv install failed; ensure ~/.local/bin is on PATH"
  log_ok "uv installed ($(uv --version))"
}

ensure_poetry() {
  if command -v poetry >/dev/null 2>&1; then log_ok "poetry present ($(poetry --version))"; return; fi
  log_info "installing poetry…"
  # Install with the uv-managed 3.12, not system python3 (which may be missing
  # or an incompatible version like 3.14). Requires ensure_python312 to run first.
  curl -sSL https://install.python-poetry.org | "$PYTHON312" -
  command -v poetry >/dev/null 2>&1 || die "poetry install failed; ensure ~/.local/bin is on PATH"
  log_ok "poetry installed ($(poetry --version))"
}

ensure_fnm() {
  if command -v fnm >/dev/null 2>&1; then log_ok "fnm present"; return; fi
  log_info "installing fnm…"
  # Pin the install dir to match the PATH export above — the installer's
  # default has varied between ~/.fnm and ~/.local/share/fnm across versions.
  curl -fsSL https://fnm.vercel.app/install | bash -s -- --install-dir "$HOME/.local/share/fnm" --skip-shell
  command -v fnm >/dev/null 2>&1 || die "fnm install failed; ensure ~/.local/share/fnm is on PATH"
  log_ok "fnm installed"
}

ensure_docker() {
  command -v docker >/dev/null 2>&1 || die \
    "Docker not found. Install Docker Desktop (macOS / WSL2 integration) or docker-ce, then re-run. See docs/LOCAL_SETUP.md."
  docker info >/dev/null 2>&1 || die \
    "Docker is installed but the daemon isn't running. Start Docker Desktop / the docker service and re-run."
  log_ok "docker present and daemon up"
}

# --- language runtimes -----------------------------------------------------
PYTHON312=""
ensure_python312() {
  uv python install 3.12              # idempotent: no-op if already managed
  PYTHON312="$(uv python find 3.12)"  # stable uv-managed path (~/.local/share/uv/python/...)
  [ -n "$PYTHON312" ] && [ -x "$PYTHON312" ] || die "could not locate uv-managed Python 3.12"
  log_ok "python 3.12: $PYTHON312"
}

ensure_node20() {
  eval "$(fnm env --shell bash)" 2>/dev/null || true
  fnm install 20 >/dev/null 2>&1 || true   # idempotent
  fnm use 20    >/dev/null 2>&1 || true
  fnm default 20 >/dev/null 2>&1 || true
  command -v node >/dev/null 2>&1 || die "node unavailable after fnm install"
  log_ok "node $(node -v), npm $(npm -v)"
}

# --- repos -----------------------------------------------------------------
# Clone any missing sibling repo. Existing repos are left untouched —
# syncing them is pull_all.sh's job, not bootstrap's.
ensure_repos() {
  if $SKIP_CLONE; then log_info "skip-clone: not touching repos"; return; fi
  local r
  for r in $REPOS; do
    if [ -d "$WORKSPACE_ROOT/$r/.git" ]; then
      log_ok "$r checked out"
    else
      log_info "cloning $r…"
      git clone "git@github.com:c-daly/$r.git" "$WORKSPACE_ROOT/$r"
    fi
  done
}

# --- per-repo python env ---------------------------------------------------
# Pin each repo's Poetry venv to the uv-managed 3.12, remove any stale venv on
# a different interpreter (the 3.13/3.14 churn), and install with the correct
# extras. poetry install is run unquoted on purpose to word-split the args.
setup_python_repo() {
  local repo="$1" dir="$WORKSPACE_ROOT/$1"
  [ -f "$dir/pyproject.toml" ] || { log_warn "$repo: no pyproject.toml, skipping"; return; }
  log_info "$repo: pin Python 3.12 + install deps"
  (
    cd "$dir"
    # Drop venvs not on 3.12 so Poetry stops oscillating between interpreters.
    poetry env list 2>/dev/null | awk '{print $1}' | while read -r env; do
      case "$env" in
        "" | *-py3.12) : ;;
        *) poetry env remove "$env" >/dev/null 2>&1 || true ;;
      esac
    done
    poetry env use "$PYTHON312" >/dev/null
    # shellcheck disable=SC2046  # intentional word-splitting of install args
    poetry install $(repo_install_args "$repo")
  )
  log_ok "$repo: $(cd "$dir" && poetry run python -V 2>/dev/null)"
}

# --- apollo webapp ---------------------------------------------------------
# Vendored @logos/* SDKs are file: deps whose `prepare` runs `tsc`. Build each
# first (its own `npm install` pulls typescript + emits dist/), else the webapp
# install fails with `tsc: not found`. Idempotent: skip when dist/ exists.
setup_webapp() {
  local web="$WORKSPACE_ROOT/apollo/webapp" sdk
  [ -d "$web" ] || { log_warn "apollo/webapp not found, skipping"; return; }
  for sdk in "$web"/vendor/@logos/*/; do
    [ -f "${sdk}package.json" ] || continue
    if [ -d "${sdk}dist" ]; then
      log_ok "vendored sdk built: $(basename "$sdk")"
    else
      log_info "building vendored sdk $(basename "$sdk")…"
      ( cd "$sdk" && npm install >/dev/null )
    fi
  done
  if [ -d "$web/node_modules" ]; then
    log_ok "webapp deps present"
  else
    log_info "installing webapp deps…"
    ( cd "$web" && npm install >/dev/null )
  fi
}

# --- config: delegate to logos --------------------------------------------
# logos owns config distribution. render_test_stacks.py generates each repo's
# stack into logos/infra/<repo>/ (ports from logos_config); copy_test_stacks.py
# distributes those into each downstream repo's containers/ (the standard
# location). Both are logos's own deterministic, idempotent scripts — bootstrap
# just runs them (fresh machine: creates the configs; existing: updates them to
# what the logos configurator describes). Dev runtime also needs the one local
# secret, OPENAI_API_KEY (read by run_apollo.sh from apollo/.env).
distribute_config() {
  log_info "generating ecosystem config via logos…"
  ( cd "$WORKSPACE_ROOT/logos" && poetry run python infra/scripts/render_test_stacks.py )
  # copy_test_stacks.py lands via a separate logos PR; guard so an unmerged
  # dependency degrades to a warning instead of failing the whole bootstrap.
  if [ -f "$WORKSPACE_ROOT/logos/infra/scripts/copy_test_stacks.py" ]; then
    ( cd "$WORKSPACE_ROOT/logos" && poetry run python infra/scripts/copy_test_stacks.py )
    log_ok "config generated (logos/infra) + copied into each repo's containers/"
  else
    log_warn "copy_test_stacks.py not present (logos copy-script PR unmerged) — rendered to logos/infra only; per-repo copy skipped."
  fi
  ensure_openai_secret
}

# Ensure apollo/.env carries the OPENAI_API_KEY line; never overwrite a real value.
ensure_openai_secret() {
  local f="$WORKSPACE_ROOT/apollo/.env"
  if [ -f "$f" ] && grep -qE '^OPENAI_API_KEY=.+' "$f"; then
    log_ok "OPENAI_API_KEY set"
    return
  fi
  [ -f "$f" ] || : > "$f"
  grep -qE '^OPENAI_API_KEY=' "$f" || printf 'OPENAI_API_KEY=\n' >> "$f"
  log_warn "Set OPENAI_API_KEY in apollo/.env (required for Hermes LLM)."
}

# --- infra (prep only) -----------------------------------------------------
# Verify Docker and pull the Neo4j+Milvus images. Does NOT start containers —
# run_apollo.sh brings them up. Compose project name is 'infra'.
setup_infra() {
  local compose="$WORKSPACE_ROOT/logos/infra/docker-compose.hcg.dev.yml"
  [ -f "$compose" ] || { log_warn "compose file missing ($compose), skipping infra"; return; }
  log_info "pulling infra images…"
  docker compose -p infra -f "$compose" pull
  log_ok "infra images present (start them with run_apollo.sh)"
}

# --- summary ---------------------------------------------------------------
summary() {
  local r
  printf '\n%s=== bootstrap complete ===%s\n' "$GREEN" "$NC"
  printf '  uv:     %s\n' "$(uv --version 2>/dev/null)"
  printf '  poetry: %s\n' "$(poetry --version 2>/dev/null)"
  printf '  python: %s\n' "$PYTHON312"
  printf '  node:   %s\n' "$(node -v 2>/dev/null)"
  for r in $REPOS; do
    printf '  %-7s %s\n' "$r:" "$(cd "$WORKSPACE_ROOT/$r" 2>/dev/null && poetry run python -V 2>/dev/null || echo '-')"
  done
  printf '\nNext: ./apollo/scripts/run_apollo.sh\n'
}


main() {
  parse_args "$@"
  detect_os
  log_info "LOGOS workspace bootstrap — OS=$OS, root=$WORKSPACE_ROOT"
  ensure_uv
  ensure_python312   # before poetry: poetry is installed with this interpreter
  ensure_poetry
  ensure_fnm
  ensure_docker
  ensure_node20
  ensure_repos
  local r
  for r in $REPOS; do setup_python_repo "$r"; done
  setup_webapp
  distribute_config
  setup_infra
  summary
}

if [ "${BASH_SOURCE[0]}" = "${0}" ]; then
  main "$@"
fi
