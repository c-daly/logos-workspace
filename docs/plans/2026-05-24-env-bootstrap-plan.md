# Env Bootstrap Script Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `logos-workspace/bootstrap.sh` — one idempotent, well-commented script that takes a bare machine (WSL2 now, macOS later) to a ready-to-run LOGOS environment.

**Architecture:** A single bash script organized as readable phases (preflight → toolchain → python → node → repos → per-repo env → webapp SDKs → .env → infra prep → summary). A small space-separated repo list plus a `case`-based config function is the single source of truth for per-repo extras. `main` is guarded so the file can be sourced by a unit test. Environment prep only — starting services stays `run_apollo.sh`'s job.

**Tech Stack:** Bash (POSIX-friendly, no associative arrays → runs on macOS stock bash 3.2), uv (Python), Poetry, fnm (Node), Docker, shellcheck.

**Spec:** `docs/plans/2026-05-24-env-bootstrap-design.md`

---

## File structure

- Create: `logos-workspace/bootstrap.sh` — the script (single file, phased, heavily commented).
- Create: `logos-workspace/tests/bootstrap_test.sh` — unit test for the pure config logic (`repo_install_args`), sourcing the script via its guard.
- Modify: `logos-workspace/docs/LOCAL_SETUP.md` — point at `bootstrap.sh` as the automated path (Task 9).

**Conventions for every task:** after editing the script, run `shellcheck bootstrap.sh` (expect no errors) and `bash -n bootstrap.sh` (syntax). The current machine (PROMETHEUS) is already provisioned, so running the script here exercises the *idempotent* path (tools present → skip) and performs the intended 3.13→3.12 re-pin. The true cold-start path is validated in a fresh container in Task 10.

---

### Task 1: Skeleton — strict mode, header, logging, OS detection, args

**Files:**
- Create: `logos-workspace/bootstrap.sh`

- [ ] **Step 1: Write the script skeleton**

```bash
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
    Linux*)  OS="linux"; grep -qi microsoft /proc/version 2>/dev/null && OS="wsl" ;;
    Darwin*) OS="macos" ;;
    *)       die "unsupported OS: $(uname -s)" ;;
  esac
}

main() {
  parse_args "$@"
  detect_os
  log_info "LOGOS workspace bootstrap — OS=$OS, root=$WORKSPACE_ROOT"
  # phases wired in subsequent tasks
}

# Sourceable seam: only run main when executed, not when sourced by tests.
if [ "${BASH_SOURCE[0]}" = "${0}" ]; then
  main "$@"
fi
```

- [ ] **Step 2: Lint and syntax-check**

Run: `cd /home/fearsidhe/projects/logos-workspace && chmod +x bootstrap.sh && shellcheck bootstrap.sh && bash -n bootstrap.sh`
Expected: no output from shellcheck, exit 0.

- [ ] **Step 3: Verify help + run**

Run: `./bootstrap.sh --help && ./bootstrap.sh`
Expected: `--help` prints the usage/header; bare run prints the `OS=... root=...` line and exits 0.

- [ ] **Step 4: Commit**

```bash
git add bootstrap.sh
git commit -m "feat(bootstrap): skeleton — strict mode, logging, OS detect, args"
```

---

### Task 2: Per-repo config logic (with a real unit test)

This is the source of truth that prevents the `-E otel`-applied-where-undefined bug. TDD it because it is pure and was the original failure.

**Files:**
- Create: `logos-workspace/tests/bootstrap_test.sh`
- Modify: `logos-workspace/bootstrap.sh`

- [ ] **Step 1: Write the failing test**

Create `tests/bootstrap_test.sh`:

```bash
#!/usr/bin/env bash
# Unit tests for bootstrap.sh pure logic. Sources the script (main is guarded).
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
# shellcheck disable=SC1091
source ./bootstrap.sh

fail=0
assert_eq() { # $1=desc $2=expected $3=actual
  if [ "$2" = "$3" ]; then printf 'ok   - %s\n' "$1"
  else printf 'FAIL - %s\n      expected: %q\n      actual:   %q\n' "$1" "$2" "$3"; fail=1; fi
}

WITH_ML=false
assert_eq "apollo extras"        "--no-interaction -E otel" "$(repo_install_args apollo)"
assert_eq "sophia extras"        "--no-interaction -E otel" "$(repo_install_args sophia)"
assert_eq "hermes extras"        "--no-interaction -E otel" "$(repo_install_args hermes)"
assert_eq "logos no extras"      "--no-interaction"         "$(repo_install_args logos)"
assert_eq "talos no extras"      "--no-interaction"         "$(repo_install_args talos)"

WITH_ML=true
assert_eq "sophia ml is a group" "--no-interaction -E otel --with ml" "$(repo_install_args sophia)"
assert_eq "hermes ml is an extra" "--no-interaction -E otel -E ml"    "$(repo_install_args hermes)"

exit "$fail"
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `chmod +x tests/bootstrap_test.sh && ./tests/bootstrap_test.sh`
Expected: FAIL — `repo_install_args: command not found` (function not defined yet).

- [ ] **Step 3: Implement `repo_install_args` in `bootstrap.sh`**

Add after the args section (before os detection). Note the ml asymmetry is real: sophia's `ml` is a Poetry **group** (`--with ml`), hermes's `ml` is a PEP-621 **extra** (`-E ml`).

```bash
# --- per-repo dependency config (single source of truth) -------------------
# Prints the `poetry install` arguments for a repo. Prevents the class of bug
# where an extra is requested from a repo that doesn't define it.
repo_install_args() {
  local args="--no-interaction"
  case "$1" in
    apollo) args="$args -E otel" ;;
    sophia) args="$args -E otel"; $WITH_ML && args="$args --with ml" ;;  # ml = group
    hermes) args="$args -E otel"; $WITH_ML && args="$args -E ml" ;;      # ml = extra
    logos|talos) : ;;                                                    # no extras
    *) die "unknown repo: $1" ;;
  esac
  printf '%s' "$args"
}
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `./tests/bootstrap_test.sh && shellcheck bootstrap.sh tests/bootstrap_test.sh`
Expected: all `ok` lines, exit 0; shellcheck clean.

- [ ] **Step 5: Commit**

```bash
git add bootstrap.sh tests/bootstrap_test.sh
git commit -m "feat(bootstrap): per-repo install config + unit test"
```

---

### Task 3: Toolchain — uv, Poetry, fnm (install-if-missing), Docker (verify-only)

**Files:**
- Modify: `logos-workspace/bootstrap.sh`

- [ ] **Step 1: Add the toolchain functions**

Add before `main`:

```bash
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
  curl -sSL https://install.python-poetry.org | python3 -
  command -v poetry >/dev/null 2>&1 || die "poetry install failed; ensure ~/.local/bin is on PATH"
  log_ok "poetry installed ($(poetry --version))"
}

ensure_fnm() {
  if command -v fnm >/dev/null 2>&1; then log_ok "fnm present"; return; fi
  log_info "installing fnm…"
  curl -fsSL https://fnm.vercel.app/install | bash -s -- --skip-shell
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
```

- [ ] **Step 2: Wire into `main`**

Replace the `# phases wired in subsequent tasks` line with:

```bash
  ensure_uv
  ensure_poetry
  ensure_fnm
  ensure_docker
```

- [ ] **Step 3: Lint + run on this (provisioned) machine**

Run: `shellcheck bootstrap.sh && ./bootstrap.sh`
Expected: shellcheck clean; four `[ OK ]` lines (uv/poetry/fnm/docker all present — install branches skipped). Exit 0.

- [ ] **Step 4: Commit**

```bash
git add bootstrap.sh
git commit -m "feat(bootstrap): toolchain install (uv/poetry/fnm) + docker verify"
```

---

### Task 4: Python 3.12 + Node 20

**Files:**
- Modify: `logos-workspace/bootstrap.sh`

- [ ] **Step 1: Add the runtime functions**

```bash
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
```

- [ ] **Step 2: Wire into `main`** (after `ensure_docker`)

```bash
  ensure_python312
  ensure_node20
```

- [ ] **Step 3: Lint + run**

Run: `shellcheck bootstrap.sh && ./bootstrap.sh`
Expected: prints `python 3.12: /…/uv/python/cpython-3.12…/bin/python3.12` and `node v20.x`. Re-running is a fast no-op. Exit 0.

- [ ] **Step 4: Commit**

```bash
git add bootstrap.sh
git commit -m "feat(bootstrap): install + pin Python 3.12 and Node 20"
```

---

### Task 5: Repos — clone if missing

**Files:**
- Modify: `logos-workspace/bootstrap.sh`

- [ ] **Step 1: Add the function**

```bash
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
```

- [ ] **Step 2: Wire into `main`** (after `ensure_node20`)

```bash
  ensure_repos
```

- [ ] **Step 3: Lint + run**

Run: `shellcheck bootstrap.sh && ./bootstrap.sh`
Expected: five `[ OK ] <repo> checked out` lines (all present on this machine). Exit 0.

- [ ] **Step 4: Commit**

```bash
git add bootstrap.sh
git commit -m "feat(bootstrap): clone missing repos"
```

---

### Task 6: Per-repo Python env — pin 3.12, kill venv churn, install deps

**Files:**
- Modify: `logos-workspace/bootstrap.sh`

- [ ] **Step 1: Add the function**

```bash
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
```

- [ ] **Step 2: Wire into `main`** (after `ensure_repos`)

```bash
  local r
  for r in $REPOS; do setup_python_repo "$r"; done
```

- [ ] **Step 3: Lint + run**

Run: `shellcheck bootstrap.sh && ./bootstrap.sh`
Expected: each repo logs `Python 3.12.x`; any leftover `*-py3.13`/`*-py3.14` venvs are removed (this performs the intended re-pin). Re-running is fast (deps satisfied). Exit 0.

- [ ] **Step 4: Verify the re-pin took**

Run: `for r in $REPOS; do (cd "$r" && echo "$r -> $(poetry run python -V)"); done` (from workspace root, with `REPOS="logos sophia hermes talos apollo"`)
Expected: every repo reports `Python 3.12.x`.

- [ ] **Step 5: Commit**

```bash
git add bootstrap.sh
git commit -m "feat(bootstrap): pin repos to Python 3.12 and install deps"
```

---

### Task 7: Vendored webapp SDKs + webapp install

**Files:**
- Modify: `logos-workspace/bootstrap.sh`

- [ ] **Step 1: Add the function**

```bash
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
```

- [ ] **Step 2: Wire into `main`** (after the per-repo loop)

```bash
  setup_webapp
```

- [ ] **Step 3: Lint + run**

Run: `shellcheck bootstrap.sh && ./bootstrap.sh`
Expected: both vendored SDKs report built; webapp deps present/installed. Exit 0.

- [ ] **Step 4: Commit**

```bash
git add bootstrap.sh
git commit -m "feat(bootstrap): build vendored SDKs + webapp deps"
```

---

### Task 8: Config distribution (run logos scripts) + infra image prep + summary, then wire `main`

**Files:**
- Modify: `logos-workspace/bootstrap.sh`

- [ ] **Step 1: Add the functions**

```bash
# --- config: delegate to logos --------------------------------------------
# logos owns config distribution. render_test_stacks.py stamps each repo's
# .env.test + docker-compose.test.yml IN PLACE (its --output-root defaults to
# each repo's path; ports injected from logos_config). We do NOT hand-roll
# .env copying. We only provision the one genuine local secret the dev stack
# needs: OPENAI_API_KEY (read by run_apollo.sh from apollo/.env). Idempotent.
distribute_config() {
  log_info "rendering ecosystem config via logos (writes .env.test + compose into each repo)…"
  ( cd "$WORKSPACE_ROOT/logos" && poetry run python infra/scripts/render_test_stacks.py )
  log_ok "config rendered (ports from logos_config; 'render_test_stacks.py --check' guards drift)"
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
```

- [ ] **Step 2: Wire into `main`** (after `setup_webapp`)

```bash
  distribute_config
  setup_infra
  summary
```

- [ ] **Step 3: Lint + full run**

Run: `shellcheck bootstrap.sh && ./bootstrap.sh`
Expected: logos render runs and writes each repo's `.env.test`/compose; `OPENAI_API_KEY` reported set or warned; infra images pulled; summary block prints uv/poetry/python/node + each repo on Python 3.12. Exit 0.

- [ ] **Step 4: Commit**

```bash
git add bootstrap.sh
git commit -m "feat(bootstrap): delegate config to logos render, infra image prep, summary"
```

---

### Task 9: Point `LOCAL_SETUP.md` at the script

**Files:**
- Modify: `logos-workspace/docs/LOCAL_SETUP.md`

- [ ] **Step 1: Add an automated-path note near the top**

After the intro line ("This guide takes you from a fresh machine to a running LOGOS stack."), insert:

```markdown
> **Automated path:** after cloning this workspace, run `./bootstrap.sh` — it
> installs the toolchain, pins every repo to Python 3.12, installs deps, builds
> the webapp SDKs, scaffolds `.env` files, and pulls the infra images. It is
> idempotent (safe to re-run) and does everything below except start services
> (use `apollo/scripts/run_apollo.sh` for that). The manual steps below remain
> as reference and for non-standard setups.
```

- [ ] **Step 2: Verify the doc renders and the link is correct**

Run: `grep -n "bootstrap.sh" docs/LOCAL_SETUP.md`
Expected: the new note is present.

- [ ] **Step 3: Commit**

```bash
git add docs/LOCAL_SETUP.md
git commit -m "docs: point LOCAL_SETUP at bootstrap.sh"
```

---

### Task 10: Cold-start verification in a clean container

This is the real proof — the prior tasks only exercised the idempotent path on an already-provisioned box.

**Files:** none (verification only).

- [ ] **Step 1: Run shellcheck across everything**

Run: `shellcheck bootstrap.sh tests/bootstrap_test.sh && ./tests/bootstrap_test.sh`
Expected: shellcheck clean; unit tests all `ok`.

- [ ] **Step 2: Idempotency check — run twice on this machine**

Run: `./bootstrap.sh && ./bootstrap.sh`
Expected: both exit 0; the second run logs only `[ OK ]`/present lines (no installs, no clones, no venv removals, no SDK builds).

- [ ] **Step 3: Cold-start in a fresh Ubuntu container**

Run:
```bash
docker run --rm -it -v "$PWD":/ws -w /ws ubuntu:24.04 bash -lc '
  apt-get update -qq && apt-get install -y -qq curl git ca-certificates python3 >/dev/null
  ./bootstrap.sh --skip-clone   # repos already mounted; skip SSH clone in container
'
```
Expected: toolchain installs from scratch (uv/poetry/fnm), Python 3.12 + Node 20 install, per-repo venvs build on 3.12, SDKs build, `.env` files created, exit 0. (Docker-in-container is unavailable, so the infra `pull` will warn/skip — acceptable for this check; the goal is to prove the language/dep cold-start.)

- [ ] **Step 4: Commit any fixes surfaced by the container run**

```bash
git add -A && git commit -m "fix(bootstrap): cold-start issues from clean-container run"
```
(Skip if the container run was clean.)

---

## Out of scope (tracked separately — see spec follow-ups)

- **Re-pin 3.13 → 3.12:** performed automatically by Task 6 when the script runs on PROMETHEUS; record the correction as a new memory entry (not an edit).
- **neo4j stale-pidfile guard in `run_apollo.sh`:** the auto-recreate-on-stale-pidfile fix belongs in `run_apollo.sh`'s infra `up` step, not in bootstrap (which is infra-prep-only). Separate change in the apollo repo.
- **Thin out per-repo `scripts/setup-local-dev.sh`:** make them shims that defer to `bootstrap.sh`, or remove. Separate cleanup once bootstrap is proven.
