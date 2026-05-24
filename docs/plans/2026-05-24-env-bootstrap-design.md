# Env Bootstrap Script — Design

- **Date:** 2026-05-24
- **Status:** Approved (design); pending spec review
- **Artifact:** `logos-workspace/bootstrap.sh`

## Motivation

A run of `apollo/scripts/run_apollo.sh` on a freshly-checked-out workspace (new machine, PROMETHEUS) surfaced a cluster of failures: sophia's `poetry install -E otel` failed because sophia had no `otel` extra (masked by `|| true`), the webapp build died on `tsc: not found` building vendored SDKs, and Poetry built venvs on inconsistent interpreters (uv 3.13 vs system 3.14).

These were **not caused by the new machine** — they were *unmasked* by it. The previous (now-dead) machine had accreted working state — populated venvs, pre-built SDK `dist/`, a matching interpreter — that hid latent first-run assumptions. A clean checkout is the first honest test of the cold-start path, and that path was never validated.

The fix for the *class* of problem is a tested, idempotent bootstrap that assumes nothing pre-exists. This is the reproducibility goal made concrete: one command on a blank box reaches a known-good state.

## Goal & non-goals

**Goal:** take a bare machine (WSL2/Ubuntu now, macOS later) to a ready-to-run LOGOS stack, idempotently.

**Non-goal — does NOT start app services.** Environment prep only. Starting Sophia/Hermes/Apollo stays `run_apollo.sh`'s job, which may assume `bootstrap.sh` has run. Clean split:
- `bootstrap.sh` — set up once; safe to re-run.
- `run_apollo.sh` — start the stack each session.

## Decisions

| Decision | Choice |
|----------|--------|
| Scope | Full bare-machine bootstrap (installs toolchain) |
| Install strategy | Official cross-platform installers; Docker verify-only |
| Python | **3.12**, uv-installed and pinned per repo (per `LOCAL_SETUP.md`: 3.14 breaks pymilvus/protobuf) |
| Node | **20 LTS** via fnm (webapp needs `>=18`; 18 is EOL) |
| Infra | **Prep-only**: verify Docker + `compose pull`; do NOT bring containers up |
| Repos | All five (logos, sophia, hermes, talos, apollo) |
| Idempotency | Hard requirement — check-then-act, converges, re-run is a fast no-op |
| Form | Single well-commented `bootstrap.sh`, organized as readable phases |

## Phases

The script reads top-to-bottom as this narrative; each phase is guarded for idempotency.

1. **Preflight** — `set -euo pipefail`; detect OS (Linux/WSL2 vs macOS via `uname`); define the repo config table + colored logging helpers (mirroring `apollo/scripts/lib/common.sh` style).
2. **Toolchain** — install-if-missing via official installers: `uv` (`astral.sh/uv/install.sh`), `poetry` (official installer), `fnm`. Docker is **verify-only**: if absent or the daemon is down, print OS-specific guidance and stop (the one thing we won't auto-install).
3. **Python** — `uv python install 3.12`; resolve its stable path (`~/.local/share/uv/python/...`, via the `~/.local/bin/python3.12` shim — never the hash-addressed `~/.cache/uv/archive-*` path) for pinning.
4. **Node** — `fnm install 20` (+ default); verify `node`/`npm`.
5. **Repos** — clone any missing sibling repo from `git@github.com:c-daly/<repo>.git`; if the dir already exists, leave it untouched. Syncing existing repos is out of scope — that's `pull_all.sh`'s job, not bootstrap's.
6. **Per-repo env** — for each repo: `poetry env use <uv-3.12>` then `poetry install` with the correct extras (table below). Remove any stray venv not on 3.12 (kills the interpreter churn).
7. **Vendored SDKs** — for `apollo/webapp/vendor/@logos/*`: build if `dist/` absent (`npm install` → `prepare` → `tsc`), then `apollo/webapp` `npm install`.
8. **Config (delegate to logos)** — do NOT hand-roll `.env`. logos owns config distribution: bootstrap runs `render_test_stacks.py` (generates each repo's stack into `logos/infra/<repo>/`; ports from `logos_config`), then `copy_test_stacks.py` (distributes those into each downstream repo's `containers/` — the standard location, skipping logos which is source-only). Both are deterministic and idempotent: on a fresh machine they create the configs, on an existing one they update them to what the configurator describes. Bootstrap also provisions the one local secret, `OPENAI_API_KEY`, in `apollo/.env`.
9. **Infra (prep)** — verify Docker; `docker compose -p infra -f logos/infra/docker-compose.hcg.dev.yml pull`. Do not `up`.
10. **Verify + summary** — print tool versions, per-repo Python + venv path, image pull status, config/secret status; end with `next: ./apollo/scripts/run_apollo.sh`.

> **Config model:** `logos` is the central config system — `logos/infra/test_stack/` (`repos.yaml`+`services.yaml`+`overlays/`) + `logos_config.ports` → `render_test_stacks.py` generates per-repo stacks into `logos/infra/<repo>/`, and the companion `copy_test_stacks.py` distributes them into each downstream repo's `containers/` (the standard location; `tests/e2e/stack/<repo>/` is the legacy schema only apollo/sophia have migrated off). These configs are *derived*, so (re)generating + copying them is the correct, idempotent behavior, not a hazard — fresh machines get them created, existing ones get them updated to match the configurator. Dev *runtime* reads `logos_config` defaults + local `.env`. Bootstrap runs logos's own render + copy scripts and provisions the `OPENAI_API_KEY` secret; the copy step previously had no automation (done by hand).

## Per-repo config table

The single source of truth that prevents the `-E otel`-class bug (applying an extra a repo doesn't define):

| Repo | Python deps | Webapp | Notes |
|------|-------------|--------|-------|
| logos | `poetry install` (no extras) | — | foundry; consumed as `logos-foundry` git dep, local install is dev-only |
| sophia | `poetry install -E otel` | — | `ml` group behind `--with-ml` |
| hermes | `poetry install -E otel` | — | `ml` group behind `--with-ml` |
| talos | `poetry install` (no extras) | — | not in the apollo runtime stack; bootstrapped for completeness |
| apollo | `poetry install -E otel` | ✅ (2 vendored SDKs) | webapp + SDK build |

All five declare Python `>=3.12`, so 3.12 is compatible everywhere.

## Idempotency strategy

Every phase checks-then-acts and converges:
- Installers skip when the tool is already present (`command -v`).
- `uv python install 3.12` / `fnm install 20` are no-ops when satisfied.
- `poetry env use <3.12>` is stable → no new venvs on re-run.
- Clone skips existing dirs.
- `.env` is never overwritten.
- `compose pull` is a no-op when images are current.
- Vendored SDK build skips when `dist/` exists.

Re-running is safe and fast.

## Flags (minimal)

- `--with-ml` — include the heavy `ml` deps (torch etc.) for sophia/hermes. Off by default.
- `--skip-clone` — assume repos already present.
- `--help`.

## Error handling

- Strict mode (`set -euo pipefail`) throughout.
- Fail fast on the one unfixable prerequisite (Docker) with OS-specific guidance.
- Each phase logs what it did vs skipped.

## Testing

- `shellcheck` + `bash -n` (syntax/lint).
- **Idempotency check:** run twice; the second run must be a clean, fast no-op.
- **Cold-start proof:** run in a fresh `ubuntu` container to validate the Linux bootstrap end-to-end.

## Relationship to existing setup tooling

- Supersedes the divergent per-repo `scripts/setup-local-dev.sh` (5–36 lines, inconsistent, none pin/build/scaffold). They become thin shims that call back into the relevant bootstrap step, or are removed.
- `docs/LOCAL_SETUP.md` (human guide) updated to point at `bootstrap.sh` as the automated path.

## Follow-ups (separate from this script)

1. **Re-pin 3.13 → 3.12.** Earlier today the repos were pinned to uv 3.13; this design standardizes on 3.12. The pin must be corrected (and the memory entry superseded — via a new memory, not an edit).
2. **neo4j stale-pidfile guard in `run_apollo.sh`.** Since bootstrap is infra-prep-only, the auto-recreate-on-stale-pidfile fix belongs in `run_apollo.sh`'s infra `up` step, not here.
