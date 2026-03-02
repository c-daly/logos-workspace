# Code Quality

Measures being taken to improve and maintain code quality across the LOGOS ecosystem.

## 1. Documentation Overhaul

The current documentation is scattered, stale, and often wrong (incorrect ports, phase-era references, duplicated content across repos). The `DOC_MANIFEST.md` in this directory defines the target state.

**What's changing:**
- Six new/rewritten ecosystem docs replace the existing sprawl (see manifest)
- Per-repo docs slimmed to README + AGENTS.md + CLAUDE.md + .env.example
- Phase references removed everywhere
- Research notes, plans, and scratch files archived to Obsidian
- Workflow files renamed to remove "phase" prefixes
- `.env.example` files fixed with correct ports and comments

**Why:** A developer should be able to set up and contribute within an hour. Wrong ports and stale instructions waste more time than no documentation at all.

## 2. Testing Strategy

### Philosophy

Tests run against real infrastructure by default. Unit-only runs are the exception, not the rule.

- **Default mode**: Start Neo4j + Milvus, run all tests (unit + integration). This is what CI does, and what local development should match.
- **Quick mode**: A `--no-infra` flag (or `pytest -m "not integration and not e2e"`) for fast iteration when you're editing code and want a quick check. This is a convenience, not the standard.
- **ML tests**: Excluded from default runs. Scheduled weekly in CI and available via `--ml` flag or label-triggered workflow dispatch.
- **Skipping tests because infrastructure isn't running is not acceptable.** If a test should run and infra isn't up, that's a setup problem to fix — not a reason to skip the test.

### Test Consolidation

The current test suite has too many tests that provide false confidence:

| Problem | Example | Fix |
|---------|---------|-----|
| Mock-heavy unit tests that test wiring, not behavior | Tests that mock the Neo4j driver and assert `driver.session().run()` was called with specific Cypher | Replace with integration tests that run real Cypher against Neo4j |
| Duplicate coverage across repos | HCG client tested in both logos and sophia with different mocks | Single integration test in sophia against real Neo4j |
| Trivial model tests | Tests that instantiate a Pydantic model and assert fields exist | Delete — Pydantic's own validation is the test |
| Tests that mirror implementation | Assertions that exactly match the code path rather than the business outcome | Rewrite to assert observable behavior |

**Target**: Reduce from ~100 test files to ~60, while increasing the mutation score (see below) and maintaining or improving line coverage.

### Mutation Testing

Line coverage measures what code was _executed_, not what was _verified_. A test that calls a function but doesn't assert anything meaningful gets 100% coverage with 0% value.

**Mutation testing** introduces small changes (mutations) to source code and checks whether tests catch them. Surviving mutants = tests that execute code without verifying it.

**Tool**: `mutmut` (pure Python, works with pytest, no config needed).

**Priority targets for mutation testing:**

| Module | Why |
|--------|-----|
| `sophia/hcg_client/client.py` | Complex Cypher generation, conditional logic in ON MATCH/CREATE, query_neighbors |
| `sophia/ingestion/proposal_processor.py` | Entity dedup, relation extraction edge processing, name-to-UUID resolution, experiment tracking |
| `hermes/relation_extractor.py` | Verb-to-relation mapping, entity matching, confidence scoring |
| `hermes/services/proposal_builder.py` | Context assembly for LLM — wrong field mappings are silent bugs |
| `logos_hcg/` | Shared HCG primitives used by every service |

**Usage:**
```bash
cd ~/projects/LOGOS/sophia
poetry run mutmut run --paths-to-mutate=src/sophia/hcg_client/client.py
poetry run mutmut results   # see surviving mutants
poetry run mutmut html      # browsable report
```

**CI integration**: Not gated initially. Run periodically as a quality signal, not a merge blocker.

### Test Infrastructure

**Local development:**
```bash
# Start shared infrastructure (one command)
cd ~/projects/LOGOS/logos
docker compose -f infra/docker-compose.hcg.dev.yml up -d

# Run all tests in a repo (default mode)
cd ~/projects/LOGOS/sophia
poetry run pytest tests/

# Quick check without infrastructure (convenience only)
poetry run pytest tests/ -m "not integration and not e2e"
```

**CI runs all tests by default.** Each repo's CI workflow starts a test stack (Neo4j + Milvus on repo-specific offset ports), installs dependencies, and runs the full suite. ML tests are excluded unless triggered by schedule, label, or manual dispatch.

## 3. Dependency Hygiene

Python dependency resolution across LOGOS repos has been a recurring source of CI failures. Key lessons:

### Version Pinning

| Package | Constraint | Reason |
|---------|-----------|--------|
| `numpy` | `>=1.24.0,<2.4` | Poetry resolves two versions (2.3.x and 2.4.x) when uncapped, due to pandas' marker-split dependency. Dual-version installs corrupt numpy's files on fresh CI installs (cache miss). |
| `python` | `>=3.12,<4.0` | System Python on macOS is 3.14 which breaks protobuf/pymilvus. Always use `/usr/local/bin/python3.12` for Poetry venvs. |
| `logos-foundry` | Tag pin (`tag = "v0.5.0"`) | Never use `branch = "main"` in production. Tag pins are reproducible and verifiable by the foundry alignment CI check. |

### Known Fragile Areas

- **numpy + pymilvus**: pymilvus imports numpy at module load time. Any numpy installation corruption causes immediate `SyntaxError` or `ImportError` — not at test time, but at collection time.
- **pandas marker resolution**: pandas 3.x has different numpy requirements for Python 3.12 vs 3.14. Poetry resolves both when `python = ">=3.12"`. Upper-bounding numpy prevents the split.
- **spacy + numba**: spacy pulls in numba which caps numpy at `<2.4`. When installing ML extras, this constraint naturally prevents the dual-version issue. Without ML extras, the cap must be explicit.
- **Poetry venv creation**: Don't run `poetry env use` in parallel across repos — causes `envs.toml` corruption.

### Rules

1. **Cap floating dependencies** that have known marker-split resolution issues.
2. **Regenerate lock files** after any pyproject.toml dependency change — don't hand-edit.
3. **Verify locally** (`poetry install && poetry run python -c "import numpy; import pandas; import pymilvus"`) before pushing lock changes.
4. **One numpy, one pandas** in the lock file. If you see two entries for either package without distinguishing markers, the resolution is broken.

## 4. CI Improvements

### Foundry Alignment Check

Every downstream repo (hermes, sophia, apollo, talos) has a CI job that verifies:
- The `Dockerfile` `FROM ghcr.io/c-daly/logos-foundry:<version>` tag
- The `pyproject.toml` `logos-foundry = {git = "...", tag = "v<version>"}` tag

These must match. Drift means the Python dependencies and the Docker base image are from different foundry versions.

### Reusable Workflow Pinning

Workflows reference `@ci/v1` tags, not `@main`. Breaking changes require a new tag (`ci/v2`) and controlled rollout.

### Cache-Resilient Installs

CI should produce identical results whether the Poetry venv cache hits or misses. The numpy dual-version fix (capping at `<2.4`) is an example of making installs cache-resilient. Any dependency resolution that works with cache but fails without it is a latent bug.

## 5. Code Review

### Automated Review

- **Greptile**: Runs on every PR. Catches surface-level issues (stale comments, obvious bugs).
- **Ruff + Black + mypy**: Enforced in CI. PRs that fail lint or type checks don't merge.

### What Reviewers Should Focus On

Automated tools catch syntax and style. Human review should focus on:

1. **Does the Cypher do what it claims?** — Most bugs in LOGOS have been in Cypher query construction (wrong MATCH patterns, missing ON MATCH clauses, incorrect relationship directions).
2. **Are new code paths tested with real infrastructure?** — If a PR adds a new query or endpoint, there should be an integration test, not just a mock-based unit test.
3. **Dependency changes** — Any change to pyproject.toml or poetry.lock should be scrutinized for version resolution issues.
4. **Port and config values** — Hardcoded ports are bugs. Everything should flow through `logos_config`.
