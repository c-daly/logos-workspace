# CI/CD

## Overview

Every LOGOS repo uses GitHub Actions with two types of workflows:

1. **CI** — lint, type check, test (runs on every push/PR to `main`)
2. **Publish** — build and push Docker image to `ghcr.io` (runs on push to `main`)

## Reusable Workflows

Most repos delegate CI to shared workflows in the logos repo:

```yaml
uses: c-daly/logos/.github/workflows/reusable-standard-ci.yml@ci/v1
uses: c-daly/logos/.github/workflows/reusable-publish.yml@ci/v1
```

These are pinned to `ci/vN` tags, **not** `@main`. This prevents breaking workflow changes from immediately affecting all repos.

### `reusable-standard-ci.yml`

Provides:
- Python setup (Poetry install, virtualenv caching)
- Ruff lint + Black format check
- mypy type checking
- pytest execution
- Optional Docker Compose for test infrastructure
- Optional Node.js lint/test/build
- Coverage upload to Codecov
- **Foundry version alignment check** (validates Dockerfile FROM tag matches pyproject.toml tag)

Repos customize behavior through `with:` inputs (Python versions, lint paths, compose files, etc.).

### `reusable-publish.yml`

Builds and pushes Docker images to `ghcr.io`. Checks out the **calling repo** (not logos), extracts the version from the calling repo's `pyproject.toml`, builds its `Dockerfile`, and tags the image.

### Workflow Versioning

| Change type | Action |
|-------------|--------|
| Backwards-compatible | Move the existing `ci/v1` tag forward: `git tag -f ci/v1 && git push origin ci/v1 --force` |
| Breaking | Create `ci/v2`, update downstream repos one at a time |

The `scripts/bump-downstream.sh --ci-tag ci/v2` flag automates updating workflow refs across repos.

## How CI Works for Each Repo

### Simple repos (sophia, hermes, talos)

```
push to main
  → checkout
  → poetry install
  → ruff + black + mypy
  → pytest (unit tests only, or with test stack)
  → upload coverage
```

### Apollo (most complex)

```
push to main
  → checkout
  → docker compose up (Neo4j, Milvus, Sophia container, Hermes container, Apollo container)
  → wait for all services healthy
  → poetry install
  → ruff + black + mypy
  → pytest (all tests, including integration)
  → npm ci + lint + type-check + test + build (webapp)
  → Playwright E2E tests
```

Apollo's CI depends on published Docker images for sophia and hermes. If those images are broken, apollo CI fails even if apollo's code is fine.

## Container Publishing

Each repo with a service has a `publish.yml` workflow:

```
push to main
  → build Docker image
  → push to ghcr.io/c-daly/<repo>:latest
  → tag with commit SHA
```

Images are published to:
- `ghcr.io/c-daly/sophia:latest`
- `ghcr.io/c-daly/hermes:latest`
- `ghcr.io/c-daly/apollo:latest`

### Hermes ML Image

Hermes has two image variants:
- Standard: `ghcr.io/c-daly/hermes:latest`
- ML (with whisper, TTS, spaCy): `ghcr.io/c-daly/hermes:ml-latest`

The ML build takes longer due to large model dependencies.

## Version Propagation

When you change `logos_config` or any foundry package:

```
1. Make changes on a feature branch, open a PR, merge
2. Bump version: poetry version patch
3. Tag: git tag vX.Y.Z && git push origin vX.Y.Z --tags
4. Run bump script: ./scripts/bump-downstream.sh vX.Y.Z
   (creates PRs in hermes, sophia, apollo, talos)
5. Review and merge each downstream PR
6. Each merge triggers: CI → tests pass → publish workflow → new Docker image
```

The bump script updates **both** references in each downstream repo:
- `pyproject.toml` git tag (Python dependency)
- `Dockerfile` FROM tag (Docker base image)

Each downstream PR body includes an auto-generated changelog (grouped by `feat:`, `fix:`, other) showing what changed in logos between the previous and new tag.

These must always match. The `check_foundry_alignment` CI job enforces this — if they drift, CI fails with a clear error message. Enable it per-repo:

```yaml
check_foundry_alignment: true
```

### What NOT to do

- Don't push foundry changes directly to main — use a PR
- Don't bump downstream repos manually — use the script
- Don't merge a downstream PR before verifying all related fixes are ready (e.g., torch compatibility)
- Don't mix infrastructure changes (Python version) with library changes in the same release

See `docs/operations/PACKAGE_PUBLISHING.md` for the full release checklist.

## Debugging CI Failures

### "Start test services" fails

The Docker Compose step failed. Common causes:
- A service container image is broken (check if hermes/sophia published recently)
- Port conflict with another running test stack
- Docker pull rate limiting

Check logs:
```bash
gh run view <run-id> --job <job-id> --log
```

### "Wait for services" times out

Infrastructure started but a service never became healthy.
```bash
# Check which service timed out (look at the last "Waiting for..." line)
gh run view <run-id> --log | grep "Waiting for"
```

Common causes:
- Service crashes on startup (stale `logos_config` in the image — check Dockerfile base image version)
- Neo4j or Milvus takes too long to initialize (increase timeout)
- Service can't connect to its dependencies (wrong port in compose env vars)

### Tests fail but pass locally

- Local dev uses shared ports (7687); CI uses offset ports (27687)
- CI installs from `poetry.lock` which may pin a different `logos_config` than your local env
- CI runs all tests including integration; you may only run unit tests locally

### Container publish fails

- Check that the Dockerfile builds locally: `docker build -t test .`
- Verify the Dockerfile `FROM` tag matches what's published — the tag in `Dockerfile` and the tag in `pyproject.toml` must be the same version
- Check GHCR authentication (repo secrets)
- For ML images: verify that ML dependencies (torch, spacy, etc.) have wheels for the Python version in the foundry base image

## Useful Commands

```bash
# List recent CI runs
gh run list --limit 10

# View a specific run
gh run view <run-id>

# Get failed step logs
gh run view <run-id> --log-failed

# Re-run a failed workflow
gh run rerun <run-id>

# Check container image details
docker pull ghcr.io/c-daly/hermes:latest
docker inspect ghcr.io/c-daly/hermes:latest | grep -i version
```
