# Package Publishing Guide

How to publish `logos-foundry` and propagate changes to downstream repos.

## Overview

`logos-foundry` is published as a container image to GitHub Container Registry (ghcr.io). Downstream repos (hermes, sophia, apollo, talos) consume it two ways:

1. **Python dependency** — Git tag reference in `pyproject.toml` (for `logos_config`, `logos_hcg`, etc.)
2. **Docker base image** — `FROM ghcr.io/c-daly/logos-foundry:<version>` in `Dockerfile`

Both references must point to the same version. CI enforces this when `check_foundry_alignment: true` is set.

**Registry:** `ghcr.io/c-daly/logos-foundry`

## What's Included

The foundry container bundles all logos packages:
- `logos_config`, `logos_hcg`, `logos_test_utils`, `logos_observability`
- `logos_perception`, `logos_persona`, `logos_sophia`, `logos_cwm_e`
- `logos_tools`, `planner_stub`
- All runtime and test dependencies (pytest, rdflib, neo4j, pymilvus, opentelemetry, etc.)
- Poetry for package management
- Python 3.12

## Release Checklist

### 1. Make changes on a feature branch

Never push directly to main. Infrastructure changes (Python version, base image) and library changes (logos_config fixes) should be separate PRs when possible.

```bash
git checkout -b feature/my-change
# ... make changes ...
git push -u origin feature/my-change
gh pr create
```

### 2. Merge PR and tag the release

```bash
# After PR is merged:
git checkout main && git pull

# Bump version
poetry version patch  # or minor/major

# Commit, tag, push
git add pyproject.toml
git commit -m "chore: bump version to $(poetry version -s)"
git tag "v$(poetry version -s)"
git push origin main --tags
```

This triggers the publish workflow, which builds and pushes:
- `ghcr.io/c-daly/logos-foundry:<version>`
- `ghcr.io/c-daly/logos-foundry:latest`

### 3. Verify the image published

- Check workflow: https://github.com/c-daly/logos/actions
- Check container: https://github.com/c-daly/logos/pkgs/container/logos-foundry

### 4. Bump downstream repos

Use the bump script to update all downstream repos at once:

```bash
./scripts/bump-downstream.sh vX.Y.Z
```

This creates a PR in each downstream repo (hermes, sophia, apollo, talos) that updates:
- `pyproject.toml` git tag reference
- `Dockerfile` FROM tag
- `poetry.lock` (regenerated)

Each PR body includes an auto-generated changelog showing what changed in logos between the previous tag and the new one, grouped by type (features, fixes, other). This gives reviewers context on what they're pulling in without digging through commits.

To also update CI workflow refs:

```bash
./scripts/bump-downstream.sh vX.Y.Z --ci-tag ci/v2
```

### 5. Review and merge downstream PRs

Review each PR. The changelog in the PR body tells you what changed. CI must pass before merging. When merged, the downstream repo's publish workflow will automatically build a new container image using the updated foundry base.

## CI Workflow Versioning

Downstream repos reference reusable CI workflows from logos:

```yaml
uses: c-daly/logos/.github/workflows/reusable-standard-ci.yml@ci/v1
```

These are pinned to a `ci/vN` tag, **not** `@main`. This prevents breaking workflow changes from immediately affecting downstream repos.

### Updating the CI tag

When you make backwards-compatible changes to the reusable workflows:

```bash
# Move the existing tag forward
git tag -f ci/v1
git push origin ci/v1 --force
```

When you make breaking changes:

```bash
# Create a new tag
git tag ci/v2
git push origin ci/v2
# Then update downstream repos (via bump script --ci-tag or manually)
```

## Version Alignment

Each downstream repo has two independent references to logos-foundry that must match:

| File | Reference |
|------|-----------|
| `pyproject.toml` | `logos-foundry = {git = "...", tag = "v0.4.2"}` |
| `Dockerfile` | `FROM ghcr.io/c-daly/logos-foundry:0.4.2` |

The `check_foundry_alignment` job in `reusable-standard-ci.yml` validates these match. Enable it in downstream CI:

```yaml
uses: c-daly/logos/.github/workflows/reusable-standard-ci.yml@ci/v1
with:
  check_foundry_alignment: true
```

## What NOT to Do

- **Don't push infrastructure changes directly to main.** Python version changes, base image changes, and dependency overhauls affect every downstream container. Always use a feature branch and PR.
- **Don't merge a downstream PR before all related fixes are ready.** If a foundry bump requires torch/spacy/dependency changes, commit those before merging.
- **Don't bump foundry version ad-hoc, repo by repo.** Use `scripts/bump-downstream.sh` to coordinate.
- **Don't mix unrelated changes in a version bump.** A config port fix and a Python version change should be separate releases.

## Versioning Strategy

Follow [Semantic Versioning](https://semver.org/):

- **PATCH** (0.4.x): Bug fixes, config changes, documentation
- **MINOR** (0.x.0): New packages, new utilities, backward-compatible features
- **MAJOR** (x.0.0): Breaking API changes, Python version bumps, removed packages

## Authentication

### Public access (read-only)

```bash
docker pull ghcr.io/c-daly/logos-foundry:0.4.2
```

### CI/CD (GitHub Actions)

Handled automatically via `secrets: inherit` in the reusable publish workflow.

### Local development

```bash
echo $GITHUB_TOKEN | docker login ghcr.io -u USERNAME --password-stdin
```

## Related

- `scripts/bump-downstream.sh` — Cross-repo bump automation
- `.github/workflows/reusable-standard-ci.yml` — Reusable CI workflow
- `.github/workflows/reusable-publish.yml` — Reusable publish workflow
- `AGENTS.md` — Ecosystem standards and port allocation
