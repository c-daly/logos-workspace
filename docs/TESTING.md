# Testing

## Test Categories

| Category | Needs Infrastructure | Location | Run With |
|----------|---------------------|----------|----------|
| Unit | No | `tests/unit/` | `poetry run pytest tests/unit` |
| Integration | Yes (Neo4j, Milvus) | `tests/integration/` | `poetry run pytest tests/integration` |
| E2E | Yes (full stack) | `tests/e2e/` | `poetry run pytest tests/e2e` |

**Unit tests** should always pass without any external services. If a test needs Neo4j or Milvus, it belongs in `integration/` or `e2e/`.

## Running Tests Locally

```bash
# Just unit tests (fast, no setup needed)
poetry run pytest tests/unit

# All tests (needs infrastructure running)
poetry run pytest tests/

# With coverage
poetry run pytest tests/ --cov --cov-report=term
```

## Infrastructure for Integration Tests

Integration tests need Neo4j and Milvus running on shared ports:

```bash
cd ~/projects/LOGOS/logos
docker compose -f infra/docker-compose.hcg.dev.yml up -d
```

Tests discover these services via `logos_config` defaults (Neo4j on 7687, Milvus on 19530) or environment variables (`NEO4J_URI`, `MILVUS_PORT`, etc.).

## Test Stacks (CI-style Isolation)

Each repo has test stack compose files that spin up isolated infrastructure with repo-specific port offsets (see [ARCHITECTURE.md](ARCHITECTURE.md#port-scheme-shared-vs-test-isolation) for the port table). The actual paths are:

- `infra/test_stack/docker-compose.test.yml` — generated test stack
- `tests/e2e/stack/logos/docker-compose.test.yml` — E2E stack
- `infra/{repo}/docker-compose.test.yml` — per-repo stacks

```bash
# Start test stack (example using generated test stack)
docker compose -f infra/test_stack/docker-compose.test.yml up -d

# Run tests against it (set env vars for offset ports)
export NEO4J_URI=bolt://localhost:27687  # apollo example
poetry run pytest tests/

# Tear down
docker compose -f infra/test_stack/docker-compose.test.yml down -v
```

Some repos have overlay compose files (e.g., `docker-compose.test.apollo.yml`) that add service containers (sophia, hermes) on top of infrastructure.

## Writing Tests

### Conventions

- Use `pytest` with fixtures
- Mark tests that need infrastructure: `@pytest.mark.integration` or `@pytest.mark.e2e`
- Use `logos_test_utils` fixtures where available (shared across repos)
- Never hardcode ports — use `logos_config.ports` or environment variables

### Getting Service Configuration in Tests

```python
from logos_config.ports import get_repo_ports

ports = get_repo_ports("apollo")  # respects env var overrides
neo4j_uri = f"bolt://localhost:{ports.neo4j_bolt}"
```

Or use the repo-specific helpers:

```python
from apollo.env import get_neo4j_config
config = get_neo4j_config()  # returns {"uri": "...", "user": "...", "password": "..."}
```

## Coverage

All repos target minimum coverage thresholds enforced in CI:

```bash
poetry run pytest tests/ --cov=<package> --cov-fail-under=50
```

Check each repo's `pyproject.toml` for the exact threshold.

## Linting and Type Checking

Run before committing:

```bash
poetry run ruff check --fix src tests
poetry run ruff format src tests
poetry run mypy src
```

These are enforced in CI. A PR that fails lint or type checks will not pass.
