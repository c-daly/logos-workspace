# LOGOS Architecture

## System Overview

LOGOS is a non-linguistic cognitive architecture. It reasons with graph structures, not language. Language is handled by Hermes as an I/O utility.

```
User → Apollo → Sophia ↔ HCG (Neo4j + Milvus)
         (UI)    (Brain)    (Knowledge)
                    ↓
                Hermes (Language / Embeddings)
                    ↓
                Talos (Hardware / Simulation)
```

## Repositories

| Repo | Purpose | API Port | Runtime Role |
|------|---------|----------|-------------|
| **logos** | Foundry — contracts, ontology, SDKs, `logos_config` | 37000 | Library / Validation Service (port 37000 is for the SHACL validation service) |
| **sophia** | Cognitive core — Orchestrator, CWM, Planner | 47000 | Service |
| **hermes** | Language & embedding utility — STT, TTS, NLP, LLM | 17000 | Service |
| **talos** | Hardware abstraction — sensors, actuators, simulation | 57000 | Library / Service |
| **apollo** | Client — web UI, CLI, API gateway | 27000 | Service + Frontend |

## Dependency Graph

```
logos (foundry)
  ├── logos_config     — ports, env helpers, shared config
  ├── logos_test_utils — test fixtures, shared test infrastructure
  ├── sdk/python/sophia — Sophia SDK (used by apollo)
  └── sdk/python/hermes — Hermes SDK (used by apollo)

sophia  → depends on: logos (logos_config, logos_test_utils)
hermes  → depends on: logos (logos_config, logos_test_utils)
talos   → depends on: logos (logos_config, logos_test_utils)
apollo  → depends on: logos (logos_config, logos_test_utils, sophia SDK, hermes SDK)
          uses at runtime: sophia (API), hermes (API)
```

### How Dependencies Are Consumed

**In local development:**
- Services run from source: `poetry run uvicorn sophia.api.app:app --port 47000`
- `logos_config` is installed via `pyproject.toml` pointing to a git tag (e.g., `vX.Y.Z`)
- To use local logos changes before tagging, reinstall from source: `poetry run pip install --force-reinstall --no-deps ../logos/`
- Apollo calls sophia and hermes as HTTP services on localhost

**In CI:**
- Infrastructure (Neo4j, Milvus) and services (sophia, hermes) run as **Docker containers** via `docker-compose`
- Container images are published to `ghcr.io/c-daly/{repo}:latest` by each repo's publish workflow
- Tests run on the **host**, connecting to containers via mapped ports
- `logos_config` is baked into each container image at build time

**Version flow:**
```
1. Make changes in logos on a feature branch → PR → merge
2. Bump version + tag: poetry version patch && git tag vX.Y.Z && git push --tags
3. Bump downstream: ./scripts/bump-downstream.sh vX.Y.Z
   (updates pyproject.toml tag + Dockerfile FROM tag in each repo, opens PRs)
4. Merge downstream PRs → CI runs → container images rebuild automatically
```

Each downstream repo references logos-foundry in **two places** that must stay in sync:

| File | Reference | Example |
|------|-----------|---------|
| `pyproject.toml` | Git tag (Python packages) | `logos-foundry = {git = "...", tag = "vX.Y.Z"}` |
| `Dockerfile` | Image tag (Docker base) | `FROM ghcr.io/c-daly/logos-foundry:X.Y.Z` |

CI validates these match via the `check_foundry_alignment` job. See [CI_CD.md](CI_CD.md) for details.

## Infrastructure

All repos share the same infrastructure services:

| Service | Internal Port | Purpose |
|---------|--------------|---------|
| Neo4j | 7474 (HTTP), 7687 (Bolt) | Graph database for HCG |
| Milvus | 19530 (gRPC), 9091 (metrics) | Vector database for embeddings |

### Port Scheme: Shared vs Test Isolation

There are **two port schemes**. This is the single most important thing to understand:

#### Local Development (shared infrastructure)

All repos connect to the **same** Neo4j and Milvus instances on standard ports. `logos_config/ports.py` returns these shared ports for infrastructure services — only the API port varies per repo.

| Service | Port | Used By |
|---------|------|---------|
| Neo4j HTTP | 7474 | All repos |
| Neo4j Bolt | 7687 | All repos |
| Milvus gRPC | 19530 | All repos |
| Milvus Metrics | 9091 | All repos |

```
Neo4j Bolt:    bolt://localhost:7687
Milvus gRPC:   localhost:19530
```

#### CI Test Stack (docker-compose host mappings)

Each repo's test stack maps container ports to **repo-specific host ports** so multiple test stacks can run simultaneously without conflicts. These offset ports exist **only** in docker-compose port directives for CI isolation — they are not used in local development.

| Repo | Neo4j HTTP | Neo4j Bolt | Milvus gRPC | Milvus Metrics | MinIO |
|------|-----------|-----------|-------------|----------------|-------|
| hermes | 17474 | 17687 | 17530 | 17091 | 17900 |
| apollo | 27474 | 27687 | 27530 | 27091 | 27900 |
| logos | 37474 | 37687 | 37530 | 37091 | 37900 |
| sophia | 47474 | 47687 | 47530 | 47091 | 47900 |
| talos | 57474 | 57687 | 57530 | 57091 | 57900 |

**Inside containers**, services always use standard ports (7687, 19530). The offset only applies to the **host-side mapping** in `docker-compose` port directives (e.g., `27687:7687`).

### Single Source of Truth

Port allocation lives in `logos_config/ports.py`. Do not duplicate port values in documentation — they will drift. To check current values:

```bash
poetry run python -c "from logos_config.ports import APOLLO_PORTS; print(APOLLO_PORTS)"
```

> **Note:** `logos_config/README.md` currently shows offset ports without clarifying they are CI-only. This is tracked for cleanup in [DOC_MANIFEST.md](DOC_MANIFEST.md).

Environment variables override defaults: `NEO4J_URI`, `NEO4J_BOLT_PORT`, `MILVUS_PORT`, etc.

## Service APIs

**Sophia** (cognitive core, port 47000):
- `POST /plan` — generate action plan
- `POST /execute` — execute a plan
- `POST /simulate` — run simulation
- `POST /ingest` — ingest knowledge
- `POST /ingest/hermes_proposal` — ingest entities and relations from Hermes
- `GET /state` — current world model state
- `GET /health` — health check

**Hermes** (language utility, port 17000):
- `POST /stt` — speech to text
- `POST /tts` — text to speech
- `POST /embed` — generate embeddings
- `POST /llm` — language model inference (with cognitive loop: entity/relation extraction → Sophia context enrichment)
- `GET /health` — health check

**Apollo** (client API, port 27000):
- `GET /api/hcg/health` — health check
- Proxies to sophia and hermes

API contracts are defined in `logos/contracts/`.

## Running Locally

```bash
# 1. Start shared infrastructure
cd logos && docker compose -f infra/docker-compose.hcg.dev.yml up -d

# 2. Start services (each in its own terminal)
cd sophia && poetry run uvicorn sophia.api.app:app --host 0.0.0.0 --port 47000
cd hermes && poetry run uvicorn hermes.main:app --host 0.0.0.0 --port 17000

# 3. Start apollo
cd apollo && poetry run apollo-api   # API on 27000
cd apollo/webapp && npm run dev      # Frontend on 3000
```

Or use `apollo/scripts/run_apollo.sh` which orchestrates everything.

## Running Tests

```bash
# Unit tests (no infrastructure needed)
poetry run pytest tests/unit

# Integration tests (needs infrastructure)
# Option A: Use shared infrastructure (already running from local dev)
poetry run pytest tests/

# Option B: Use test stack (isolated, CI-style)
# Test stacks are located at:
#   infra/test_stack/docker-compose.test.yml  (generated test stack)
#   tests/e2e/stack/logos/docker-compose.test.yml  (E2E stack)
#   infra/{repo}/docker-compose.test.yml  (per-repo stacks)
docker compose -f infra/test_stack/docker-compose.test.yml up -d
# ... run tests ...
docker compose -f infra/test_stack/docker-compose.test.yml down -v
```

## Common Issues

**"Connection refused on port 27687"** — You have stale `logos_config`. The installed version uses test-offset ports instead of shared ports. Fix: `poetry run pip install --force-reinstall --no-deps ../logos/` or bump the tag in `pyproject.toml`.

**"Service X won't start in CI"** — Check that the Docker image was rebuilt after `logos_config` changes. The base image version in `Dockerfile` must match the `logos-foundry` tag in `pyproject.toml`.

