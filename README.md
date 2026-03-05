# LOGOS

A non-linguistic cognitive architecture built on the principle that language is better suited as an interface than a substrate for thought.

LOGOS reasons over a hierarchical knowledge graph rather than text, treating language as an I/O layer. The cognitive core thinks in typed nodes, semantic edges, and causal relationships. Hermes provides language services both as the human interface and as a utility for the cognitive core, generating queries, interpreting results, and mediating between graph-native reasoning and the text-based world.

## Architecture

```
          User → Apollo → Sophia ↔ HCG (Neo4j + Milvus)
                  (UI)    (Brain)    (Knowledge)
                             ↕
                         Hermes (Language / Embeddings)
                             ↕
                         Talos (Hardware / Simulation)

                     Redis (Event Bus)
                  ┌───────┼───────────┐
               Sophia   Hermes     Talos
               pub/sub for ontology sync,
               feedback loops, sensor events
```

## Repositories

| Repo | Purpose | Port |
|------|---------|------|
| [logos](https://github.com/c-daly/logos) | Foundry: contracts, ontology, shared packages, `logos_config` | 37000 |
| [sophia](https://github.com/c-daly/sophia) | Cognitive core: orchestrator, working memory, planner | 47000 |
| [hermes](https://github.com/c-daly/hermes) | Language services: STT, TTS, NLP, embeddings, LLM gateway | 17000 |
| [talos](https://github.com/c-daly/talos) | Hardware abstraction: sensors, actuators, simulation | 57000 |
| [apollo](https://github.com/c-daly/apollo) | Client: React web UI, Python API, CLI | 27000 |

Each repo has its own CI pipeline, Docker images, and release cadence. The `logos` foundry publishes shared packages that downstream services consume via git tags.

## Infrastructure

| Service | Port(s) |
|---------|---------|
| Neo4j | 7474 (HTTP), 7687 (Bolt) |
| Milvus | 19530 (gRPC), 9091 (metrics) |
| Redis | 6379 |

## Key Concepts

**Hybrid Cognitive Graph (HCG)**: Neo4j stores typed nodes with IS_A inheritance, REQUIRES/CAUSES edges for planning, and semantic relationships. Milvus stores vector embeddings for similarity search. Together they form the knowledge substrate.

**Cognitive Loop**: User input flows through Hermes (entity extraction, embeddings) to Sophia (proposal processing, graph reasoning, LLM orchestration) and back as a response. Redis pub/sub carries events between services.

**Ontology**: A formal type system with inheritance, constraints, and validation. Defined in the logos foundry and distributed to downstream services via event-driven synchronization.

**Working Memory**: Three concurrent representations:
- **CWM-G** (Grounded): perceptual state via JEPA models
- **CWM-A** (Abstract): conceptual/symbolic state from the knowledge graph
- **CWM-E** (Emotional): affective state influencing reasoning

## Getting Started

### Prerequisites

- Python 3.12
- Poetry
- Node.js 18+
- Docker

### Quick Start

```bash
# Clone the workspace and all repos
mkdir -p ~/projects/LOGOS && cd ~/projects/LOGOS
git clone git@github.com:c-daly/logos-workspace.git .
git clone git@github.com:c-daly/logos.git
git clone git@github.com:c-daly/sophia.git
git clone git@github.com:c-daly/hermes.git
git clone git@github.com:c-daly/talos.git
git clone git@github.com:c-daly/apollo.git

# Start infrastructure
cd logos && docker compose -f infra/docker-compose.hcg.dev.yml up -d

# Install dependencies (order matters: logos first)
cd ../logos && poetry install
cd ../sophia && poetry install
cd ../hermes && poetry install
cd ../talos && poetry install
cd ../apollo && poetry install

# Start services (each in its own terminal)
cd sophia && poetry run uvicorn sophia.api.app:app --host 0.0.0.0 --port 47000
cd hermes && poetry run uvicorn hermes.main:app --host 0.0.0.0 --port 17000
cd apollo && poetry run apollo-api
```

See [LOCAL_SETUP.md](docs/LOCAL_SETUP.md) for detailed setup instructions.

## Workspace Tools

| Script | Purpose |
|--------|---------|
| `pull_all.sh` | Fetch and fast-forward all repos, sync wiki |
| `scripts/reconcile-issues.sh` | Audit issue/PR linkage across all repos |
| `scripts/reinstall-foundry.sh` | Reinstall logos-foundry packages in all downstream repos |

## Documentation

| Doc | Purpose |
|-----|---------|
| [VISION.md](docs/VISION.md) | Project vision, goals, and non-goals |
| [ARCHITECTURE.md](docs/ARCHITECTURE.md) | System architecture, dependency graph, port management |
| [WHY.md](docs/WHY.md) | Why the multi-repo structure exists |
| [COGNITIVE_LOOP.md](docs/COGNITIVE_LOOP.md) | End-to-end perception to reasoning to action flow |
| [STATUS.md](docs/STATUS.md) | Current project status |
| [LOCAL_SETUP.md](docs/LOCAL_SETUP.md) | Local development environment setup |
| [OBSERVABILITY.md](docs/OBSERVABILITY.md) | OpenTelemetry, Jaeger, Grafana |
| [CI_CD.md](docs/CI_CD.md) | CI/CD pipelines and GitHub Actions |
| [TESTING.md](docs/TESTING.md) | Testing strategy: unit, integration, E2E |
| [CODE_QUALITY.md](docs/CODE_QUALITY.md) | Linting (ruff), formatting (black), type checking (mypy) |
| [PROJECT_TRACKING.md](docs/PROJECT_TRACKING.md) | Issue tracking, epics, label conventions |
| [PACKAGE_PUBLISHING.md](docs/PACKAGE_PUBLISHING.md) | Publishing logos-foundry packages |

Design docs and implementation plans live in [docs/plans/](docs/plans/) and [docs/designs/](docs/designs/).

## Additional Directories

| Directory | Purpose |
|-----------|---------|
| `PoCs/` | Proof-of-concept experiments (JEPA, TinyMind) |
| `adversarial-tests/` | Three-agent adversarial test suite |
| `wiki/` | Project wiki (synced from logos repo) |

## Code Quality

All repos share the same quality toolchain:

```bash
poetry run ruff check --fix .    # Lint
poetry run black .               # Format
poetry run mypy src/             # Type check
poetry run pytest tests/         # Test
```
