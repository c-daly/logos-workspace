# CLAUDE.md — logos-workspace

## What This Is

Top-level workspace for the LOGOS cognitive architecture. Contains the 5 service repos as subdirectories, shared documentation, and cross-repo tooling.

## Ecosystem

| Repo | Purpose | Port |
|------|---------|------|
| **logos** | Foundry — contracts, ontology, shared `logos_*` packages | 37000 |
| **sophia** | Cognitive core — orchestrator, CWM, planner, ingestion, feedback | 47000 |
| **hermes** | Language services — STT, TTS, NLP, embeddings, LLM gateway | 17000 |
| **talos** | Hardware abstraction — sensors, actuators, simulation | 57000 |
| **apollo** | Client layer — React webapp, Python API, CLI | 27000 |

**Infrastructure** (shared, default ports):
- Neo4j: 7474 (HTTP), 7687 (Bolt)
- Milvus: 19530 (gRPC), 9091 (metrics)
- Redis: 6379

**Each sub-repo has its own CLAUDE.md.** Read it before working in that repo.

---

## Cross-Repo Workflow

### Contract changes flow downstream
```
logos (contracts, ontology, logos_* packages) → sophia, hermes, talos, apollo
```
Update contracts in logos first, then propagate.

### Shared config
All repos use `logos_config` for environment, ports, and settings:
```python
from logos_config import Neo4jConfig, MilvusConfig, RedisConfig, OtelConfig
from logos_config.ports import get_repo_ports
```

### Inter-service communication
- **Redis pub/sub** (`logos_events.EventBus`) for async events between services
- **REST APIs** for synchronous calls (contracts in `logos/contracts/`)
- Sophia publishes ontology events → Hermes subscribes for type sync

---

## Workspace-Level Tools

| Tool | Purpose |
|------|---------|
| `pull_all.sh` | Fetch + fast-forward all repos, sync wiki |
| `scripts/reconcile-issues.sh` | Audit issue/PR linkage across all 5 GitHub repos |

The project wiki is maintained in `logos` and mirrored here via `pull_all.sh`.

---

## Workspace-Level Docs

The `docs/` directory contains ecosystem-wide documentation:

| Doc | Purpose |
|-----|---------|
| `ARCHITECTURE.md` | System architecture and dependency graph |
| `VISION.md` | Project vision and major goals |
| `STATUS.md` | Current project status (keep updated) |
| `LOCAL_SETUP.md` | Local dev setup — Docker, Poetry, services |
| `COGNITIVE_LOOP.md` | Perception → reasoning → planning → action loop |
| `CI_CD.md` | CI/CD pipeline and GitHub Actions |
| `CODE_QUALITY.md` | Linting (ruff), formatting (black), type checking (mypy) |
| `TESTING.md` | Testing strategy — unit, integration, E2E |
| `OBSERVABILITY.md` | OpenTelemetry, Jaeger, Grafana |
| `PROJECT_TRACKING.md` | Issue tracking, epics, label conventions |
| `PACKAGE_PUBLISHING.md` | Publishing logos-foundry packages |
| `plans/` | Design docs and implementation specs |

---

## Agent Teams Over Individual Subagents

For non-trivial work, **prefer agent teams** (via `/orchestrate` or `TeamCreate`) over individual subagents. Teams provide coordinated task lists, parallel execution, and structured handoff.

Use individual subagents only for quick, isolated lookups.

### Subagent Types
- `Explore` — fast codebase exploration, file searches
- `general-purpose` — complex multi-step tasks, code changes
- `Plan` — architecture planning, implementation strategy

### When to Use Subagents
- Code exploration, cross-repo searches, research
- Large refactors, test execution, documentation generation

### When NOT to Use Subagents
- Quick single-file edits (do directly)
- Tasks requiring conversation history
- Tasks with <3 steps

---

## Workflow Patterns

### Before Coding
1. Explore relevant areas of codebase
2. Plan implementation for complex changes (3+ files)
3. Verify understanding with user if ambiguous

### During Implementation
- Track progress with task tools for multi-step work
- Run tests in background while continuing other work
- Spawn agents to investigate errors rather than guessing

### Code Review
- After significant changes, explore related code for regressions
- Check for similar patterns elsewhere that might need updates

---

## Paper Tracking

LOGOS has 13 candidate academic papers (see `LOGOS_Implementation_Spec.md` Appendix C). Paper logs live in the Obsidian vault at `10-projects/LOGOS/papers/`.

When implementation work produces results relevant to a paper, prompt the user: "This looks relevant to paper C.X — want to log it?"

---

## Common Commands

```bash
# Sync all repos
./pull_all.sh

# Lint & format (all repos use ruff + black + mypy)
poetry run ruff check --fix .
poetry run black .
poetry run mypy src/

# Run tests
poetry run pytest tests/

# Clear notebook outputs
jupyter nbconvert --clear-output --inplace NOTEBOOK.ipynb
```

---

## Code Standards (all repos)

- **Type hints**: required for public functions
- **Docstrings**: required for complex logic
- **Small functions**: prefer composable over monolithic
- **Backward compat**: maintain unless explicitly breaking
- **Security**: never log secrets/PII, sanitize inputs

See `logos/docs/TESTING_STANDARDS.md` and `logos/docs/GIT_PROJECT_STANDARDS.md` for full standards.
