# Why LOGOS Works This Way

If you're looking at five repositories, Docker containers, port offset schemes, and wondering why — this page is for you.

## Why is it split into five repos?

Each repo has a distinct responsibility with a clean boundary:

- **logos** — the shared foundation. Contracts, ontology, configuration. Everything else depends on this, nothing depends on them.
- **sophia** — the brain. Reasons over graph structures. Has no idea what language is.
- **hermes** — the mouth and ears. Handles all language: speech, text, embeddings, LLM calls. Stateless utility.
- **talos** — the body. Abstracts hardware (or simulated hardware) behind a consistent interface.
- **apollo** — the face. Web UI and API gateway. The only thing users interact with directly.

This isn't microservices for the sake of microservices. LOGOS is a *non-linguistic* cognitive architecture — the core insight is that cognition (sophia) should be separated from language (hermes). They're different repos because they're fundamentally different concerns. Apollo, talos, and logos exist because UI, hardware, and shared contracts are also distinct concerns.

## Why not a monorepo?

Each repo has its own:
- CI pipeline and test suite
- Docker image published independently
- Release cadence (sophia changes faster than talos)
- Dependency footprint (hermes pulls ML libraries, talos pulls hardware SDKs, sophia pulls neither)

A monorepo would mean every CI run installs every dependency and runs every test. With separate repos, a change to hermes only rebuilds hermes.

The cost is coordination — when `logos_config` changes, you bump the version in every downstream repo. This is managed by git tags and `pyproject.toml` references. See [ARCHITECTURE.md](ARCHITECTURE.md) for the version flow.

## Why `logos_config`?

Every repo needs to know things like "what port does Neo4j run on?" and "where's the .env file?". Rather than hardcode these in five places, `logos_config` is a small Python package in the logos repo that all other repos install as a dependency.

It provides:
- Port allocation (`logos_config.ports`)
- Environment variable resolution (`logos_config.env`)
- Shared test utilities (`logos_test_utils`)

When a value changes, you change it once in `logos_config`, tag a new logos release, and bump the tag in downstream repos. This is better than finding and updating the same value in 15 files across 5 repos.

## Why are there two port schemes?

**Short answer:** so you can run multiple test stacks at the same time without port conflicts.

**Longer answer:**

In local development, there's one Neo4j and one Milvus, shared by all repos. They run on standard ports (7687, 19530).

In CI, each repo spins up its *own* Neo4j and Milvus in Docker containers. If they all used port 7687, they'd collide. So each repo maps to offset host ports:

| Repo | Neo4j Bolt (host) | Internal |
|------|-------------------|----------|
| hermes | 17687 | 7687 |
| apollo | 27687 | 7687 |
| logos | 37687 | 7687 |
| sophia | 47687 | 7687 |
| talos | 57687 | 7687 |

Inside the containers, everything still uses 7687. The offset is only on the host side, in `docker-compose` port mappings. Your code should never hardcode offset ports — use `logos_config` or environment variables.

## Why Docker for CI but not local dev?

Locally, you run services from source (`poetry run uvicorn ...`). This gives you hot reload, debugger access, and fast iteration.

In CI, you can't clone and install five repos to test one. Instead, sophia and hermes are pre-built Docker images pulled from the GitHub container registry. Apollo's CI pulls these images, starts them alongside Neo4j and Milvus, and runs tests against the running containers.

This means:
- Changing sophia code doesn't require re-running apollo's CI (it uses the published image)
- But publishing a new sophia image *does* affect apollo's next CI run
- If apollo's CI fails, check whether the service images are current

## Why Poetry and not pip/uv/conda?

Poetry handles:
- Dependency resolution with a lock file (reproducible installs)
- Virtual environment management
- Git-based dependencies (how we install `logos_config` from the logos repo)
- Extras groups (e.g., `poetry install -E otel` for optional OTel instrumentation)

The git dependency feature is critical — `pyproject.toml` references like `logos-foundry = {git = "...", tag = "vX.Y.Z"}` are how downstream repos pin to specific logos versions. Each repo also has a `Dockerfile` with `FROM ghcr.io/c-daly/logos-foundry:X.Y.Z` — both must match. See [CI_CD.md](CI_CD.md) for how this is enforced.

## Why Neo4j + Milvus?

LOGOS uses a Hybrid Cognitive Graph (HCG):
- **Neo4j** stores structured knowledge as a property graph — entities, relationships, causal links, plans
- **Milvus** stores vector embeddings for similarity search — perceptual features, semantic associations

This isn't a traditional database setup. The graph *is* the cognitive model. Sophia reasons by traversing and modifying the graph, not by querying a database.

## Why is the ontology in logos, not sophia?

The ontology (what kinds of nodes and relationships exist) is a *contract* shared across repos. Sophia creates nodes, hermes embeds them, apollo displays them. If the ontology lived in sophia, hermes and apollo would depend on sophia — creating a circular dependency.

By putting contracts in logos (the foundry), all repos can depend on logos without depending on each other.

## Common Confusions

**"I changed a port in logos_config but nothing happened"**
You need to tag a new logos release, bump the tag in downstream repos, and reinstall. See [ARCHITECTURE.md](ARCHITECTURE.md) version flow.

**"Why does my .env say 7687 but CI uses 27687?"**
Your .env is for local dev (shared infrastructure). CI uses offset ports for test isolation. Both are correct in their context.

**"Apollo depends on sophia and hermes — why doesn't it just import them?"**
Apollo talks to sophia and hermes over HTTP. They're separate processes (or containers). This is intentional — they have different scaling, deployment, and resource profiles. The SDKs in `logos/sdk/python/` provide typed HTTP clients.

**"Where are the docs for X?"**
If it's about the ecosystem (architecture, ports, setup), look in `logos/docs/`. If it's about a specific repo (how to contribute, directory structure), look in that repo's `AGENTS.md`.
