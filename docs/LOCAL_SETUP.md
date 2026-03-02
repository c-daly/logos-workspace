# Local Development Setup

This guide takes you from a fresh machine to a running LOGOS stack.

## Prerequisites

Install these before starting:

| Tool | Version | Install |
|------|---------|---------|
| Python | 3.12 | `brew install python@3.12` |
| Poetry | latest | `brew install poetry` |
| Node.js | 18+ | `brew install node@18` |
| Docker | latest | [Docker Desktop for Mac](https://www.docker.com/products/docker-desktop/) |
| Git | latest | `brew install git` |
| gh | latest | `brew install gh` (for PR workflows and bump script) |

> **Python version matters.** The ecosystem requires Python 3.12. System default `python3` may be 3.14, which breaks `protobuf` and `pymilvus` (metaclass errors). Always use 3.12 explicitly.

Verify:
```bash
python3.12 --version   # Must be 3.12.x
poetry --version
node --version          # 18+
docker --version
```

## Clone Repositories

All 5 repos must live side by side in the same parent directory:

```bash
mkdir -p ~/projects/LOGOS && cd ~/projects/LOGOS
git clone git@github.com:c-daly/logos.git
git clone git@github.com:c-daly/sophia.git
git clone git@github.com:c-daly/hermes.git
git clone git@github.com:c-daly/talos.git
git clone git@github.com:c-daly/apollo.git
```

Expected layout:
```
~/projects/LOGOS/
  logos/
  sophia/
  hermes/
  talos/
  apollo/
```

## Start Infrastructure

Neo4j and Milvus run as shared Docker containers. All repos connect to the same instances.

```bash
cd ~/projects/LOGOS/logos
docker compose -f infra/docker-compose.hcg.dev.yml up -d
```

Verify infrastructure is healthy:
```bash
# Neo4j (may take 30s on first start)
curl -s http://localhost:7474 | head -1

# Milvus
curl -s http://localhost:9091/healthz
```

## Install Python Dependencies

Install logos first (it provides `logos_config` used by all other repos), then the rest. Order doesn't matter after logos.

> **Important:** If your system `python3` is not 3.12, you must tell Poetry which Python to use **before** installing. Run this once per repo, sequentially (not in parallel — concurrent `poetry env use` corrupts Poetry's environment cache):
> ```bash
> cd ~/projects/LOGOS/logos && poetry env use /usr/local/bin/python3.12
> cd ~/projects/LOGOS/sophia && poetry env use /usr/local/bin/python3.12
> cd ~/projects/LOGOS/hermes && poetry env use /usr/local/bin/python3.12
> cd ~/projects/LOGOS/talos && poetry env use /usr/local/bin/python3.12
> cd ~/projects/LOGOS/apollo && poetry env use /usr/local/bin/python3.12
> ```

```bash
# Logos (foundry — must be first)
cd ~/projects/LOGOS/logos && poetry install

# Sophia
cd ~/projects/LOGOS/sophia && poetry install

# Hermes
cd ~/projects/LOGOS/hermes && poetry install

# Talos
cd ~/projects/LOGOS/talos && poetry install

# Apollo (Python backend + Node frontend)
cd ~/projects/LOGOS/apollo && poetry install -E otel
cd ~/projects/LOGOS/apollo/webapp && npm ci
```

## Configure Environment

Copy example env files for each repo that has them:

```bash
# Apollo
cp ~/projects/LOGOS/apollo/.env.example ~/projects/LOGOS/apollo/.env
cp ~/projects/LOGOS/apollo/webapp/.env.example ~/projects/LOGOS/apollo/webapp/.env

# Sophia (if .env.example exists)
cp ~/projects/LOGOS/sophia/.env.example ~/projects/LOGOS/sophia/.env 2>/dev/null || true

# Hermes (if .env.example exists)
cp ~/projects/LOGOS/hermes/.env.example ~/projects/LOGOS/hermes/.env 2>/dev/null || true
```

The defaults connect to shared infrastructure on standard ports (Neo4j 7687, Milvus 19530). You shouldn't need to change anything for local dev. Check each repo's `.env.example` for repo-specific settings (e.g., model paths for Hermes, cognitive core config for Sophia).

## Start Services

Services must start in dependency order: Sophia first (it owns the cognitive core), then Hermes, then Apollo.

**Option A: Manual (each in its own terminal)**

```bash
# Terminal 1: Sophia
cd ~/projects/LOGOS/sophia
poetry run uvicorn sophia.api.app:app --host 0.0.0.0 --port 47000

# Terminal 2: Hermes
cd ~/projects/LOGOS/hermes
poetry run uvicorn hermes.main:app --host 0.0.0.0 --port 17000

# Terminal 3: Apollo API
cd ~/projects/LOGOS/apollo
poetry run apollo-api

# Terminal 4: Apollo Frontend
cd ~/projects/LOGOS/apollo/webapp
npm run dev
```

**Option B: Orchestrated**

```bash
cd ~/projects/LOGOS/apollo/scripts
./run_apollo.sh
```

This starts everything in the right order with log files at `/tmp/*.log`.

## Verify

Check each service is running:

```bash
# Sophia
curl -s http://localhost:47000/health | python3 -m json.tool

# Hermes
curl -s http://localhost:17000/health | python3 -m json.tool

# Apollo API
curl -s http://localhost:27000/api/hcg/health | python3 -m json.tool
```

Open the Apollo web UI: http://localhost:3000

## Run Tests

```bash
# Unit tests (no infrastructure needed)
cd ~/projects/LOGOS/sophia && poetry run pytest tests/unit
cd ~/projects/LOGOS/hermes && poetry run pytest tests/unit
cd ~/projects/LOGOS/apollo && poetry run pytest tests/unit

# Integration tests (needs infrastructure running)
cd ~/projects/LOGOS/apollo && poetry run pytest tests/
```

## Troubleshooting

**`poetry install` fails with git dependency errors:**
Your SSH key may not be configured for GitHub. Verify: `ssh -T git@github.com`

**"Connection refused" on port 7687 or 19530:**
Infrastructure isn't running. Start it: `cd ~/projects/LOGOS/logos && docker compose -f infra/docker-compose.hcg.dev.yml up -d`

**"Connection refused" on port 27687 (note the 2):**
You have a stale `logos_config` installed. The offset ports (27xxx) are for CI test isolation, not local dev. Fix:
```bash
cd ~/projects/LOGOS/<repo> && poetry run pip install --force-reinstall --no-deps ../logos/
```

**Sophia/Hermes won't start — "address already in use":**
A previous instance is still running. Kill it: `lsof -ti:47000 | xargs kill` (adjust port as needed).

**Neo4j container fails to start:**
Docker Desktop may need more memory. Allocate at least 4GB in Docker Desktop settings.

**Apollo frontend shows connection errors:**
Make sure the backend services (Sophia, Hermes, Apollo API) are running first. The frontend expects them on the ports configured in `webapp/.env`.
