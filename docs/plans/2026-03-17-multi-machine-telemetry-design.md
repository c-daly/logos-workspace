# Multi-Machine Telemetry Aggregation

**Date:** 2026-03-17
**Status:** Design
**Goal:** Unified telemetry view (Grafana, Obsidian, experiment data) across 3 machines

---

## Problem

Each machine runs its own local OTel stack (`~/.claude/infra/otel/`). Telemetry, experiment results, and knowledge artifacts are siloed per-machine. Code syncs fine via Git, but runtime data doesn't.

## Machine Roles

| Machine | Role | GPU | Runs |
|---------|------|-----|------|
| **Desktop** | Hub (central aggregator) | Yes | Experiments, training, full OTel stack, Grafana |
| **Machine B** | Spoke | No | Development, Claude Code sessions, lightweight testing |
| **Machine C** | Spoke | No | Development, Claude Code sessions, lightweight testing |

## Architecture

```
Machine B (spoke)                    Desktop (hub)                    Machine C (spoke)
┌──────────────┐                    ┌──────────────────┐              ┌──────────────┐
│ OTel Collector│──OTLP/HTTP───────→│ OTel Collector   │←──OTLP/HTTP─│ OTel Collector│
│ (local)       │                    │ (aggregating)    │              │ (local)       │
│               │                    │   ├→ Prometheus  │              │               │
│ Claude Code   │                    │   ├→ Tempo       │              │ Claude Code   │
│ LOGOS services│                    │   ├→ Loki        │              │ LOGOS services│
└──────────────┘                    │   └→ Grafana     │              └──────────────┘
                                    │                  │
       ┌────── Git (experiment journals, harness metrics) ──────┐
       │                            │                           │
  Obsidian vault ←── Git sync ──→ Obsidian vault ←── Git sync ──→ Obsidian vault
```

### Design Decisions

1. **Desktop is the hub** — it has the GPU, runs experiments, and is the most data-rich machine
2. **Spokes forward, hub aggregates** — spoke collectors forward all telemetry to hub via OTLP/HTTP
3. **Spokes also store locally** — if hub is unreachable, spokes buffer locally and still have local Grafana
4. **Machine identity via `host.name` resource attribute** — every trace/metric tagged with origin machine
5. **Experiment data stays in Git** — harness journals and metrics are already version-controlled
6. **Obsidian syncs via Git** — no special tooling needed; just commit + push/pull

---

## 1. OTel Collector: Hub vs Spoke Configs

### Spoke Config (Machine B / C)

The spoke collector receives local telemetry and forwards it to the hub. It also exports locally for offline use.

```yaml
# ~/.claude/infra/otel/otel-collector-config.spoke.yaml
receivers:
  otlp:
    protocols:
      grpc:
        endpoint: 0.0.0.0:4317
      http:
        endpoint: 0.0.0.0:4318

processors:
  batch:
    timeout: 5s
    send_batch_size: 512

  resource:
    attributes:
      - key: host.name
        value: "${MACHINE_NAME}"
        action: upsert
      - key: deployment.environment
        value: "dev"
        action: upsert

exporters:
  # Forward to hub
  otlphttp/hub:
    endpoint: "http://${HUB_HOST}:4318"
    retry_on_failure:
      enabled: true
      initial_interval: 5s
      max_interval: 30s
      max_elapsed_time: 300s

  # Local Prometheus (for offline access)
  prometheus:
    endpoint: 0.0.0.0:8889

  # Local debug logging
  debug:
    verbosity: basic

service:
  pipelines:
    traces:
      receivers: [otlp]
      processors: [resource, batch]
      exporters: [otlphttp/hub]
    metrics:
      receivers: [otlp]
      processors: [resource, batch]
      exporters: [otlphttp/hub, prometheus]
    logs:
      receivers: [otlp]
      processors: [resource, batch]
      exporters: [otlphttp/hub]
```

### Hub Config (Desktop)

The hub receives from spokes AND local services, then fans out to storage backends.

```yaml
# ~/.claude/infra/otel/otel-collector-config.hub.yaml
receivers:
  otlp:
    protocols:
      grpc:
        endpoint: 0.0.0.0:4317
      http:
        endpoint: 0.0.0.0:4318

processors:
  batch:
    timeout: 5s
    send_batch_size: 1024

  resource:
    attributes:
      - key: host.name
        value: "${MACHINE_NAME}"
        action: upsert
      - key: deployment.environment
        value: "dev"
        action: upsert

  # Only tag local data — forwarded data already has host.name
  resource/local_only:
    attributes:
      - key: host.name
        from_attribute: host.name
        action: upsert

exporters:
  otlphttp/tempo:
    endpoint: http://tempo:3200

  prometheusremotewrite:
    endpoint: http://prometheus:9090/api/v1/write

  loki:
    endpoint: http://loki:3100/loki/api/v1/push

  debug:
    verbosity: basic

service:
  pipelines:
    traces:
      receivers: [otlp]
      processors: [resource, batch]
      exporters: [otlphttp/tempo]
    metrics:
      receivers: [otlp]
      processors: [resource, batch]
      exporters: [prometheusremotewrite]
    logs:
      receivers: [otlp]
      processors: [resource, batch]
      exporters: [loki]
```

---

## 2. Machine Identity

Every machine sets `MACHINE_NAME` in its environment. This propagates as `host.name` on all telemetry.

```bash
# ~/.claude/infra/otel/.env
MACHINE_NAME=desktop    # or "laptop", "server", etc.
HUB_HOST=192.168.1.100  # desktop's LAN IP (spokes only)
```

Grafana dashboards filter/group by `host_name` label to show per-machine or aggregated views.

---

## 3. Network Considerations

| Scenario | Solution |
|----------|----------|
| Same LAN | Spokes point to desktop's LAN IP |
| Different networks | Tailscale/ZeroTier mesh VPN (recommended) |
| Hub offline | Spokes buffer locally via retry config; local Prometheus still works |
| Firewall | Hub needs port 4318 (OTLP HTTP) open to spoke IPs |

**Recommended: Tailscale** — gives each machine a stable 100.x.y.z IP. No port forwarding, works across networks, encrypted.

```bash
# After Tailscale setup, spokes use:
HUB_HOST=100.x.y.z  # desktop's Tailscale IP
```

---

## 4. Experiment Data (Already Solved)

The experiment harness stores journals and metrics in Git:

```
PoCs/logos-experiments/experiments/<name>/
├── journal/    # Append-only markdown entries
├── logs/       # Structured result.json per attempt
└── workspace/  # Code and artifacts
```

**Multi-machine workflow:**
1. Run experiment on desktop (GPU)
2. `git commit && git push`
3. Other machines `git pull` to see results
4. `harness-metrics <name>` and `harness-journal <name> summary` work anywhere

No changes needed — this already works.

---

## 5. Obsidian Vault Sync

Paper logs live in the Obsidian vault (`10-projects/LOGOS/papers/`). Options:

| Method | Pros | Cons |
|--------|------|------|
| **Git** | Free, version-controlled, merge-friendly for markdown | Manual commit/push/pull |
| **Obsidian Sync** | Real-time, conflict resolution, mobile | $4/mo |
| **Syncthing** | Free, real-time, P2P, no cloud | Needs LAN or Tailscale |

**Recommendation: Git + pre-commit hook** to auto-commit vault changes. This keeps everything version-controlled and works with the existing Git workflow.

```bash
# In Obsidian vault repo, add a sync script:
#!/bin/bash
cd ~/obsidian-vault
git add -A
git diff --cached --quiet || git commit -m "vault: auto-sync $(hostname) $(date +%Y-%m-%d-%H%M)"
git pull --rebase origin main
git push origin main
```

---

## 6. Grafana: Multi-Machine Dashboards

Add a `host_name` variable to dashboards for filtering:

```json
{
  "templating": {
    "list": [{
      "name": "host_name",
      "type": "query",
      "query": "label_values(host_name)",
      "includeAll": true,
      "multi": true
    }]
  }
}
```

Dashboard panels use `{host_name=~"$host_name"}` in queries to filter by machine.

Key dashboard additions for multi-machine:
- **Fleet Overview** — sessions, cost, tokens per machine
- **Experiment Tracker** — GPU utilization (desktop only), experiment status
- **Cross-Machine Diff** — compare Claude Code usage patterns across machines

---

## 7. Bootstrap Script

A single script to configure a machine as hub or spoke:

```bash
./scripts/setup-machine.sh hub desktop      # On the desktop
./scripts/setup-machine.sh spoke laptop 100.x.y.z   # On spoke, pointing to hub
```

This script:
1. Copies the appropriate collector config (hub vs spoke)
2. Sets `MACHINE_NAME` and `HUB_HOST` in `.env`
3. Starts/restarts the OTel Docker stack
4. Verifies connectivity (spoke → hub health check)

---

## Implementation Plan

| Step | What | Where |
|------|------|-------|
| 1 | Create hub/spoke collector configs | `~/.claude/infra/otel/` |
| 2 | Add `host.name` resource attribute to all configs | Collector configs |
| 3 | Create `setup-machine.sh` bootstrap script | `scripts/` |
| 4 | Update docker-compose to use config variants | `~/.claude/infra/otel/docker-compose.yml` |
| 5 | Add `host_name` variable to Grafana dashboards | Dashboard JSON |
| 6 | Set up Tailscale on all 3 machines | Manual (one-time) |
| 7 | Create Fleet Overview dashboard | Grafana provisioning |
| 8 | (Optional) Obsidian Git auto-sync hook | Vault repo |

---

## Open Questions

1. What are the 3 machines? (Desktop + laptop + server? Desktop + 2 laptops?)
2. Are they always on the same network, or do you work from different locations?
3. Is Tailscale acceptable, or do you need a different networking approach?
4. Do you want Obsidian Git auto-sync, or is manual sync fine?
5. Should the hub Grafana be accessible from spokes via browser (port 3000)?
