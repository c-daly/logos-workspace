**OpenTelemetry Infrastructure Overview**

**Architecture:**
```
[Claude Code / LOGOS Services] → OTLP → [OTel Collector] → [Tempo (traces)]
                                                          → [Loki (logs)]
                                                          → [Prometheus (metrics)]
                                                          → [Grafana Cloud (optional)]
                                                          ↓
                                                     [Grafana]
```

**Stack (running from `~/.claude/infra/otel/`):**
- **OTel Collector** — receives OTLP traces/logs/metrics, fans out to Tempo, Loki, Prometheus
- **Tempo 2.3.1** — trace storage + metrics_generator (service graphs, span metrics)
- **Loki** — log storage (Claude Code events land here as structured streams)
- **Prometheus** — metrics storage, scrapes OTel collector; also receives remote_write from Tempo
- **Grafana** — dashboards, provisioned from `~/.claude/infra/otel/grafana/`

**Managing the Stack:**
```bash
cd ~/.claude/infra/otel
docker compose up -d        # start
docker compose down         # stop
docker compose restart grafana tempo prometheus   # restart specific services
docker compose up -d --force-recreate prometheus  # recreate (needed after command-line changes)
```

**Accessing UIs:**
- **Grafana:** http://localhost:3000
- **Prometheus:** http://localhost:9090
- **Tempo:** http://localhost:3200

**Ports (service → collector):**
- OTLP gRPC: `localhost:4317`
- OTLP HTTP: `localhost:4318`

**Dashboards (provisioned, auto-loaded):**

| UID | Title | Datasources |
|-----|-------|-------------|
| `claude-code-obs` | Claude Code Observability | Prometheus + Loki |
| `claude-code-obs-cloud` | Claude Code AI Intelligence (Cloud) | Prometheus + Loki |
| `agent-swarm-telemetry` | Agent Swarm Telemetry | Prometheus |
| `sophia-otel` | Sophia OTel Dashboard | Tempo |
| `hermes-otel` | Hermes OTel Dashboard | Tempo |
| `apollo-otel` | Apollo OTel Dashboard | Tempo |
| `logos-key-signals` | LOGOS Key Signals | Tempo |

**Prometheus Metrics Available:**
- `claude_code_cost_usage_USD_total` — cost per model/session
- `claude_code_token_usage_tokens_total` — tokens by type (input/output/cacheRead/cacheCreation)
- `claude_code_session_count_total` — sessions
- `claude_code_active_time_seconds_total` — active coding time
- `claude_code_lines_of_code_count_total` — lines added/removed
- `claude_code_code_edit_tool_decision_total` — edit tool accept/reject decisions
- `agent_swarm_*` — agent swarm orchestration metrics
- `tool_sequence_count`, `tool_transition_count` — tool usage patterns

**Loki Streams:**
- `{service_name="claude-code"}` — all Claude Code events
- Key label: `event_name` (`api_request`, `tool_result`, `tool_use`, `api_error`, etc.)
- Note: high cardinality — event fields (cost, tokens, duration) are stream labels

**Tempo Traces:**
- LOGOS services (sophia, hermes, apollo) send traces when running
- Tempo metrics_generator produces service graph / span metrics → Prometheus remote_write

**Troubleshooting:**
```bash
docker logs otel-collector --tail=20
docker logs tempo --tail=20
docker logs grafana --tail=20
docker logs prometheus --tail=20

# Health checks
curl http://localhost:3200/ready          # Tempo
curl http://localhost:9090/-/healthy      # Prometheus
curl http://localhost:3000/api/health     # Grafana
```

**Known Limitations:**
- `claude_code_commit_count_total` and `claude_code_pull_request_count_total` metrics are not currently exported by Claude Code — panels that reference them show 0 (use `or vector(0)`)
- LOGOS service dashboards (sophia/hermes/apollo) will show "No data" unless those services are running and instrumented
