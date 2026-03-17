#!/usr/bin/env bash
# Setup a machine as hub (desktop/GPU) or spoke (laptop/other) for
# multi-machine LOGOS telemetry aggregation.
#
# Usage:
#   ./scripts/setup-machine.sh hub desktop
#   ./scripts/setup-machine.sh spoke laptop 100.64.0.1
#   ./scripts/setup-machine.sh spoke server desktop.tail1234.ts.net
#
# Hub: runs full OTel stack (collector + Prometheus + Tempo + Loki + Grafana)
# Spoke: runs lightweight collector that forwards to hub

set -euo pipefail

OTEL_DIR="${HOME}/.claude/infra/otel"

usage() {
    echo "Usage: $0 <hub|spoke> <machine-name> [hub-host]"
    echo ""
    echo "  hub   <name>             — Configure as central aggregator (desktop)"
    echo "  spoke <name> <hub-host>  — Configure as forwarder to hub"
    echo ""
    echo "Examples:"
    echo "  $0 hub desktop"
    echo "  $0 spoke laptop 100.64.0.1"
    echo "  $0 spoke server desktop.tail1234.ts.net"
    exit 1
}

[[ $# -lt 2 ]] && usage

ROLE="$1"
MACHINE_NAME="$2"
HUB_HOST="${3:-}"

if [[ "$ROLE" == "spoke" && -z "$HUB_HOST" ]]; then
    echo "Error: spoke mode requires hub-host argument"
    usage
fi

echo "=== LOGOS Machine Setup ==="
echo "Role:    $ROLE"
echo "Name:    $MACHINE_NAME"
[[ -n "$HUB_HOST" ]] && echo "Hub:     $HUB_HOST"
echo "OTel:    $OTEL_DIR"
echo ""

# Ensure OTel directory exists
mkdir -p "$OTEL_DIR"

# --- Write .env ---
cat > "$OTEL_DIR/.env" <<EOF
MACHINE_NAME=${MACHINE_NAME}
MACHINE_ROLE=${ROLE}
HUB_HOST=${HUB_HOST:-localhost}
EOF

echo "[✓] Wrote $OTEL_DIR/.env"

# --- Write collector config ---
if [[ "$ROLE" == "hub" ]]; then
    cat > "$OTEL_DIR/otel-collector-config.yaml" <<'COLLEOF'
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
        value: "${env:MACHINE_NAME}"
        action: upsert
      - key: deployment.environment
        value: "dev"
        action: upsert

exporters:
  otlphttp/tempo:
    endpoint: http://tempo:3200

  prometheusremotewrite:
    endpoint: http://prometheus:9090/api/v1/write
    resource_to_telemetry_conversion:
      enabled: true

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
COLLEOF
    echo "[✓] Wrote hub collector config"

else
    # Spoke config — forward to hub, also keep local prometheus
    cat > "$OTEL_DIR/otel-collector-config.yaml" <<'COLLEOF'
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
        value: "${env:MACHINE_NAME}"
        action: upsert
      - key: deployment.environment
        value: "dev"
        action: upsert

exporters:
  otlphttp/hub:
    endpoint: "http://${env:HUB_HOST}:4318"
    retry_on_failure:
      enabled: true
      initial_interval: 5s
      max_interval: 30s
      max_elapsed_time: 300s

  prometheus:
    endpoint: 0.0.0.0:8889

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
COLLEOF
    echo "[✓] Wrote spoke collector config"
fi

# --- Write docker-compose ---
if [[ "$ROLE" == "hub" ]]; then
    cat > "$OTEL_DIR/docker-compose.yml" <<'DCEOF'
# LOGOS OTel Stack — Hub (central aggregator)
# Receives telemetry from local services AND remote spoke machines.

services:
  otel-collector:
    image: otel/opentelemetry-collector-contrib:0.96.0
    container_name: otel-collector
    command: ["--config=/etc/otelcol/config.yaml"]
    volumes:
      - ./otel-collector-config.yaml:/etc/otelcol/config.yaml:ro
    ports:
      - "4317:4317"   # OTLP gRPC (local + remote)
      - "4318:4318"   # OTLP HTTP (local + remote)
      - "8889:8889"   # Prometheus metrics
    env_file: .env
    restart: unless-stopped
    depends_on:
      - tempo
      - prometheus
      - loki

  tempo:
    image: grafana/tempo:2.3.1
    container_name: tempo
    command: ["-config.file=/etc/tempo/config.yaml"]
    volumes:
      - ./tempo-config.yaml:/etc/tempo/config.yaml:ro
      - tempo-data:/var/tempo
    ports:
      - "3200:3200"   # Tempo query
      - "9095:9095"   # Tempo internal
    restart: unless-stopped

  prometheus:
    image: prom/prometheus:v2.50.1
    container_name: prometheus
    command:
      - "--config.file=/etc/prometheus/prometheus.yml"
      - "--storage.tsdb.path=/prometheus"
      - "--storage.tsdb.retention.time=30d"
      - "--web.enable-remote-write-receiver"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml:ro
      - prometheus-data:/prometheus
    ports:
      - "9090:9090"
    restart: unless-stopped

  loki:
    image: grafana/loki:2.9.4
    container_name: loki
    command: ["-config.file=/etc/loki/config.yaml"]
    volumes:
      - ./loki-config.yaml:/etc/loki/config.yaml:ro
      - loki-data:/loki
    ports:
      - "3100:3100"
    restart: unless-stopped

  grafana:
    image: grafana/grafana:10.3.3
    container_name: grafana
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=logos
      - GF_USERS_ALLOW_SIGN_UP=false
    volumes:
      - ./grafana/provisioning:/etc/grafana/provisioning:ro
      - grafana-data:/var/lib/grafana
    ports:
      - "3000:3000"
    restart: unless-stopped
    depends_on:
      - prometheus
      - tempo
      - loki

volumes:
  tempo-data:
  prometheus-data:
  loki-data:
  grafana-data:
DCEOF
    echo "[✓] Wrote hub docker-compose.yml"

    # --- Prometheus config ---
    cat > "$OTEL_DIR/prometheus.yml" <<'PROMEOF'
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: "otel-collector"
    static_configs:
      - targets: ["otel-collector:8889"]

  - job_name: "prometheus"
    static_configs:
      - targets: ["localhost:9090"]
PROMEOF
    echo "[✓] Wrote prometheus.yml"

    # --- Tempo config ---
    cat > "$OTEL_DIR/tempo-config.yaml" <<'TEMPOEOF'
server:
  http_listen_port: 3200

distributor:
  receivers:
    otlp:
      protocols:
        http:
          endpoint: 0.0.0.0:3200
        grpc:
          endpoint: 0.0.0.0:9095

storage:
  trace:
    backend: local
    local:
      path: /var/tempo/traces
    wal:
      path: /var/tempo/wal

metrics_generator:
  registry:
    external_labels:
      source: tempo
  storage:
    path: /var/tempo/generator/wal
    remote_write:
      - url: http://prometheus:9090/api/v1/write
        send_exemplars: true
  processor:
    service_graphs:
      dimensions: [host.name]
    span_metrics:
      dimensions: [host.name]
TEMPOEOF
    echo "[✓] Wrote tempo-config.yaml"

    # --- Loki config ---
    cat > "$OTEL_DIR/loki-config.yaml" <<'LOKIEOF'
auth_enabled: false

server:
  http_listen_port: 3100

common:
  path_prefix: /loki
  storage:
    filesystem:
      chunks_directory: /loki/chunks
      rules_directory: /loki/rules
  replication_factor: 1
  ring:
    kvstore:
      store: inmemory

schema_config:
  configs:
    - from: 2020-10-24
      store: tsdb
      object_store: filesystem
      schema: v13
      index:
        prefix: index_
        period: 24h

limits_config:
  allow_structured_metadata: true
  volume_enabled: true
LOKIEOF
    echo "[✓] Wrote loki-config.yaml"

    # --- Grafana provisioning ---
    mkdir -p "$OTEL_DIR/grafana/provisioning/datasources"
    mkdir -p "$OTEL_DIR/grafana/provisioning/dashboards"

    cat > "$OTEL_DIR/grafana/provisioning/datasources/datasources.yaml" <<'DSEOF'
apiVersion: 1

datasources:
  - name: Prometheus
    type: prometheus
    access: proxy
    url: http://prometheus:9090
    isDefault: true
    editable: false

  - name: Tempo
    type: tempo
    access: proxy
    url: http://tempo:3200
    editable: false

  - name: Loki
    type: loki
    access: proxy
    url: http://loki:3100
    editable: false
DSEOF
    echo "[✓] Wrote Grafana datasource provisioning"

    cat > "$OTEL_DIR/grafana/provisioning/dashboards/dashboards.yaml" <<'DBEOF'
apiVersion: 1

providers:
  - name: LOGOS
    orgId: 1
    folder: LOGOS
    type: file
    disableDeletion: false
    updateIntervalSeconds: 30
    options:
      path: /etc/grafana/provisioning/dashboards/json
      foldersFromFilesStructure: false
DBEOF

    mkdir -p "$OTEL_DIR/grafana/provisioning/dashboards/json"
    echo "[✓] Wrote Grafana dashboard provisioning"

else
    # Spoke: minimal docker-compose — just the collector
    cat > "$OTEL_DIR/docker-compose.yml" <<'DCEOF'
# LOGOS OTel Stack — Spoke (forwarder)
# Forwards all telemetry to the hub machine.

services:
  otel-collector:
    image: otel/opentelemetry-collector-contrib:0.96.0
    container_name: otel-collector
    command: ["--config=/etc/otelcol/config.yaml"]
    volumes:
      - ./otel-collector-config.yaml:/etc/otelcol/config.yaml:ro
    ports:
      - "4317:4317"   # OTLP gRPC
      - "4318:4318"   # OTLP HTTP
      - "8889:8889"   # Local Prometheus metrics
    env_file: .env
    restart: unless-stopped
DCEOF
    echo "[✓] Wrote spoke docker-compose.yml"
fi

echo ""
echo "=== Setup Complete ==="
echo ""
echo "Next steps:"
if [[ "$ROLE" == "hub" ]]; then
    echo "  1. cd $OTEL_DIR && docker compose up -d"
    echo "  2. Open Grafana at http://localhost:3000 (admin/logos)"
    echo "  3. Ensure port 4318 is accessible from spoke machines"
    echo "     (If using Tailscale, this works automatically)"
else
    echo "  1. Ensure hub ($HUB_HOST) is running and port 4318 is reachable:"
    echo "     curl -sf http://$HUB_HOST:4318/v1/traces && echo OK"
    echo "  2. cd $OTEL_DIR && docker compose up -d"
    echo "  3. Verify forwarding: docker logs otel-collector --tail=5"
fi
echo ""
echo "To check machine identity:"
echo "  cat $OTEL_DIR/.env"
