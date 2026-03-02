# OpenTelemetry Infrastructure

## Architecture

```
[Sophia/Hermes/Apollo] → OTLP → [OTel Collector] → [Jaeger + Prometheus]
                                                        ↓
                                                   [Grafana]
```

## Components

### OTel Collector
- Receives traces via OTLP (gRPC on 4317, HTTP on 4318)
- Processes traces with batching and resource attribution
- Exports to Jaeger (traces) and Prometheus (metrics)

### Jaeger
- Stores and visualizes distributed traces
- UI available at http://localhost:16686
- Supports trace search by service, operation, tags, duration

### Prometheus
- Stores metrics scraped from OTel collector
- UI available at http://localhost:9090
- Data source for Grafana dashboards

## Starting the Stack

```bash
cd logos
./scripts/start-otel-stack.sh
```

## Stopping the Stack

```bash
cd logos
./scripts/stop-otel-stack.sh
```

## Accessing UIs

- **Jaeger**: http://localhost:16686
  - Search traces by service (sophia, hermes, apollo-cli, apollo-backend, apollo-webapp)
  - Filter by operation, tags, duration
  - View service dependency graph

- **Prometheus**: http://localhost:9090
  - Query metrics
  - View targets status
  - Explore time series data

## Service Configuration

Services should use these environment variables:
```bash
OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4319  # For Python/backend services (gRPC)
VITE_OTEL_EXPORTER_URL=http://localhost:4320/v1/traces  # For browser/webapp (HTTP)
```

**Note:** The OTel collector uses ports 4319/4320 (not the default 4317/4318) because Jaeger's OTLP receiver uses 4317 internally.

## Troubleshooting

See `docs/operations/OBSERVABILITY_QUERIES.md` for Jaeger filters and Prometheus query snippets.

### Collector not receiving traces
1. Check collector logs: `docker logs logos-otel-collector`
2. Verify collector health: `curl http://localhost:13133/`
3. Check service OTEL_EXPORTER_OTLP_ENDPOINT configuration

### Jaeger not showing traces
1. Check Jaeger logs: `docker logs logos-jaeger`
2. Verify collector → Jaeger connection in collector logs
3. Check service dropdown in Jaeger UI

### Browser traces not appearing
1. Verify CORS configuration in otel-collector-config.yaml
2. Check browser console for OTel errors
3. Verify VITE_OTEL_EXPORTER_URL points to HTTP endpoint (4318)

## Architecture Details

### Collector Configuration

The OTel collector is configured with:

**Receivers:**
- OTLP gRPC (port 4317) - for backend services
- OTLP HTTP (port 4318) - for browser/webapp traces
- CORS enabled for localhost:5173 (Apollo webapp)

**Processors:**
- `batch` - batches spans for efficiency (1s timeout, 1024 batch size)
- `memory_limiter` - prevents OOM (512 MiB limit)
- `resource` - adds environment=development label

**Exporters:**
- `jaeger` - exports traces to Jaeger via gRPC (port 14250)
- `prometheus` - exposes metrics on port 8889
- `logging` - logs traces for debugging

### Prometheus Scrape Config

Prometheus is configured to scrape:
- OTel collector metrics (port 8889)
- Sophia service metrics (port 8001)
- Hermes service metrics (port 8002)
- Apollo service metrics (port 8000)

All scrape intervals are set to 15 seconds.

### Storage

**Development:**
- Jaeger uses in-memory storage (data lost on restart)
- Prometheus uses a Docker volume for persistence

**Production:**
- Jaeger should use Cassandra or Elasticsearch
- Prometheus should use remote storage (e.g., Thanos, Cortex)

## Performance Considerations

- **Batch processor** reduces network overhead by batching spans
- **Memory limiter** prevents collector crashes under high load
- **Health checks** ensure services are ready before accepting traffic
- **CORS** properly configured for browser traces without performance impact

## Next Steps

1. Instrument services with OpenTelemetry SDKs (see issues #334-342)
2. Set up Grafana dashboards for visualization (see issue #344)
3. Configure alerting rules in Prometheus
4. Set up production-grade storage backends
