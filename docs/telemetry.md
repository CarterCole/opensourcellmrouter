# Telemetry (OpenTelemetry)

When enabled, every request produces an OTel trace, contributes to a handful
of request-count/latency/token metrics, and has its `tracing::*!` log calls
exported as OTel log records — all via OTLP/HTTP to a collector or backend
of your choice. Disabled by default; omitting `[telemetry]` from
`config.toml` leaves the router's behavior unchanged from before this
feature existed.

## What gets traced

```text
request (tower_http root span, one per HTTP request)
  └─ dispatch (or dispatch_stream) — fields: request_id, provider, model, tags
       └─ provider.send — fields: provider, model
            └─ provider.http — fields: provider, kind (openai_chat, ollama_chat, ...)
```

For streaming requests (`stream: true`), a second span, `dispatch_stream.body`,
covers the lifetime of the spawned task that proxies the SSE stream to the
client — `dispatch_stream` itself only covers setup (classify, route, open
the upstream connection), since the proxy loop outlives the handler's return.
Both spans share the same trace id.

## What gets measured

Four instruments, recorded once per completed request (success or error),
all tagged with `provider`, `model`, `error` attributes:

| Instrument | Type | Meaning |
|---|---|---|
| `router.requests` | counter | Requests handled |
| `router.request.duration_ms` | histogram | End-to-end pipeline duration |
| `router.tokens.input` | counter | Input tokens (omitted on error) |
| `router.tokens.output` | counter | Output tokens (omitted on error) |

## What gets logged

The existing `tracing::info!`/`warn!` call sites throughout the router (e.g.
model discovery in `router.rs`, startup in `main.rs`) are bridged into OTel
log records via `opentelemetry-appender-tracing`, exported alongside traces
and metrics — no new log call sites, no change to the existing JSONL request
log (`[logging]`) or dashboard SSE feed, which remain the primary
structured-data surfaces for request auditing.

## Config

```toml
[telemetry]
enabled       = true
otlp_endpoint = "http://localhost:4318"   # OTLP/HTTP endpoint
service_name  = "opensourcellmrouter"
sample_ratio  = 1.0                       # trace sampling, 0.0-1.0
```

| Field | Default | Meaning |
|---|---|---|
| `enabled` | `false` | Turn OTel export on. When `false` (or the section is omitted), no exporters are constructed and `tracing` behaves exactly as without this feature. |
| `otlp_endpoint` | `http://localhost:4318` | OTLP/HTTP endpoint for traces, metrics, and logs. |
| `service_name` | `opensourcellmrouter` | `service.name` resource attribute. |
| `sample_ratio` | `1.0` | Fraction of traces sampled. Applies to traces only — metrics and logs are always exported in full when enabled. |

## Correlating a log line with its trace

Every `Complete` event in the JSONL request log (`[logging]`) and SSE feed
now carries a `trace_id` field (a hex string, `null` when telemetry is
disabled). Paste it into your backend's trace search (e.g. Jaeger's
"Trace ID" search box) to jump straight from a logged request to its trace.

## Local verification

```bash
docker run -d --name jaeger \
  -p 16686:16686 -p 4317:4317 -p 4318:4318 \
  jaegertracing/all-in-one:latest
```

Jaeger's all-in-one image accepts OTLP directly on 4317/4318 — no separate
collector needed for local testing. Set `[telemetry] enabled = true`,
`otlp_endpoint = "http://localhost:4318"`, run the router, send a few
requests, then open `http://localhost:16686` and search for service
`opensourcellmrouter`.

Note that both the trace batch processor and the metrics periodic reader
buffer in memory and flush on an interval (or on graceful shutdown) — sending
one request and immediately killing the process may not be enough time to
see it exported.
