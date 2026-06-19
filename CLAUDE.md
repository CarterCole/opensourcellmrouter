# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Build
cargo build

# Run (defaults to config.toml in cwd)
cargo run
cargo run -- path/to/config.toml

# Tests
cargo test
cargo test router::tests   # run a single test module

# Logging level
RUST_LOG=debug cargo run
```

## Architecture

This is an async Rust proxy server (Axum + Tokio) that exposes two API endpoints — OpenAI-compatible `/v1/chat/completions` and Anthropic-compatible `/v1/messages` — and routes requests to one or more upstream LLM providers via a configurable pipeline.

### Pipeline

Every inbound request flows through:

```
Start → classifiers → PreRouting → routers → provider → PostResponse → response_classifiers → End → logging
```

1. Plugin `on_start` hooks
2. `classifiers` tag the request (e.g. `"vision"`, `"nsfw"`)
3. Plugin `pre_request` hooks (can set `forced_provider` to bypass routing)
4. `routers` chain picks a provider and optionally rewrites the model name
5. Request forwarded to the chosen provider
6. Plugin `post_response` hooks (e.g. JSON repair)
7. `response_classifiers` tag the response (e.g. `"refusal"`), written to `ChatResponse.tags`
8. Plugin `on_end` hooks
9. Request/response logged to JSONL file and/or broadcast to `/dashboard/events` SSE; `dispatch()` also returns a `DispatchOutcome` (response + request tags + provider + model) that the handlers turn into six `X-Router-*` response headers (`X-Router-Request-Tags`, `X-Router-Response-Tags`, `X-Router-Provider`, `X-Router-Model`, `X-Router-Input-Tokens`, `X-Router-Output-Tokens`) — not the body, so the OpenAI/Anthropic wire formats stay unmodified

A plugin hook returning `Flow::Stop` from `on_start`/`pre_request` skips routing and the provider call entirely (the plugin must populate `resp` itself). Errors in `on_start`/`pre_request` abort with 500; errors in `post_response`/`on_end` are logged and ignored.

### Key modules

- **`canonical.rs`** — internal `ChatRequest`/`ChatResponse` types that every wire format converts to/from. `ChatRequest.tags` is populated by classifiers; `ChatRequest.forced_provider` is set by plugins; `ChatResponse.tags` is populated by response classifiers. Neither `tags` field is mapped by the `formats/` `From` impls, so they reach clients only via the `X-Router-Request-Tags`/`X-Router-Response-Tags` headers (see `server::dispatch`).
- **`config.rs`** — TOML config deserialized at startup. Defines `RouterRule` variants, `ProviderConfig`, `PluginEntryConfig`, etc.
- **`router.rs`** — `ModelRouter` walks the ordered `routers` chain. At startup it calls `discover_models()` to populate `available_models` (used by the `discover` rule for Ollama).
- **`server.rs`** — Axum routes, `AppState`, and the `dispatch()` function that orchestrates the full pipeline. Also serves the live dashboard.
- **`formats/`** — Wire-format adapters: `openai.rs`, `anthropic.rs`, `ollama.rs`. Each implements `From`/`Into` for `ChatRequest`/`ChatResponse`.
- **`plugins/mod.rs`** — `Plugin` trait and `PluginRegistry`. Plugins registered here in a fixed order; config enables/disables defaults; requests can opt in per-call via `"plugins": [{"id": "..."}]`.
- **`classifiers/mod.rs`** — `Classifier` trait/`ClassifierRegistry` (pre-routing, request tags; `keyword.rs`) and `ResponseClassifier` trait/`ResponseClassifierRegistry` (post-response, response tags; `refusal.rs`).
- **`provider.rs`** — `Provider` wraps a `ProviderConfig` and handles the actual HTTP call to the upstream, translating via `formats/`.

### Router rules (evaluated in order, first match wins)

| Rule | Matches when |
|---|---|
| `prefix` | `model` starts with `model_prefix` |
| `tag` | a classifier tag matches |
| `price` | picks lowest-cost candidate (pass-through if none qualify) |
| `latency` | picks lowest-latency candidate |
| `throughput` | picks highest-throughput candidate |
| `discover` | `model` is in the provider's discovered model list (Ollama only) |
| `fallback` | always resolves; scores by `quality_bias * quality - (1-quality_bias) * cost` |

A `fallback` rule is typically the last entry to ensure every request resolves.

### Adding a new plugin

1. Create `src/plugins/<name>.rs` implementing `Plugin` for a unit struct
2. Register it in `PluginRegistry::from_config` in `src/plugins/mod.rs`
3. Document its `id()` and any config settings it reads via `ctx.settings`

### Adding a new classifier

1. Create `src/classifiers/<name>.rs` implementing `Classifier`
2. Register it in `ClassifierRegistry::from_config` in `src/classifiers/mod.rs`

### Adding a new response classifier

1. Create `src/classifiers/<name>.rs` implementing `ResponseClassifier`
2. Register it in `ResponseClassifierRegistry::from_config` in `src/classifiers/mod.rs`
3. Enable it via `[response_classifiers.<id>]` in config; tags it produces land in `ChatResponse.tags` and the `X-Router-Tags` response header

### Dashboard

`GET /dashboard` serves an embedded HTML page (`static/dashboard.html`) that streams live request events via SSE from `/dashboard/events`. Disable with `dashboard = false` under `[server]` in config.

### Logging

When `[logging] enabled = true`, every request/response is appended as a JSONL line to `path`. The same JSON is also broadcast to the SSE feed, so the dashboard and file log are always in sync.
