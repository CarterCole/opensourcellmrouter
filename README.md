# opensourcellmrouter

A fast, local-first LLM router. Drop it in front of any OpenAI- or Anthropic-compatible client and route requests across a configurable pipeline of local and cloud providers — with classifiers, cost/latency/random routing rules, plugins, a live dashboard, and a built-in TUI.

```
cargo install opensourcellmrouter
```

---

## What it does

Every request arrives on an OpenAI-compatible (`/v1/chat/completions`) or Anthropic-compatible (`/v1/messages`) endpoint and flows through:

```
request → classifiers → plugins → router → provider → plugins → response classifiers → response
```

- **Classifiers** tag the request (e.g. `vision`, `code`, `nsfw`) based on content.
- **Router rules** pick a provider based on tags, cost, latency, throughput, model name prefix, discovered Ollama models, or at random. Rules are evaluated in order; first match wins.
- **Plugins** can mutate the request/response or force a specific provider.
- The chosen **provider** receives the (possibly rewritten) request and returns a response.
- **Response classifiers** tag the response after it comes back (e.g. `refusal`), surfaced via the `X-Router-Response-Tags` header (request-side tags get their own `X-Router-Request-Tags` header) — the OpenAI/Anthropic response body is never modified.
- Every exchange is logged as JSONL and broadcast live to the dashboard.

---

## Quick start

```bash
git clone https://github.com/CarterCole/opensourcellmrouter
cd opensourcellmrouter

# Copy and edit the example config
cp config.example.toml config.toml

# Add API keys for any cloud providers you want
echo 'OPENAI_API_KEY=sk-...' >> .env
echo 'ANTHROPIC_API_KEY=sk-ant-...' >> .env
echo 'CLOUDFLARE_API_TOKEN=cfat_...' >> .env
echo 'XAI_API_KEY=xai-...' >> .env

# Build and run
cargo run -- my-config.toml
```

Or use the included demo script (starts llama-server if needed, opens the TUI):

```bash
./demo.sh
```

Point any OpenAI-compatible client at `http://localhost:8090/v1`.

---

## Providers

Each `[[providers]]` entry is an upstream backend. Three wire formats are supported:

| `format` | Speaks | Example |
|---|---|---|
| `openai` | OpenAI chat completions API | OpenAI, llama-server, vLLM, Cloudflare Workers AI, xAI (Grok) |
| `anthropic` | Anthropic Messages API | Anthropic Claude |
| `ollama` | Ollama native API (`/api/chat`) | Local Ollama instance |

```toml
# Local llama.cpp server (OpenAI-compatible)
[[providers]]
name                      = "local"
format                    = "openai"
base_url                  = "http://localhost:8080/v1"
cost_per_1m_tokens        = 0.0
quality                   = 60
latency_ms                = 900
throughput_tokens_per_sec = 20

# Ollama (native API, no /v1 suffix)
[[providers]]
name     = "ollama"
format   = "ollama"
base_url = "http://localhost:11434"
quality  = 75

# OpenAI (key read from $OPENAI_API_KEY)
[[providers]]
name               = "openai"
format             = "openai"
base_url           = "https://api.openai.com/v1"
api_key_env        = "OPENAI_API_KEY"
cost_per_1m_tokens = 5.0
quality            = 90

# Anthropic (key read from $ANTHROPIC_API_KEY)
[[providers]]
name               = "anthropic"
format             = "anthropic"
base_url           = "https://api.anthropic.com"
api_key_env        = "ANTHROPIC_API_KEY"
cost_per_1m_tokens = 15.0
quality            = 95

# Cloudflare Workers AI
[[providers]]
name               = "cloudflare"
format             = "openai"
base_url           = "https://api.cloudflare.com/client/v4/accounts/<ACCOUNT_ID>/ai/v1"
api_key_env        = "CLOUDFLARE_API_TOKEN"
cost_per_1m_tokens = 0.2
quality            = 80

# xAI (Grok) — OpenAI-compatible (key read from $XAI_API_KEY)
[[providers]]
name               = "xai"
format             = "openai"
base_url           = "https://api.x.ai/v1"
api_key_env        = "XAI_API_KEY"
cost_per_1m_tokens = 5.0
quality            = 88
```

**API keys** go in `.env` (gitignored) and are sourced automatically by `demo.sh`, or export them in your shell before running. A provider with a missing key is skipped automatically at startup (logged as a warning) — it's never selected by any router rule.

---

## Router rules

Rules live in `[[routers]]` and are evaluated top-to-bottom; first match wins.

### `prefix` — route by model name prefix

```toml
[[routers]]
type          = "prefix"
model_prefix  = "local/"
provider      = "local"
rewrite_model = "llama3.2-3b"
```

### `tag` — route by classifier tag

```toml
[[routers]]
type          = "tag"
tag           = "code"
provider      = "ollama"
rewrite_model = "deepseek-r1:latest"
```

### `discover` — route to Ollama if it has the model

Queries `GET /api/tags` at startup and routes requests whose model name appears in the result.

```toml
[[routers]]
type     = "discover"
provider = "ollama"
```

### `price` — pick the cheapest provider

```toml
[[routers]]
type                    = "price"
providers               = ["local", "openai"]   # omit for all providers
max_cost_per_1m_tokens  = 5.0                   # optional ceiling
```

### `latency` — pick the fastest provider

```toml
[[routers]]
type           = "latency"
max_latency_ms = 500
```

### `throughput` — pick the highest-throughput provider

```toml
[[routers]]
type               = "throughput"
min_tokens_per_sec = 30
```

### `fallback` — score-based catch-all

Ranks by `quality_bias * quality - (1 - quality_bias) * cost`. Good chain terminator.

```toml
[[routers]]
type         = "fallback"
quality_bias = 0.7        # 0 = cheapest, 1 = highest quality
```

### `random` — pick at random

```toml
# Random provider from all configured providers:
[[routers]]
type = "random"

# Or pick from explicit (provider, model) pairs:
[[routers]]
type = "random"
candidates = [
  { provider = "local",      model = "llama3.2-3b"              },
  { provider = "ollama",     model = "deepseek-r1:latest"       },
  { provider = "cloudflare", model = "@cf/meta/llama-3.1-8b-instruct" },
  { provider = "openai",     model = "gpt-4o-mini"              },
]
```

See [`docs/examples.md`](docs/examples.md) for full end-to-end recipes.

---

## Classifiers

Classifiers run on every request before routing and attach tags to it. The only built-in classifier is `keyword`, which matches words in the prompt:

```toml
[classifiers.keyword]
enabled = true

[classifiers.keyword.tags]
vision = ["image", "photo", "screenshot", "diagram"]
code   = ["function", "class", "import", "def ", "fn "]
nsfw   = ["nsfw", "adult", "explicit"]
```

Tags are available to `tag` router rules and appear in logs and the dashboard.

There's also a response-side counterpart: `[response_classifiers.<id>]` tags
the response *after* the provider replies, e.g. to flag a refusal:

```toml
[response_classifiers.refusal]
enabled = true
```

Every response carries six `X-Router-*` headers without ever touching the
OpenAI/Anthropic response body: `X-Router-Request-Tags` (e.g. `vision`),
`X-Router-Response-Tags` (e.g. `refusal`), `X-Router-Provider` (which
provider handled it), `X-Router-Model` (the model actually sent), and
`X-Router-Input-Tokens`/`X-Router-Output-Tokens` (token usage). The same data
shows up in logs/dashboard. See [`docs/classifiers.md`](docs/classifiers.md#response-classifiers).

---

## Dashboard and TUI

### Browser dashboard

`GET /dashboard` streams a live feed of every request via SSE. Enable it in config:

```toml
[server]
dashboard = true
port      = 8090
```

The feed emits four event types per request — `start`, `classified`, `routed`, and `complete` — so you see classifier tags and routing decisions appear in real time, before the response arrives.

### Terminal TUI

```bash
opensourcellmrouter tui http://localhost:8090
```

Three-pane UI: live pipeline feed (top), running stats by provider/tag (bottom-left), built-in chat client (bottom-right). Keys: `Tab`/`i` to focus chat, `↑↓` to scroll feed, `q` to quit.

### Watch mode

```bash
opensourcellmrouter watch http://localhost:8090
```

Prints each request's classifier tags, routing decision, and response to stdout as they happen.

---

## Logging

```toml
[logging]
enabled = true
path    = "logs/requests.jsonl"
```

Every completed request is appended as a line of JSON including provider, requested model, sent model, tags, plugins, messages, response, and duration.

---

## Plugins

Plugins hook into the pipeline at four stages: `on_start`, `pre_request`, `post_response`, and `on_end`. Two are built in:

| Plugin | What it does |
|---|---|
| `response-healing` | Repairs truncated or malformed JSON in responses |
| `pareto-router` | Forces requests to a tier (low/medium/high) based on config or per-request override |

```toml
[plugins.response-healing]
enabled = true
```

---

## Configuration reference

| Section | Docs |
|---|---|
| Providers | [`docs/providers.md`](docs/providers.md) |
| Routers | [`docs/routers.md`](docs/routers.md) |
| Classifiers | [`docs/classifiers.md`](docs/classifiers.md) |
| Plugins | [`docs/plugins.md`](docs/plugins.md) |
| Pipeline overview | [`docs/README.md`](docs/README.md) |
| Examples | [`docs/examples.md`](docs/examples.md) |
| Coding agents (Claude Code, Copilot CLI, Codex) | [`docs/coding-agents.md`](docs/coding-agents.md) |
| Security (`host`, `api_key_env`) | [`docs/security.md`](docs/security.md) |

---

## Building from source

```bash
cargo build --release
./target/release/opensourcellmrouter config.toml
```

Requires Rust 1.85+ (edition 2024).

---

## License

MIT — see [LICENSE](LICENSE).
