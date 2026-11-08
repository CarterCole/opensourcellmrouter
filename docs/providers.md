# Providers

Each `[[providers]]` entry in `config.toml` is an upstream LLM backend.
`src/provider.rs` translates the canonical `ChatRequest`/`ChatResponse`
(`src/canonical.rs`) to and from whichever wire format the provider speaks,
based on its `format`. Every format supports both plain and streaming
(`stream: true`) requests, and inline (base64) images. Audio, which only
OpenAI-format clients can send (`input_audio`), reaches only `openai`
providers; the others drop it and log a warning.

```toml
[[providers]]
name = "openai"
format = "openai"
base_url = "https://api.openai.com/v1"
api_key_env = "OPENAI_API_KEY"
cost_per_1m_tokens = 5.0
quality = 80
latency_ms = 600
throughput_tokens_per_sec = 60
```

| Field | Meaning |
|---|---|
| `name` | How `routers` and plugins refer to this provider. |
| `format` | `"openai"`, `"anthropic"`, or `"ollama"` — see below. |
| `base_url` | Provider root. Trailing `/` is stripped. |
| `api_key_env` | Env var holding the API key, read lazily at call time. Omit for providers that don't need one (local servers, Ollama). |
| `cost_per_1m_tokens`, `quality`, `latency_ms`, `throughput_tokens_per_sec` | Used by the `price`/`latency`/`throughput`/`fallback` [router rules](routers.md); otherwise informational. |

## `format = "openai"`

Speaks the OpenAI chat completions API: `POST {base_url}/chat/completions`,
bearer-auth via `api_key_env`. Also used for any OpenAI-compatible server
(e.g. `llama-server`, vLLM, or Ollama's `/v1` compatibility endpoint) and for
OpenAI-compatible cloud APIs such as Cloudflare Workers AI or
[xAI](https://docs.x.ai/) (`base_url = "https://api.x.ai/v1"`,
`api_key_env = "XAI_API_KEY"`).

## `format = "anthropic"`

Speaks the Anthropic Messages API: `POST {base_url}/messages` with an
`anthropic-version` header and `x-api-key` auth from `api_key_env`. Since
`/messages` is appended directly, `base_url` includes the `/v1`:

```toml
[[providers]]
name        = "anthropic"
format      = "anthropic"
base_url    = "https://api.anthropic.com/v1"
api_key_env = "ANTHROPIC_API_KEY"
```

Streaming works too. A `stream: true` request is sent upstream with
`"stream": true`, and the provider's SSE events (`content_block_delta` text
and `input_json_delta` tool-argument deltas, `message_delta` stop reason and
usage) are re-rendered live for whichever endpoint the client called. So
Claude Code on `/v1/messages`, Copilot CLI on `/v1/chat/completions` and
Codex on `/v1/responses` can all stream from a Claude model, tool calls
included. Some things don't carry over:

- Thinking blocks, and server-side tool blocks such as `server_tool_use`, are
  dropped from the stream, as they are from a plain reply — see
  [Thinking blocks](#thinking-blocks).
- An upstream `error` event mid-stream (e.g. `overloaded_error`) cuts off the
  client's connection, so the client sees a failed stream rather than one
  that looks complete.

### Thinking blocks

Current Claude models think adaptively by default and answer with `thinking`
blocks (their text is empty unless the request sets `thinking.display` to
`"summarized"`) or `redacted_thinking` blocks. The router doesn't carry them
through, streamed or not:

- They are dropped from the provider's reply. The thinking still happens and
  is billed, and `thinking` and `output_config.effort` in a request are
  forwarded as usual, but the client gets only the text and tool calls, and
  summarized thinking text is discarded.
- `thinking` and `redacted_thinking` blocks in an incoming `/v1/messages`
  request, such as a client replaying its history, are dropped before the
  request is forwarded. An assistant turn left with nothing in it is omitted.
- Other block types the router doesn't model (`server_tool_use`,
  `web_search_tool_result`, `document`, anything Anthropic adds later) are
  dropped the same way instead of failing the request. Unlike thinking,
  content in an incoming request is something the model won't see, so that
  is logged as a warning.

The API wants thinking blocks passed back unchanged, but accepts a history
with none, so dropping every one is safe. The cost is that the model answers
each turn without its earlier reasoning, between the tool calls of a single
task included.

## `format = "ollama"`

Speaks [Ollama](https://ollama.com)'s native API directly — `POST
{base_url}/api/chat` — rather than its OpenAI-compatible mode. `base_url` is
Ollama's root with **no** `/v1` suffix, e.g. `http://localhost:11434`.
`api_key_env` is normally omitted (Ollama doesn't require auth by default).

```toml
[[providers]]
name = "ollama"
format = "ollama"
base_url = "http://localhost:11434"
```

### Model discovery

At startup, `ModelRouter::discover_models` calls `GET {base_url}/api/tags` on
every provider (a no-op for non-`ollama` providers) to find out which models
Ollama actually has pulled. The result is logged and cached for the
[`discover` router rule](routers.md#discover), which routes a request
straight to a provider if its `model` is one Ollama reports having — useful
since the set of locally pulled models changes as you `ollama pull`/`rm`
things, without needing to keep `routers` in sync by hand.

If Ollama isn't running (or `/api/tags` fails for any reason), discovery logs
a warning and that provider simply has no known models — `discover` rules
for it pass through to the next rule, same as if it had never matched.
