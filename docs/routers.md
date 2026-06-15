# Routers

`src/router.rs` resolves a request's `model` (plus any
[classifier](classifiers.md) tags) to a provider and a final model name,
by walking the `routers` chain from `config.toml` **in order**. Each rule
either:

- **resolves** the request — returns a provider name and a (possibly
  rewritten) model name, ending the chain, or
- **passes through** — returns nothing, so the next rule gets a chance.

If no rule resolves the request, the call fails with `400 Bad Request`
("no provider configured for model '...'"). A `fallback` rule (see below)
is a sensible chain terminator that always resolves, as long as at least one
provider is configured.

This entirely replaces the older `routes` + `default_providers` /
`routing.quality_bias` config shape — there is now just one ordered list,
`routers`.

## Bypassing the chain: `forced_provider`

A [plugin](plugins.md) (e.g. `pareto-router`) can set
`ChatRequest.forced_provider` during its `pre_request` hook. If set, the
`routers` chain is skipped entirely and the request goes straight to that
named provider, with the model left unchanged.

## Rule types

Each entry in `routers` is a table with a `type` field selecting the
variant below.

### `prefix`

Matches if `model` starts with `model_prefix`. Optionally rewrites the model
name before forwarding — useful for local servers (like `llama-server`) that
ignore the `model` field but still want a recognizable name in routing config
and logs.

```toml
[[routers]]
type = "prefix"
model_prefix = "local/"
provider = "local-llama"
rewrite_model = "local-model"
```

### `tag`

Matches if `tag` is among the tags a [classifier](classifiers.md) assigned to
this request (`ChatRequest.tags`). This is how classifier output drives
routing — e.g. send anything the `keyword` classifier tagged `"vision"` to a
multimodal model, regardless of what model name the client asked for.

```toml
[[routers]]
type = "tag"
tag = "vision"
provider = "openai"
rewrite_model = "gpt-4o"
```

### `price`

Picks the candidate provider with the lowest `cost_per_1m_tokens`. If
`max_cost_per_1m_tokens` is set, candidates above it are excluded; if that
leaves nothing, this rule passes through. `providers` restricts the
candidate set (default: every configured provider).

```toml
[[routers]]
type = "price"
providers = ["local-llama", "openai"]
max_cost_per_1m_tokens = 10.0
```

### `latency`

Picks the candidate with the lowest `latency_ms`. Providers that don't
declare `latency_ms` are excluded. `max_latency_ms` works like
`max_cost_per_1m_tokens` above.

```toml
[[routers]]
type = "latency"
max_latency_ms = 1000
```

### `throughput`

Picks the candidate with the highest `throughput_tokens_per_sec`. Providers
that don't declare it are excluded. `min_tokens_per_sec` works like the
thresholds above, but candidates *below* it are excluded.

```toml
[[routers]]
type = "throughput"
min_tokens_per_sec = 30
```

### `fallback`

Always resolves (given at least one candidate): ranks candidates by

```text
score = quality_bias * quality - (1 - quality_bias) * cost_per_1m_tokens
```

and picks the highest. `quality_bias` defaults to `0.5`; `0.0` always prefers
the cheapest provider, `1.0` always prefers the highest `quality`. A good
chain terminator.

```toml
[[routers]]
type = "fallback"
quality_bias = 0.5
```

### `discover`

Matches if `model` is one of the models `provider` reports having available,
as discovered at startup via `GET /api/tags` (currently only implemented for
`format = "ollama"` providers — see [providers.md](providers.md)). If
discovery hasn't run, found nothing, or didn't report `model`, this rule
passes through.

```toml
[[routers]]
type = "discover"
provider = "ollama"
```

Useful for routing requests straight to whatever models are actually pulled
on a local Ollama instance, without hardcoding model names in `routers`.

## Provider fields used by routers

These fields on `[[providers]]` entries are read by the rules above (and are
otherwise informational):

| Field | Used by |
|---|---|
| `cost_per_1m_tokens` | `price`, `fallback` |
| `quality` | `fallback` |
| `latency_ms` | `latency` |
| `throughput_tokens_per_sec` | `throughput` |
