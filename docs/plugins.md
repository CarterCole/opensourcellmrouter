# Plugins

`src/plugins/` implements request/response middleware, modeled after
OpenRouter's `plugins` array. A plugin is anything implementing the
`Plugin` trait (`src/plugins/mod.rs`):

```rust
#[async_trait]
pub trait Plugin: Send + Sync {
    fn id(&self) -> &'static str;

    async fn on_start(&self, ctx: &PluginContext, req: &mut ChatRequest, resp: &mut Option<ChatResponse>) -> anyhow::Result<Flow> {
        Ok(Flow::Continue)
    }

    async fn pre_request(&self, ctx: &PluginContext, req: &mut ChatRequest, resp: &mut Option<ChatResponse>) -> anyhow::Result<Flow> {
        Ok(Flow::Continue)
    }

    async fn post_response(&self, ctx: &PluginContext, req: &ChatRequest, resp: &mut Option<ChatResponse>) -> anyhow::Result<Flow> {
        Ok(Flow::Continue)
    }

    async fn on_end(&self, ctx: &PluginContext, req: &ChatRequest, resp: &mut Option<ChatResponse>) -> anyhow::Result<Flow> {
        Ok(Flow::Continue)
    }
}
```

A plugin implements only the hooks it needs; the rest default to
`Ok(Flow::Continue)`.

## Stages and Flow

The four hooks correspond to four points in the pipeline (see
[docs/README.md](README.md)):

| Hook | Stage | `req` | `resp` |
|---|---|---|---|
| `on_start` | before classifiers | `&mut` | `&mut Option<ChatResponse>`, starts `None` |
| `pre_request` | after classifiers, before routing | `&mut` | `&mut Option<ChatResponse>` |
| `post_response` | after a response is available | `&` | `&mut Option<ChatResponse>`, normally `Some` |
| `on_end` | just before logging/returning | `&` | `&mut Option<ChatResponse>` |

Every hook returns `anyhow::Result<Flow>`:

- **`Flow::Continue`** — fall through to the next plugin, and eventually the
  next pipeline stage. This is the common case: a plugin mutates `req` (or
  `resp`) and lets the pipeline proceed normally.
- **`Flow::Stop`** — stop running further hooks.
  - From `on_start` or `pre_request`: also skips classifiers/routing/the
    provider call entirely. The plugin must have written a response into
    `*resp` itself (e.g. a moderation plugin that rejects the request with a
    canned reply) — otherwise the request fails with `500` ("pipeline
    stopped without producing a response").
  - From `post_response`: also skips `on_end`.
  - From `on_end`: just stops earlier than it otherwise would (no
    observable difference, since it's the last stage).

If `resp` is already `Some` after `on_start`/`pre_request` (whether via
`Flow::Stop` or because a hook set it and returned `Flow::Continue`),
routing and the provider call are skipped — `PostResponse`/`End` then run on
that response as if it came from a provider.

An error from `on_start`/`pre_request` aborts the request with
`500 Internal Server Error` and the error message (`ApiError::Plugin`). An
error from `post_response`/`on_end` is logged as a warning and treated as
`Flow::Continue` — by that point a response already exists, so a failing
"nice to have" hook shouldn't take down an otherwise-successful request.

## Enabling a plugin

**Server-side default**, via `[plugins.<id>]` in `config.toml`:

```toml
[plugins.response-healing]
enabled = true
```

Any other keys in that table become the plugin's default `settings`.

**Per-request**, via the request body's `plugins` array (not part of the
standard OpenAI/Anthropic schema; stripped before forwarding upstream):

```json
{
  "model": "gpt-4o",
  "messages": [...],
  "plugins": [{"id": "pareto-router", "tier": "high"}]
}
```

A plugin enabled by config runs for every request; its settings are its
config defaults merged with (and overridden by) any matching entry in that
request's `plugins` array. A plugin *not* enabled by config can still be
turned on for a single request by naming it in `plugins`.

## Built-in plugins

### `response-healing`

Best-effort repair of malformed JSON in a model's reply: strips markdown
code fences, trims surrounding prose, removes trailing commas, and balances
unmatched brackets/quotes left by truncated output. Only touches the
response if its content isn't already valid JSON and a repair attempt
produces valid JSON — plain prose replies are left untouched.

### `pareto-router`

Lets a request pick a named "coding quality tier" (e.g. `low`/`medium`/
`high`) instead of a specific provider. Sets `forced_provider` to the first
provider in the chosen tier, bypassing `routers` entirely.

```toml
[plugins.pareto-router]
enabled = true
default_tier = "medium"

[plugins.pareto-router.tiers]
low = ["local-llama"]
medium = ["openai"]
high = ["anthropic"]
```

A request can override the tier per-call: `{"plugins": [{"id":
"pareto-router", "tier": "high"}]}`.

### `web` (Web Search) — not yet implemented

Enabling `[plugins.web]` or requesting `{"id": "web"}` fails the request with
a clear error. To implement: configure a search backend (`base_url`,
`api_key_env`, `max_results`), query it from `pre_request` with the last user
message, and prepend results to `req.system` as context.

### `pdf` (PDF Inputs) — not yet implemented

Enabling `[plugins.pdf]` or requesting `{"id": "pdf"}` fails the request with
a clear error. To implement: extend `canonical::Message` to carry document
attachments (mirroring Anthropic's `document` blocks / OpenAI's `file`
content parts), add a PDF text-extraction crate, and have `pre_request`
replace each attachment with its extracted text.

## Writing a new plugin

1. Add a module under `src/plugins/`, implement `Plugin` for a unit struct.
2. Register it in `PluginRegistry::from_config` (`src/plugins/mod.rs`).
3. Document its `id()`, any `[plugins.<id>]` settings it reads via
   `ctx.settings`, and which hooks it implements.
