# Classifiers

`src/classifiers/` tags a request *before* it's routed, so
[routers](routers.md) can make capability- or policy-based decisions — e.g.
send anything that looks like an image request to a multimodal model, or
keep anything flagged `nsfw` on a local model instead of a cloud provider.

```text
prompt -> classifiers -> plugins (pre_request) -> routers -> provider -> ...
```

A classifier is anything implementing the `Classifier` trait
(`src/classifiers/mod.rs`):

```rust
#[async_trait]
pub trait Classifier: Send + Sync {
    fn id(&self) -> &'static str;

    async fn classify(&self, ctx: &ClassifierContext, req: &ChatRequest) -> anyhow::Result<Vec<String>>;
}
```

`classify` inspects the request (as the client sent it, before any plugin
mutates it) and returns zero or more tags. Every classifier enabled in
config runs, in registry order, and their tags are merged (de-duplicated,
in order) into `ChatRequest.tags`. A classifier that returns an error is
logged as a warning and skipped — a failed classifier never fails the
request, it just means that classifier's tags are missing.

`ChatRequest.tags` is not forwarded to providers, but is recorded in the
request log (`tags` field) alongside the plugin ids that ran.

## Enabling a classifier

Via `[classifiers.<id>]` in `config.toml`:

```toml
[classifiers.keyword]
enabled = true

[classifiers.keyword.tags]
vision = ["image", "photo", "picture", "screenshot"]
nsfw = ["..."]
```

Any other keys in that table become the classifier's `settings`
(`ctx.settings`). Unlike plugins, classifiers don't currently have a
per-request opt-in — they're a server-side policy, not something a client
toggles.

## Built-in classifiers

### `keyword`

Concatenates the system prompt and every message's content, lowercases it,
and checks each configured tag's keyword list for a substring match. Any tag
with at least one match is added to `ChatRequest.tags`.

This is a simple, configurable baseline — not a real moderation or modality
model — but it's enough to drive `tag`-based routing (see
[routers.md](routers.md#tag)) for common cases: route "vision"-tagged
requests to a multimodal model, or "nsfw"-tagged requests to a local/
moderation provider.

## Using tags in routers

Add a `tag`-type rule to `routers` (checked in order, like any other rule):

```toml
[[routers]]
type = "tag"
tag = "vision"
provider = "openai"
rewrite_model = "gpt-4o"
```

## Writing a new classifier

1. Add a module under `src/classifiers/`, implement `Classifier` for a unit
   struct.
2. Register it in `ClassifierRegistry::from_config`
   (`src/classifiers/mod.rs`).
3. Document its `id()`, any `[classifiers.<id>]` settings it reads via
   `ctx.settings`, and the tags it can produce.

More sophisticated classifiers — e.g. an actual vision/audio modality
detector (which would need `canonical::Message` to carry multimodal content
blocks, not just `String`), or an ML-based moderation call — fit the same
trait; `keyword` is deliberately the simplest possible implementation to
build against.
