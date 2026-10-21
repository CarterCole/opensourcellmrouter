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

    async fn classify(&self, ctx: &ClassifierContext, req: &ChatRequest) -> anyhow::Result<Classification>;
}

pub struct Classification {
    pub tags: Vec<String>,
    pub route: Option<RouteChoice>,
}
```

`classify` inspects the request (as the client sent it, before any plugin
mutates it) and returns zero or more tags. Every classifier enabled in
config runs, in registry order, and their tags are merged (de-duplicated,
in order) into `ChatRequest.tags`. A classifier that returns an error is
logged as a warning and skipped — a failed classifier never fails the
request, it just means that classifier's tags are missing.

A classifier can also pick the model to route to outright, by returning a
`route` (see [`cev`](#cev) and [`llm`](#llm)). The first one returned
becomes `ChatRequest.route`, which a [`cev`](routers.md#cev) or
[`llm`](routers.md#llm) router rule acts on — each only on its own
classifier's picks.

`ChatRequest.tags` and `ChatRequest.route` are not forwarded to providers,
but are recorded in the request log (`tags` and `route` fields) alongside
the plugin ids that ran.

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

### `cev`

Asks a [cev](https://github.com/CarterCole/rust-cev) model which of the
configured models should answer the request, and learns from corrections.
Each request becomes one cev *choice* question about the system prompt and
the latest user message; the options are the `models` below, and the answer
is one of them plus a probability for each. No text is generated.

```toml
[classifiers.cev]
enabled = true
url     = "http://127.0.0.1:8081"   # a running `cev` server

[[classifiers.cev.models]]
provider    = "ollama"
model       = "deepseek-r1:latest"
description = "Programming, debugging, math and step-by-step reasoning"

[[classifiers.cev.models]]
provider    = "ollama"
model       = "llama3.1:8b"
description = "General questions, writing, summarizing and conversation"

[[routers]]
type = "cev"
```

It produces no tags. Its pick only affects routing through a
[`cev` rule](routers.md#cev) in `routers`, so where that rule sits in the
chain decides which rules can override it.

| Setting | Default | |
|---|---|---|
| `url` | | Base URL of a `cev` server. Set this or `model`. |
| `api_key_env` | | Environment variable holding the server's bearer token. |
| `model` | | Qwen3 checkpoint (Hugging Face repo id or directory) to run in-process instead of `url`. |
| `db` | `cev.db` | With `model`: SQLite file for decisions, feedback and adapters. |
| `models` | | The options to choose between (at least two). |
| `instructions` | "Which model is best suited to answer this request?" | The question. |
| `task` | `llmrouter.route` | Key of the online adapter that learns from feedback. |
| `timeout_ms` | `5000` | How long to wait for cev before routing without it. |
| `max_chars` | `2000` | How much of the system prompt and of the message to send. |

Each of `models` has a `provider` and `model` to route to, an optional
`description` of what the model is good at (cev sees it next to the name),
and an optional `name` to pick and label it by, which defaults to `model`.

**Server or in-process.** With `url`, picks come from a `cev` server
(`cargo install cev-server --features metal`). With `model`, the router
loads the checkpoint itself, which needs a build with the `cev-local`
feature:

```bash
cargo run --release --features cev-local
```

On macOS that build runs the model on the GPU (Metal); add `--features cuda`
for an NVIDIA one. Either way a GPU matters: Qwen3-1.7B takes roughly 0.1 s
per pick on Metal and over a second on CPU. If cev can't be reached, or
doesn't answer within `timeout_ms`, the request is routed without a pick.

**Feedback.** Every pick has a decision id, returned to the client in the
`x-cev-decision-id` response header (with the picked name in
`x-cev-choice`) and recorded in the request log. Send it back with the
model that should have been picked:

```bash
curl localhost:8090/v1/feedback -H 'content-type: application/json' \
  -d '{"decision_id": "dec_…", "label": "llama3.1:8b", "comment": "optional note"}'
```

The [dashboard](README.md#live-dashboard) does the same when you click a
model on a request. cev stores the label with the decision and updates its
adapter for `task` straight away; it starts serving the adapter once it has
8 labels and predicts them better than the base model does. The response
says where that stands (`task_examples`, `adapter_active`). Labelling the
model that was picked confirms it, which is as useful as a correction.

### `llm`

Asks a chat model which of the configured models should answer the
request — any model a configured provider serves, e.g. Claude Haiku through
an [`anthropic` provider](providers.md#format--anthropic), or a local one
through Ollama. The model gets the options (name and description) in its
system prompt and the request's system prompt and latest user message as
its message, and replies with the name of one option.

```toml
[classifiers.llm]
enabled  = true
provider = "anthropic"          # the provider to ask
model    = "claude-haiku-5-5"   # and which of its models

[[classifiers.llm.models]]
provider    = "anthropic"
model       = "claude-sonnet-5-5"
description = "Hard problems: architecture, tricky bugs, long documents"

[[classifiers.llm.models]]
provider    = "ollama"
model       = "llama3.1:8b"
description = "General questions, writing, summarizing and conversation"

[[routers]]
type = "llm"
```

Like `cev`, it produces no tags, and its pick only affects routing through
an [`llm` rule](routers.md#llm).

| Setting | Default | |
|---|---|---|
| `provider` | | Name of the provider to ask. If it has an `api_key_env`, that variable has to be set, or the router won't start. |
| `model` | | The model to ask, as that provider names it. |
| `models` | | The options to choose between (at least two), the same as `cev`'s. |
| `instructions` | "Which model is best suited to answer this request?" | The question. |
| `timeout_ms` | `5000` | How long to wait for an answer before routing without one. |
| `max_chars` | `2000` | How much of the system prompt and of the message to send. |
| `max_tokens` | `64` | Cap on the answer. Raise it for a model that thinks before answering (a `<think>…</think>` block is skipped). |

An option on a provider whose `api_key_env` isn't set is dropped with a
warning, the same way the router skips that provider. If the answer
doesn't name exactly one option, the call fails, or it takes longer than
`timeout_ms`, the request is routed without a pick.

**Compared with `cev`.** No second server or GPU, and the judgment of
whichever model you pick — but every request waits for a full chat
completion (and, for a cloud model, pays for one) before it's routed. A pick
has no probabilities (`confidence` is always 1), and can't be corrected:
there's no decision id, so no `x-cev-*` headers and no `/v1/feedback`. With
both classifiers enabled only `cev`'s pick is kept, so enable one.

Both classify each request on its own: the latest user message that has
text (so an agent's tool-result turns are classified by what the user last
asked), not the conversation, which means turns of one conversation can go
to different models.

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
   struct (or, like `cev` and `llm`, for one built from its settings at
   startup).
2. Register it in `ClassifierRegistry::from_config`
   (`src/classifiers/mod.rs`).
3. Document its `id()`, any `[classifiers.<id>]` settings it reads via
   `ctx.settings`, and the tags it can produce.

More sophisticated classifiers — e.g. an actual vision/audio modality
detector (which would need `canonical::Message` to carry multimodal content
blocks, not just `String`), or an ML-based moderation call — fit the same
trait; `keyword` is deliberately the simplest possible implementation to
build against.

## Response classifiers

`classifiers/mod.rs` also defines a `ResponseClassifier` trait and
`ResponseClassifierRegistry` — the mirror image of `Classifier`, but running
*after* the provider replies (at `Stage::PostResponse`, once plugins have
had a chance to repair/transform the response):

```text
... -> routers -> provider -> plugins (post_response) -> response_classifiers -> plugins (on_end) -> logging
```

```rust
#[async_trait]
pub trait ResponseClassifier: Send + Sync {
    fn id(&self) -> &'static str;

    async fn classify(
        &self,
        ctx: &ClassifierContext,
        req: &ChatRequest,
        resp: &ChatResponse,
    ) -> anyhow::Result<Vec<String>>;
}
```

Enabled via `[response_classifiers.<id>]` in config (same `EntryConfig` shape
as `[classifiers.<id>]`). Tags are merged into `ChatResponse.tags`, kept
separate from the request's classifier tags (`ChatRequest.tags`) — the two
are never combined.

**Neither tag set ever appears in the OpenAI/Anthropic response body** — the
`From<ChatResponse>` impls in `formats/openai.rs` and `formats/anthropic.rs`
build the wire response field-by-field and don't map `tags`, so the JSON
shape clients see is unchanged. Instead, `server.rs` sets six response
headers from a `DispatchOutcome` (see `dispatch()`):

| Header | From | Example |
|---|---|---|
| `X-Router-Request-Tags` | `ChatRequest.tags` | `vision` |
| `X-Router-Response-Tags` | `ChatResponse.tags` | `refusal` |
| `X-Router-Provider` | the provider that handled the request | `openai` |
| `X-Router-Model` | the model actually sent to that provider | `gpt-4o` |
| `X-Router-Input-Tokens` | `ChatResponse.usage.input_tokens` | `583` |
| `X-Router-Output-Tokens` | `ChatResponse.usage.output_tokens` | `42` |

The two tag headers are comma-separated and omitted entirely when empty; the
other four are always present once a response exists. The same data is
recorded in the request log/dashboard feed: request tags under `tags`,
response tags and usage nested under `response.tags`/`response.usage`, and
`provider`/`sent_model` at the top level of each `LogEntry`.

### `refusal`

A simple substring classifier, the response-side analog of `keyword`:
lowercases the response content and checks it against a list of refusal-ish
phrases (`"i cannot help with that"`, `"as an ai language model"`, etc. —
see `DEFAULT_PHRASES` in `src/classifiers/refusal.rs`), tagging `"refusal"`
on a match. Override the phrase list via `[response_classifiers.refusal]`:

```toml
[response_classifiers.refusal]
enabled = true
phrases = ["i cannot help with that", "i can't assist with"]
```

Like `keyword`, this is a baseline, not a real refusal detector — it won't
catch a refusal phrased differently than the configured list. A more
sophisticated implementation (e.g. a small classifier model, or an LLM-judge
call) fits the same trait.

### Writing a new response classifier

1. Add a module under `src/classifiers/`, implement `ResponseClassifier` for
   a unit struct.
2. Register it in `ResponseClassifierRegistry::from_config`
   (`src/classifiers/mod.rs`).
3. Document its `id()`, any `[response_classifiers.<id>]` settings it reads
   via `ctx.settings`, and the tags it can produce.
