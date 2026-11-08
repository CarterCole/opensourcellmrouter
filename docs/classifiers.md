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
config runs, all at once, and their tags are merged (de-duplicated, in
registry order) into `ChatRequest.tags`. A classifier that returns an error is
logged as a warning and skipped — a failed classifier never fails the
request, it just means that classifier's tags are missing.

A classifier can also pick the model to route to outright, by returning a
`route` (see [`cev`](#cev) and [`llm`](#llm)). Every pick is kept in
`ChatRequest.routes`, and a [`cev`](routers.md#cev) or
[`llm`](routers.md#llm) router rule acts on its own classifier's pick. Both
can only be enabled together with `mode = "teacher"` on `llm`
([below](#learning-from-llm)); then the first of those rules in `routers`
decides.

`ChatRequest.tags` and `ChatRequest.routes` are not forwarded to providers,
but are recorded in the request log (`tags` and `routes` fields) alongside
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
is one of them plus a probability for each. No text is generated. If cev's
model takes images or audio (see [models](#models)), it also gets those
of the latest user turn.

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
| `model` | | Model to run in-process instead of `url`: a short name such as `d1` or `kev:4b`, a Hugging Face repo id, or a directory. See [models](#models). |
| `db` | `cev.db` | With `model`: SQLite file for decisions, feedback and adapters. |
| `models` | | The options to choose between (at least two). |
| `instructions` | "Which model is best suited to answer this request?" | The question. |
| `task` | `llmrouter.route` | Key of the online adapter that learns from feedback. |
| `timeout_ms` | `5000` | How long to wait for cev before routing without it. |
| `max_chars` | `2000` | How much of the system prompt and of the message to send. |
| `max_images` | `4` | Most images to send, for a model that takes them. `0` sends none. |
| `llm_label_weight` | `1.0` | Weight of a label taken from an [`llm`](#llm) pick, against 1 for a correction. `0` takes none. See [learning from `llm`](#learning-from-llm). |

Each of `models` has a `provider` and `model` to route to, an optional
`description` of what the model is good at (cev sees it next to the name),
and an optional `name` to pick and label it by, which defaults to `model`.

**Server or in-process.** With `url`, picks come from a `cev` server
(`cargo install cev-server --features metal`). With `model`, the router
loads the model itself, which needs a build with the `cev-local` feature:

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

cev decides every request whether or not its pick routes it — with no
`cev` rule in `routers`, or one whose `min_confidence` it falls short of,
or another rule ahead of it — so every request still has a decision id to
correct.

#### Models

`model`, like a cev server's `--model`, can be any model cev runs, named by
a short name, a Hugging Face repo id or a directory. Weights come from the
Hugging Face cache, or are downloaded the first time. All of them take text,
and some take more:

| `model` | Takes | Per pick, through the router on an M4 Max |
|---|---|---|
| `Qwen/Qwen3-*` | text | 0.05 s (`Qwen/Qwen3-0.6B`) |
| `Qwen/Qwen3.5-*` | text, images | 0.4–1 s; 2 s with an image (`Qwen/Qwen3.5-0.8B`) |
| `kev` (`kev:0.8b`, `:4b`, `:9b`) | text | 0.1 s (`kev:0.8b`) |
| `nimble-v3` (`:9b`, `:27b`) | text | 2–2.5 s (`nimble-v3`, 9B) |
| `d3` (`d3:2b`, `:4b`, `:8b`, `:27b`) | text, images | 0.15 s; 2–3 s with an image (`d3:2b`) |
| `d1` (`d1:2b`, `:3b`) | text, images | 0.1–0.35 s; 1.5–2.5 s with an image (`d1:3b`) |
| `d1-omni` (`:600m`, `:2b`) | text, and images or audio | 0.03 s; 0.1 s with an image, 2–3 s with 10 s of audio (`d1-omni`) |

A model that takes images gets the latest user turn's PNG, JPEG, WebP and
GIF images, up to `max_images`; one that takes audio gets its clip, if it's
a WAV (OpenAI's `input_audio` with `"format": "wav"`). d1-omni takes images
or audio but not both, and gets the audio when a turn has both. With `url`,
the router asks the server's `/v1/models` what its model takes the first
time a request has images or audio; a server older than cev 0.2 is sent
text only.

A model's first decision can be slow: d1-omni's first, on a Mac that hadn't
run it before, took 22 s, and later ones a few milliseconds. Loading a model
in-process ends with one decision, so that wait comes at startup rather than
on a request (which `timeout_ms` would route without a pick).

#### Learning from `llm`

Running `cev` and [`llm`](#llm) together takes `mode = "teacher"` under
`[classifiers.llm]`: llm is the teacher, cev the student. With both enabled
and no mode, the router refuses to start, rather than guess which one to
drop; so does `mode = "teacher"` without `cev` enabled, as it would teach
nobody.

```toml
[classifiers.llm]
enabled  = true
provider = "anthropic"
model    = "claude-haiku-5-5"
mode     = "teacher"
# ...and [[classifiers.llm.models]], plus [classifiers.cev] as above.
```

In teacher mode both run on every request, and llm's pick becomes the
label for cev's decision about the same request. cev learns from it as it
would from a correction, so it trains on every request without anyone
clicking anything, and once it predicts the LLM well it can take over the
routing. Which classifier routes is up to `routers`:

```toml
# llm routes; cev only learns.
[[routers]]
type = "llm"

# Or: cev routes what it's sure of and llm the rest. llm is still asked
# about every request, and labels all of them.
[[routers]]
type = "cev"
min_confidence = 0.8

[[routers]]
type = "llm"
```

An llm pick labels the cev option with the same `provider` and `model`, so
the two lists of `models` can name and describe them differently but should
route to the same places; the router warns at startup about an llm option
cev has no match for, whose picks label nothing. Labels go to cev in the
background, so routing doesn't wait for them.

A correction sent to `/v1/feedback` (or clicked on the dashboard, which
shows whether cev agreed with llm) replaces llm's label in what cev learns
from. cev keeps both with the decision, each with a `metadata.source` of
`"llm"` (plus the `provider` and `model` that were asked) or `"feedback"`,
which makes the decisions a training set for cev itself. Its export has a
row per labeled decision: the exact prompt cev saw, what it answered, and
the latest label, so the correction where there is one:

```bash
curl 'localhost:8081/v1/export?task=llmrouter.route' > labels.jsonl
```

Filter on `metadata.source` to train on corrections alone. Every label a
decision has is at `GET /v1/decisions/<id>` on the cev server.

`llm_label_weight` sets how much an llm label counts against a correction's
1; set it to `0` to have cev learn only from corrections while both still
run. With `model` instead of `url`, the same decisions and labels are in
the `db` file.

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
| `mode` | `route` | `teacher` to run alongside the [`cev` classifier](#cev) and [train it](#learning-from-llm). Both enabled without it, or it without `cev`, is a startup error. |

An option on a provider whose `api_key_env` isn't set is dropped with a
warning, the same way the router skips that provider. If the answer
doesn't name exactly one option, the call fails, or it takes longer than
`timeout_ms`, the request is routed without a pick.

**Compared with `cev`.** No second server or GPU, and the judgment of
whichever model you pick — but every request waits for a full chat
completion (and, for a cloud model, pays for one) before it's routed. A pick
has no probabilities (`confidence` is always 1), and no decision id of its
own, so on its own there are no `x-cev-*` headers and nothing to correct.
Enable `cev` alongside it, with `mode = "teacher"`, and each llm pick
[labels cev's decision](#learning-from-llm) instead, which cev learns from
while llm keeps routing; corrections then go to cev's decision. Without
the mode the router won't start with both enabled.

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

More sophisticated classifiers — e.g. a modality detector reading the
`Image` and `Audio` parts of `canonical::Message`, as `cev` does, or an
ML-based moderation call — fit the same trait; `keyword` is deliberately
the simplest possible implementation to build against.

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
