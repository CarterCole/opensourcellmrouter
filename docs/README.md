# Pipeline overview

Every request, whether it arrives on the OpenAI-compatible
`/v1/chat/completions` endpoint or the Anthropic-compatible `/v1/messages`
endpoint, is converted to a canonical `ChatRequest` and run through the same
pipeline:

```text
Start -> classifiers -> PreRouting -> routers -> provider -> PostResponse -> End -> logging
```

1. **[Plugins](plugins.md)** `on_start` hooks run first, on the request as
   the client sent it — before classifiers see it.
2. **[Classifiers](classifiers.md)** inspect the request and attach tags
   (e.g. `"vision"`, `"nsfw"`) to `ChatRequest.tags`.
3. **Plugins** `pre_request` hooks run next, in config/request order. They
   can mutate the request (inject context, force a provider via
   `forced_provider`, etc).
4. **[Routers](routers.md)** pick a provider and (optionally) rewrite the
   model name, by walking an ordered chain of rules. If a plugin set
   `forced_provider`, this step is bypassed entirely.
5. The request is sent to the chosen **[provider](providers.md)**.
6. **Plugins** `post_response` hooks run over the reply (e.g. JSON repair).
7. **Plugins** `on_end` hooks run last, just before logging.
8. If logging is enabled, the whole exchange — including the tags and plugin
   ids involved — is appended to the request log as one line of JSON.

Any plugin hook can short-circuit the rest of the pipeline: a hook returns
`Flow::Continue` to fall through as normal, or `Flow::Stop` to stop. For
`on_start`/`pre_request`, stopping skips classifiers/routing/the provider
call entirely (the hook must have written a response itself — e.g. a
moderation plugin rejecting the request outright). For `post_response`,
stopping skips `on_end`. See [plugins.md](plugins.md#stages-and-flow) for
details.

See each linked doc for its config schema and how to add new rules/plugins/
classifiers.
