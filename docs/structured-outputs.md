# Structured outputs

All three wire formats let a client force the model's response into a given
JSON Schema ("structured outputs" / "JSON mode"). The router models this as
a single canonical field, `ChatRequest.output_schema: Option<serde_json::Value>`
(`src/canonical.rs`) — the raw JSON Schema object, with no provider-specific
envelope. Each `formats/*.rs` adapter wraps or unwraps that schema into
whatever shape the wire format actually uses, so a schema set on an
OpenAI-shaped request still applies if the router resolves to an Anthropic
or Ollama provider, and vice versa.

```text
client (any shape) -> ChatRequest.output_schema -> formats/*.rs -> provider (any shape)
```

## Per-format wire shape

| Format | Field | Shape |
|---|---|---|
| Anthropic | `output_config.format` | `{"type": "json_schema", "schema": {...}}` |
| OpenAI | `response_format` | `{"type": "json_schema", "json_schema": {"name": "response", "schema": {...}, "strict": true}}` |
| Ollama | `format` (top-level) | the JSON Schema object directly, or the literal string `"json"` |

- **Anthropic**: `OutputConfig.format` (`src/formats/anthropic.rs`) — a sibling
  of the `effort`/`task_budget` fields under the same `output_config` object.
- **OpenAI**: `OpenAiChatRequest.response_format` (`src/formats/openai.rs`).
  The router always sends `strict: true` and a fixed `name: "response"` on
  outbound requests — neither is meaningful to the router itself, and OpenAI
  requires both to be present for the constraint to take effect.
- **Ollama**: `OllamaChatRequest.format` (`src/formats/ollama.rs`). Ollama
  also accepts the literal string `"json"` for "valid JSON, any shape" — the
  router doesn't generate that itself, but a client request that already
  resolves to a bare `"json"` value (rather than a schema object) would still
  round-trip correctly, since `output_schema` is stored as an untyped
  `serde_json::Value`.

Inbound extraction and outbound construction both happen in the relevant
`From<...>` impl in each `formats/*.rs` file — see
`formats::anthropic::tests`, `formats::openai::tests`, and
`formats::ollama::tests` for the exact round-trip behavior.

## Using it

Send the schema in whichever wire shape your client already speaks; the
router translates it before forwarding.

**OpenAI-shaped** (`POST /v1/chat/completions`):

```json
{
  "model": "gpt-4o",
  "messages": [{"role": "user", "content": "Extract the contact info"}],
  "response_format": {
    "type": "json_schema",
    "json_schema": {
      "name": "contact",
      "schema": {
        "type": "object",
        "properties": {
          "name": {"type": "string"},
          "email": {"type": "string"}
        },
        "required": ["name", "email"],
        "additionalProperties": false
      },
      "strict": true
    }
  }
}
```

**Anthropic-shaped** (`POST /v1/messages`):

```json
{
  "model": "claude-opus-4-8",
  "max_tokens": 1024,
  "messages": [{"role": "user", "content": "Extract the contact info"}],
  "output_config": {
    "format": {
      "type": "json_schema",
      "schema": {
        "type": "object",
        "properties": {
          "name": {"type": "string"},
          "email": {"type": "string"}
        },
        "required": ["name", "email"],
        "additionalProperties": false
      }
    }
  }
}
```

Either request, once routed, produces the equivalent shape for whichever
provider format handles it — including Ollama, even though there's no
Ollama-shaped client endpoint to send one from directly.

## Schema compatibility

The router does **not** validate or rewrite the schema you send — it passes
it through as-is to whichever provider format you're routed to. Each
provider has its own constraints on what a valid schema looks like (e.g.
Anthropic requires `additionalProperties: false` and rejects numeric/string
length constraints; OpenAI has similar `additionalProperties: false` and
`required`-must-list-everything rules). If you route the same schema across
providers with different constraints, make sure it's compatible with all of
them, or use the `tag`/`prefix` [router rules](routers.md) to keep
schema-constrained requests on a single provider format.

## Provider docs

- Anthropic: <https://platform.claude.com/docs/en/build-with-claude/structured-outputs>
- OpenAI: <https://developers.openai.com/api/docs/guides/structured-outputs>
- Ollama: <https://docs.ollama.com/capabilities/structured-outputs>

## Related: Anthropic-only generation controls

`ChatRequest` also carries a few Anthropic-only passthrough fields that sit
next to `output_schema` under the same `output_config` object on the wire:
`thinking` (extended/adaptive thinking), `effort` (`output_config.effort`),
and `task_budget` (`output_config.task_budget`, requires the
`task-budgets-2026-03-13` beta header — see
`provider::anthropic_beta_header` in `src/provider.rs`). These have no
OpenAI/Ollama equivalent and are dropped silently if the router resolves to
one of those formats. See the doc comments on the corresponding
`ChatRequest` fields in `src/canonical.rs` for details.
