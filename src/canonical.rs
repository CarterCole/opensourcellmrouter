//! Internal representation that every wire format is translated to/from.
//!
//! Inbound requests (OpenAI or Anthropic shaped) are converted into a
//! [`ChatRequest`], dispatched to a provider (itself OpenAI or Anthropic
//! shaped), and the provider's reply is converted back into a
//! [`ChatResponse`] before being rendered in whichever format the caller
//! used.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    User,
    Assistant,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    pub role: Role,
    pub content: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatRequest {
    pub model: String,
    /// System prompt, kept separate from `messages` since Anthropic models
    /// it as a top-level field while OpenAI models it as a message.
    pub system: Option<String>,
    pub messages: Vec<Message>,
    pub max_tokens: Option<u32>,
    pub temperature: Option<f32>,
    /// Anthropic-only extended-thinking config (e.g. `{"type": "adaptive"}`),
    /// passed through opaquely. Ignored by OpenAI/Ollama providers, which
    /// have no equivalent.
    pub thinking: Option<serde_json::Value>,
    /// Anthropic-only `output_config.effort` value (`"low"`..`"max"`).
    /// Ignored by OpenAI/Ollama providers.
    pub effort: Option<String>,
    /// Anthropic-only `output_config.task_budget` (e.g.
    /// `{"type": "tokens", "total": 64000}`), passed through opaquely.
    /// Forwarding this requires the `task-budgets-2026-03-13` beta header on
    /// the outbound request — see `provider::anthropic_beta_header`. Ignored
    /// by OpenAI/Ollama providers.
    pub task_budget: Option<serde_json::Value>,
    /// JSON Schema the response must conform to ("structured outputs").
    /// Unlike `thinking`/`effort`/`task_budget` this is *not* Anthropic-only —
    /// it's translated into each provider's native mechanism: Anthropic
    /// `output_config.format`, OpenAI `response_format.json_schema.schema`,
    /// Ollama's top-level `format`. See `docs/structured-outputs.md`.
    pub output_schema: Option<serde_json::Value>,
    pub stream: bool,
    /// Plugins requested for this call, e.g. `{"id": "response-healing"}`.
    /// Not forwarded to providers.
    #[serde(default)]
    pub plugins: Vec<PluginRequest>,
    /// Set by a plugin (e.g. `pareto-router`) to force routing to a
    /// specific provider by name, bypassing the `routers` chain.
    #[serde(default, skip_serializing)]
    pub forced_provider: Option<String>,
    /// Tags assigned by [`crate::classifiers`] before routing, e.g.
    /// `"vision"`, `"nsfw"`, `"tools"`. Consumed by `routers` rules such as
    /// [`crate::config::RouterRule::Tag`]. Not forwarded to providers.
    #[serde(default, skip_serializing)]
    pub tags: Vec<String>,
}

/// One entry of a request's `plugins` array: `{"id": "<plugin-id>", ...settings}`.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PluginRequest {
    pub id: String,
    #[serde(flatten)]
    pub settings: serde_json::Map<String, serde_json::Value>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StopReason {
    EndTurn,
    MaxTokens,
    Other,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Usage {
    pub input_tokens: u32,
    pub output_tokens: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatResponse {
    pub id: String,
    pub model: String,
    pub content: String,
    pub stop_reason: StopReason,
    pub usage: Usage,
    /// Tags assigned by [`crate::classifiers`]' response classifiers (e.g.
    /// `"refusal"`), after the provider has replied. Kept separate from the
    /// request's classifier tags (`ChatRequest.tags`) — see
    /// [`crate::server::dispatch`], which surfaces each as its own
    /// `X-Router-*-Tags` response header rather than merging them. Never
    /// mapped into the OpenAI/Anthropic wire response — see
    /// `formats::openai`/`formats::anthropic`, whose `From<ChatResponse>`
    /// impls don't map this field — so the response body stays unmodified.
    #[serde(default)]
    pub tags: Vec<String>,
}
