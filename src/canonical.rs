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

/// One piece of a [`Message`]'s content. A message can mix multiple parts,
/// e.g. an assistant turn requesting a tool call alongside explanatory text.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ContentPart {
    Text {
        text: String,
    },
    /// `data` is base64-encoded, with no `data:` URL prefix.
    Image {
        media_type: String,
        data: String,
    },
    /// An assistant requesting a tool call.
    ToolUse {
        id: String,
        name: String,
        input: serde_json::Value,
    },
    /// The result of a tool call, supplied back as part of a user turn.
    ToolResult {
        tool_use_id: String,
        content: String,
    },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Message {
    pub role: Role,
    pub content: Vec<ContentPart>,
}

impl Message {
    /// Builds a message containing a single text part — the common case.
    /// Only exercised by tests today; kept as the canonical way to build a
    /// plain-text message rather than constructing `ContentPart::Text` by hand.
    #[allow(dead_code)]
    pub fn text(role: Role, text: impl Into<String>) -> Self {
        Message {
            role,
            content: vec![ContentPart::Text { text: text.into() }],
        }
    }

    /// Joins every `Text` part's text, skipping images/tool parts. Used by
    /// anything that only cares about the textual content of a message,
    /// e.g. keyword classification.
    pub fn text_content(&self) -> String {
        self.content
            .iter()
            .filter_map(|part| match part {
                ContentPart::Text { text } => Some(text.as_str()),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join("")
    }
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
    /// Tools the model may call. Translated into each provider's native
    /// tool/function-calling mechanism.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tools: Vec<Tool>,
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

impl ChatRequest {
    /// Capabilities this request structurally needs from whichever model
    /// handles it — `"vision"` if any message contains an image, `"tools"`
    /// if `tools` is non-empty. Consulted by [`crate::router::ModelRouter::resolve`]
    /// to skip a candidate model that doesn't have one of these. Deliberately
    /// structural (inspects `ContentPart`/`tools` directly) rather than the
    /// keyword classifier, which only scans prose and explicitly ignores
    /// non-text content.
    pub fn needed_capabilities(&self) -> Vec<String> {
        let mut needed = Vec::new();
        if self
            .messages
            .iter()
            .any(|m| m.content.iter().any(|p| matches!(p, ContentPart::Image { .. })))
        {
            needed.push("vision".to_string());
        }
        if !self.tools.is_empty() {
            needed.push("tools".to_string());
        }
        needed
    }
}

/// A tool the model may call, described by name and a JSON Schema for its
/// parameters.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Tool {
    pub name: String,
    pub description: Option<String>,
    pub input_schema: serde_json::Value,
}

/// A tool invocation requested by the model, surfaced on [`ChatResponse`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCall {
    pub id: String,
    pub name: String,
    pub input: serde_json::Value,
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
    ToolUse,
    Other,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
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
    /// Tool calls the model requested, if any. Empty for a plain text reply.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tool_calls: Vec<ToolCall>,
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

/// One incremental update from a streaming provider call, format-agnostic so
/// a single upstream stream can be rendered into OpenAI, Anthropic, or
/// Responses-API SSE depending on which client endpoint was hit — see
/// `formats::openai::render_stream`, `formats::anthropic::render_stream`,
/// `formats::responses::render_stream`.
#[derive(Debug, Clone, PartialEq)]
pub enum StreamEvent {
    TextDelta {
        text: String,
    },
    /// Opens a new tool call. `id` is the upstream's own call id if it
    /// provides one (OpenAI does), otherwise a router-synthesized id.
    ToolCallStart {
        id: String,
        name: String,
    },
    /// A fragment of the tool call's JSON-arguments text, not a parsed
    /// `Value` — both OpenAI and Anthropic stream arguments as raw string
    /// fragments, so renderers re-emit them as-is.
    ToolCallDelta {
        id: String,
        partial_input: String,
    },
    Done {
        stop_reason: StopReason,
        usage: Usage,
    },
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn text_part_serializes_as_tagged_shape() {
        let part = ContentPart::Text { text: "hi".to_string() };
        assert_eq!(serde_json::to_value(&part).unwrap(), json!({"type": "text", "text": "hi"}));
    }

    #[test]
    fn image_part_serializes_as_tagged_shape() {
        let part = ContentPart::Image { media_type: "image/png".to_string(), data: "abc123".to_string() };
        assert_eq!(
            serde_json::to_value(&part).unwrap(),
            json!({"type": "image", "media_type": "image/png", "data": "abc123"})
        );
    }

    #[test]
    fn tool_use_part_serializes_as_tagged_shape() {
        let part = ContentPart::ToolUse {
            id: "call_1".to_string(),
            name: "get_weather".to_string(),
            input: json!({"city": "nyc"}),
        };
        assert_eq!(
            serde_json::to_value(&part).unwrap(),
            json!({"type": "tool_use", "id": "call_1", "name": "get_weather", "input": {"city": "nyc"}})
        );
    }

    #[test]
    fn tool_result_part_serializes_as_tagged_shape() {
        let part = ContentPart::ToolResult { tool_use_id: "call_1".to_string(), content: "sunny".to_string() };
        assert_eq!(
            serde_json::to_value(&part).unwrap(),
            json!({"type": "tool_result", "tool_use_id": "call_1", "content": "sunny"})
        );
    }

    #[test]
    fn content_part_round_trips_through_serde() {
        for part in [
            ContentPart::Text { text: "hi".to_string() },
            ContentPart::Image { media_type: "image/png".to_string(), data: "abc".to_string() },
            ContentPart::ToolUse { id: "1".to_string(), name: "f".to_string(), input: json!({}) },
            ContentPart::ToolResult { tool_use_id: "1".to_string(), content: "r".to_string() },
        ] {
            let value = serde_json::to_value(&part).unwrap();
            let back: ContentPart = serde_json::from_value(value).unwrap();
            assert_eq!(part, back);
        }
    }

    #[test]
    fn text_constructor_produces_single_text_part() {
        let msg = Message::text(Role::User, "hi");
        assert_eq!(msg.role, Role::User);
        assert_eq!(msg.content, vec![ContentPart::Text { text: "hi".to_string() }]);
    }

    #[test]
    fn text_content_joins_multiple_text_parts() {
        let msg = Message {
            role: Role::User,
            content: vec![
                ContentPart::Text { text: "hello ".to_string() },
                ContentPart::Text { text: "world".to_string() },
            ],
        };
        assert_eq!(msg.text_content(), "hello world");
    }

    #[test]
    fn text_content_skips_non_text_parts() {
        let msg = Message {
            role: Role::User,
            content: vec![
                ContentPart::Text { text: "describe this".to_string() },
                ContentPart::Image { media_type: "image/png".to_string(), data: "abc".to_string() },
                ContentPart::ToolUse { id: "1".to_string(), name: "f".to_string(), input: json!({}) },
                ContentPart::ToolResult { tool_use_id: "1".to_string(), content: "r".to_string() },
            ],
        };
        assert_eq!(msg.text_content(), "describe this");
    }

    fn base_request(messages: Vec<Message>) -> ChatRequest {
        ChatRequest {
            model: "test-model".to_string(),
            system: None,
            messages,
            max_tokens: None,
            temperature: None,
            thinking: None,
            effort: None,
            task_budget: None,
            output_schema: None,
            tools: Vec::new(),
            stream: false,
            plugins: Vec::new(),
            forced_provider: None,
            tags: Vec::new(),
        }
    }

    #[test]
    fn needed_capabilities_empty_for_plain_text_request() {
        let req = base_request(vec![Message::text(Role::User, "hi")]);
        assert_eq!(req.needed_capabilities(), Vec::<String>::new());
    }

    #[test]
    fn needed_capabilities_includes_vision_for_image_content() {
        let req = base_request(vec![Message {
            role: Role::User,
            content: vec![ContentPart::Image { media_type: "image/png".to_string(), data: "abc".to_string() }],
        }]);
        assert_eq!(req.needed_capabilities(), vec!["vision".to_string()]);
    }

    #[test]
    fn needed_capabilities_includes_tools_for_non_empty_tools() {
        let mut req = base_request(vec![Message::text(Role::User, "hi")]);
        req.tools = vec![Tool {
            name: "get_weather".to_string(),
            description: None,
            input_schema: json!({"type": "object"}),
        }];
        assert_eq!(req.needed_capabilities(), vec!["tools".to_string()]);
    }

    #[test]
    fn needed_capabilities_includes_both_when_applicable() {
        let mut req = base_request(vec![Message {
            role: Role::User,
            content: vec![ContentPart::Image { media_type: "image/png".to_string(), data: "abc".to_string() }],
        }]);
        req.tools = vec![Tool {
            name: "get_weather".to_string(),
            description: None,
            input_schema: json!({"type": "object"}),
        }];
        assert_eq!(req.needed_capabilities(), vec!["vision".to_string(), "tools".to_string()]);
    }
}
