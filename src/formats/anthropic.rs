//! Wire types for the Anthropic `/v1/messages` shape, and conversions
//! to/from the [`canonical`](crate::canonical) representation.
//!
//! These types are used both when the router receives a Claude-shaped
//! request from a client, and when it forwards a request to a provider that
//! itself speaks the Anthropic Messages API.

use serde::{Deserialize, Serialize};

use crate::canonical::{ChatRequest, ChatResponse, Message, PluginRequest, Role, StopReason, Usage};

/// `content` may be a plain string or a list of content blocks. Anthropic
/// accepts both shapes on input; only blocks are accepted on output, but we
/// keep this untagged so a single type covers requests in either shape.
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(untagged)]
pub enum AnthropicContent {
    Text(String),
    Blocks(Vec<ContentBlock>),
}

impl AnthropicContent {
    pub fn into_text(self) -> String {
        match self {
            AnthropicContent::Text(text) => text,
            AnthropicContent::Blocks(blocks) => blocks
                .into_iter()
                .filter(|b| b.block_type == "text")
                .map(|b| b.text)
                .collect::<Vec<_>>()
                .join(""),
        }
    }
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ContentBlock {
    #[serde(rename = "type")]
    pub block_type: String,
    #[serde(default)]
    pub text: String,
}

impl ContentBlock {
    pub fn text(text: impl Into<String>) -> Self {
        ContentBlock {
            block_type: "text".to_string(),
            text: text.into(),
        }
    }
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct AnthropicMessage {
    pub role: String,
    pub content: AnthropicContent,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct AnthropicMessagesRequest {
    pub model: String,
    pub max_tokens: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system: Option<String>,
    pub messages: Vec<AnthropicMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    /// Extended-thinking config, e.g. `{"type": "adaptive"}`. Passed through
    /// opaquely — the router doesn't interpret it.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub thinking: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output_config: Option<OutputConfig>,
    #[serde(default)]
    pub stream: bool,
    /// Plugins to run for this request, e.g. `[{"id": "response-healing"}]`.
    /// Not part of the standard Anthropic API; stripped before forwarding.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub plugins: Vec<PluginRequest>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct OutputConfig {
    /// `"low"`, `"medium"`, `"high"`, `"xhigh"`, or `"max"`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub effort: Option<String>,
    /// e.g. `{"type": "tokens", "total": 64000}`. Requires the
    /// `task-budgets-2026-03-13` beta header — see
    /// `provider::anthropic_beta_header`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub task_budget: Option<serde_json::Value>,
    /// Structured-outputs config: `{"type": "json_schema", "schema": {...}}`.
    /// See <https://platform.claude.com/docs/en/build-with-claude/structured-outputs>.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub format: Option<OutputFormat>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct OutputFormat {
    #[serde(rename = "type")]
    pub format_type: String,
    pub schema: serde_json::Value,
}

#[derive(Debug, Clone, Deserialize, Serialize, Default)]
pub struct AnthropicUsage {
    #[serde(default)]
    pub input_tokens: u32,
    #[serde(default)]
    pub output_tokens: u32,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct AnthropicMessagesResponse {
    pub id: String,
    #[serde(rename = "type")]
    pub response_type: String,
    pub role: String,
    pub model: String,
    pub content: Vec<ContentBlock>,
    pub stop_reason: Option<String>,
    #[serde(default)]
    pub usage: AnthropicUsage,
}

/// Fallback `max_tokens` for requests where the client (or our own
/// translation from an OpenAI-shaped request) didn't specify one, since
/// Anthropic's API requires the field.
const DEFAULT_MAX_TOKENS: u32 = 4096;

/// An inbound request from a client speaking the Anthropic format.
impl From<AnthropicMessagesRequest> for ChatRequest {
    fn from(req: AnthropicMessagesRequest) -> Self {
        let messages = req
            .messages
            .into_iter()
            .map(|msg| Message {
                role: match msg.role.as_str() {
                    "assistant" => Role::Assistant,
                    _ => Role::User,
                },
                content: msg.content.into_text(),
            })
            .collect();

        let (effort, task_budget, output_schema) = match req.output_config {
            Some(c) => (c.effort, c.task_budget, c.format.map(|f| f.schema)),
            None => (None, None, None),
        };

        ChatRequest {
            model: req.model,
            system: req.system,
            messages,
            max_tokens: Some(req.max_tokens),
            temperature: req.temperature,
            thinking: req.thinking,
            effort,
            task_budget,
            output_schema,
            stream: req.stream,
            plugins: req.plugins,
            forced_provider: None,
            tags: Vec::new(),
        }
    }
}

/// An outbound request to a provider that speaks the Anthropic format.
impl From<&ChatRequest> for AnthropicMessagesRequest {
    fn from(req: &ChatRequest) -> Self {
        let messages = req
            .messages
            .iter()
            .map(|msg| AnthropicMessage {
                role: match msg.role {
                    Role::Assistant => "assistant".to_string(),
                    Role::User => "user".to_string(),
                },
                content: AnthropicContent::Text(msg.content.clone()),
            })
            .collect();

        let output_config = if req.effort.is_some() || req.task_budget.is_some() || req.output_schema.is_some() {
            Some(OutputConfig {
                effort: req.effort.clone(),
                task_budget: req.task_budget.clone(),
                format: req.output_schema.clone().map(|schema| OutputFormat {
                    format_type: "json_schema".to_string(),
                    schema,
                }),
            })
        } else {
            None
        };

        AnthropicMessagesRequest {
            model: req.model.clone(),
            max_tokens: req.max_tokens.unwrap_or(DEFAULT_MAX_TOKENS),
            system: req.system.clone(),
            messages,
            temperature: req.temperature,
            thinking: req.thinking.clone(),
            output_config,
            stream: false,
            plugins: Vec::new(),
        }
    }
}

/// A reply from a provider that speaks the Anthropic format.
impl From<AnthropicMessagesResponse> for ChatResponse {
    fn from(resp: AnthropicMessagesResponse) -> Self {
        let content = resp
            .content
            .into_iter()
            .filter(|b| b.block_type == "text")
            .map(|b| b.text)
            .collect::<Vec<_>>()
            .join("");

        let stop_reason = match resp.stop_reason.as_deref() {
            Some("end_turn") => StopReason::EndTurn,
            Some("max_tokens") => StopReason::MaxTokens,
            _ => StopReason::Other,
        };

        ChatResponse {
            id: resp.id,
            model: resp.model,
            content,
            stop_reason,
            usage: Usage {
                input_tokens: resp.usage.input_tokens,
                output_tokens: resp.usage.output_tokens,
            },
            tags: Vec::new(),
        }
    }
}

/// A reply rendered for a client that speaks the Anthropic format.
impl From<ChatResponse> for AnthropicMessagesResponse {
    fn from(resp: ChatResponse) -> Self {
        let stop_reason = match resp.stop_reason {
            StopReason::EndTurn => "end_turn",
            StopReason::MaxTokens => "max_tokens",
            StopReason::Other => "end_turn",
        };

        AnthropicMessagesResponse {
            id: resp.id,
            response_type: "message".to_string(),
            role: "assistant".to_string(),
            model: resp.model,
            content: vec![ContentBlock::text(resp.content)],
            stop_reason: Some(stop_reason.to_string()),
            usage: AnthropicUsage {
                input_tokens: resp.usage.input_tokens,
                output_tokens: resp.usage.output_tokens,
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn anthropic_request(
        thinking: Option<serde_json::Value>,
        effort: Option<&str>,
        task_budget: Option<serde_json::Value>,
        output_schema: Option<serde_json::Value>,
    ) -> AnthropicMessagesRequest {
        let output_config = if effort.is_some() || task_budget.is_some() || output_schema.is_some() {
            Some(OutputConfig {
                effort: effort.map(str::to_string),
                task_budget,
                format: output_schema.map(|schema| OutputFormat {
                    format_type: "json_schema".to_string(),
                    schema,
                }),
            })
        } else {
            None
        };

        AnthropicMessagesRequest {
            model: "claude-opus-4-8".to_string(),
            max_tokens: 1024,
            system: None,
            messages: vec![AnthropicMessage {
                role: "user".to_string(),
                content: AnthropicContent::Text("hi".to_string()),
            }],
            temperature: None,
            thinking,
            output_config,
            stream: false,
            plugins: Vec::new(),
        }
    }

    fn chat_request(
        thinking: Option<serde_json::Value>,
        effort: Option<&str>,
        task_budget: Option<serde_json::Value>,
        output_schema: Option<serde_json::Value>,
    ) -> ChatRequest {
        ChatRequest {
            model: "claude-opus-4-8".to_string(),
            system: None,
            messages: vec![Message {
                role: Role::User,
                content: "hi".to_string(),
            }],
            max_tokens: Some(1024),
            temperature: None,
            thinking,
            effort: effort.map(str::to_string),
            task_budget,
            output_schema,
            stream: false,
            plugins: Vec::new(),
            forced_provider: None,
            tags: Vec::new(),
        }
    }

    #[test]
    fn inbound_thinking_and_effort_map_to_chat_request() {
        let req = anthropic_request(Some(json!({"type": "adaptive"})), Some("xhigh"), None, None);
        let chat: ChatRequest = req.into();
        assert_eq!(chat.thinking, Some(json!({"type": "adaptive"})));
        assert_eq!(chat.effort, Some("xhigh".to_string()));
        assert_eq!(chat.task_budget, None);
    }

    #[test]
    fn inbound_without_thinking_or_effort_leaves_both_none() {
        let req = anthropic_request(None, None, None, None);
        let chat: ChatRequest = req.into();
        assert_eq!(chat.thinking, None);
        assert_eq!(chat.effort, None);
        assert_eq!(chat.task_budget, None);
        assert_eq!(chat.output_schema, None);
    }

    #[test]
    fn inbound_task_budget_maps_to_chat_request_without_effort() {
        let budget = json!({"type": "tokens", "total": 64000});
        let req = anthropic_request(None, None, Some(budget.clone()), None);
        let chat: ChatRequest = req.into();
        assert_eq!(chat.task_budget, Some(budget));
        assert_eq!(chat.effort, None);
    }

    #[test]
    fn inbound_output_schema_extracted_from_json_schema_format() {
        let schema = json!({"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"], "additionalProperties": false});
        let req = anthropic_request(None, None, None, Some(schema.clone()));
        let chat: ChatRequest = req.into();
        assert_eq!(chat.output_schema, Some(schema));
    }

    #[test]
    fn outbound_thinking_and_effort_forwarded_under_output_config() {
        let chat = chat_request(Some(json!({"type": "adaptive"})), Some("high"), None, None);
        let req = AnthropicMessagesRequest::from(&chat);
        assert_eq!(req.thinking, Some(json!({"type": "adaptive"})));
        assert_eq!(req.output_config.unwrap().effort, Some("high".to_string()));
    }

    #[test]
    fn outbound_output_schema_wrapped_as_json_schema_format() {
        let schema = json!({"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"], "additionalProperties": false});
        let chat = chat_request(None, None, None, Some(schema.clone()));
        let req = AnthropicMessagesRequest::from(&chat);
        let format = req.output_config.unwrap().format.unwrap();
        assert_eq!(format.format_type, "json_schema");
        assert_eq!(format.schema, schema);
    }

    #[test]
    fn outbound_without_effort_or_task_budget_omits_output_config() {
        let chat = chat_request(None, None, None, None);
        let req = AnthropicMessagesRequest::from(&chat);
        assert_eq!(req.thinking, None);
        assert!(req.output_config.is_none());
    }

    #[test]
    fn outbound_task_budget_alone_still_creates_output_config() {
        let budget = json!({"type": "tokens", "total": 64000});
        let chat = chat_request(None, None, Some(budget.clone()), None);
        let req = AnthropicMessagesRequest::from(&chat);
        let output_config = req.output_config.unwrap();
        assert_eq!(output_config.task_budget, Some(budget));
        assert_eq!(output_config.effort, None);
    }

    #[test]
    fn outbound_request_serializes_effort_task_budget_and_format_under_output_config_key() {
        let budget = json!({"type": "tokens", "total": 64000});
        let schema = json!({"type": "object", "properties": {}, "additionalProperties": false});
        let chat = chat_request(Some(json!({"type": "adaptive"})), Some("high"), Some(budget.clone()), Some(schema.clone()));
        let req = AnthropicMessagesRequest::from(&chat);
        let value = serde_json::to_value(&req).unwrap();
        assert_eq!(value["thinking"], json!({"type": "adaptive"}));
        assert_eq!(
            value["output_config"],
            json!({"effort": "high", "task_budget": budget, "format": {"type": "json_schema", "schema": schema}})
        );
    }
}
