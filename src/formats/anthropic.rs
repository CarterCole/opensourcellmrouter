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
    #[serde(default)]
    pub stream: bool,
    /// Plugins to run for this request, e.g. `[{"id": "response-healing"}]`.
    /// Not part of the standard Anthropic API; stripped before forwarding.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub plugins: Vec<PluginRequest>,
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

        ChatRequest {
            model: req.model,
            system: req.system,
            messages,
            max_tokens: Some(req.max_tokens),
            temperature: req.temperature,
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

        AnthropicMessagesRequest {
            model: req.model.clone(),
            max_tokens: req.max_tokens.unwrap_or(DEFAULT_MAX_TOKENS),
            system: req.system.clone(),
            messages,
            temperature: req.temperature,
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
