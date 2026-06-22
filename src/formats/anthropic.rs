//! Wire types for the Anthropic `/v1/messages` shape, and conversions
//! to/from the [`canonical`](crate::canonical) representation.
//!
//! These types are used both when the router receives a Claude-shaped
//! request from a client, and when it forwards a request to a provider that
//! itself speaks the Anthropic Messages API.

use bytes::Bytes;
use futures_core::Stream;
use serde::{Deserialize, Serialize};
use serde_json::json;
use tokio_stream::StreamExt;
use tokio_stream::wrappers::ReceiverStream;

use crate::canonical::{
    ChatRequest, ChatResponse, ContentPart, Message, PluginRequest, Role, StopReason, StreamEvent,
    Tool, ToolCall, Usage,
};

/// `content` may be a plain string or a list of content blocks. Anthropic
/// accepts both shapes on input; only blocks are accepted on output, but we
/// keep this untagged so a single type covers requests in either shape.
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(untagged)]
pub enum AnthropicContent {
    Text(String),
    Blocks(Vec<ContentBlock>),
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ContentBlock {
    Text { text: String },
    Image { source: ImageSource },
    ToolUse { id: String, name: String, input: serde_json::Value },
    ToolResult { tool_use_id: String, content: String },
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ImageSource {
    #[serde(rename = "type")]
    pub source_type: String,
    pub media_type: String,
    pub data: String,
}

fn content_parts_from_anthropic(content: AnthropicContent) -> Vec<ContentPart> {
    match content {
        AnthropicContent::Text(text) => vec![ContentPart::Text { text }],
        AnthropicContent::Blocks(blocks) => blocks
            .into_iter()
            .map(|block| match block {
                ContentBlock::Text { text } => ContentPart::Text { text },
                ContentBlock::Image { source } => {
                    ContentPart::Image { media_type: source.media_type, data: source.data }
                }
                ContentBlock::ToolUse { id, name, input } => ContentPart::ToolUse { id, name, input },
                ContentBlock::ToolResult { tool_use_id, content } => {
                    ContentPart::ToolResult { tool_use_id, content }
                }
            })
            .collect(),
    }
}

fn anthropic_content_from_parts(parts: &[ContentPart]) -> AnthropicContent {
    // Collapse the common "single text part" case back to a bare string, so
    // a no-tools/no-images request produces the same JSON shape as before
    // this type existed.
    if let [ContentPart::Text { text }] = parts {
        return AnthropicContent::Text(text.clone());
    }

    AnthropicContent::Blocks(
        parts
            .iter()
            .map(|part| match part {
                ContentPart::Text { text } => ContentBlock::Text { text: text.clone() },
                ContentPart::Image { media_type, data } => ContentBlock::Image {
                    source: ImageSource {
                        source_type: "base64".to_string(),
                        media_type: media_type.clone(),
                        data: data.clone(),
                    },
                },
                ContentPart::ToolUse { id, name, input } => {
                    ContentBlock::ToolUse { id: id.clone(), name: name.clone(), input: input.clone() }
                }
                ContentPart::ToolResult { tool_use_id, content } => {
                    ContentBlock::ToolResult { tool_use_id: tool_use_id.clone(), content: content.clone() }
                }
            })
            .collect(),
    )
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct AnthropicMessage {
    pub role: String,
    pub content: AnthropicContent,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct AnthropicTool {
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    pub input_schema: serde_json::Value,
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
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tools: Vec<AnthropicTool>,
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
                content: content_parts_from_anthropic(msg.content),
            })
            .collect();

        let (effort, task_budget, output_schema) = match req.output_config {
            Some(c) => (c.effort, c.task_budget, c.format.map(|f| f.schema)),
            None => (None, None, None),
        };

        let tools = req
            .tools
            .into_iter()
            .map(|t| Tool { name: t.name, description: t.description, input_schema: t.input_schema })
            .collect();

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
            tools,
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
                content: anthropic_content_from_parts(&msg.content),
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

        let tools = req
            .tools
            .iter()
            .map(|t| AnthropicTool {
                name: t.name.clone(),
                description: t.description.clone(),
                input_schema: t.input_schema.clone(),
            })
            .collect();

        AnthropicMessagesRequest {
            model: req.model.clone(),
            max_tokens: req.max_tokens.unwrap_or(DEFAULT_MAX_TOKENS),
            system: req.system.clone(),
            messages,
            temperature: req.temperature,
            thinking: req.thinking.clone(),
            output_config,
            tools,
            stream: req.stream,
            plugins: Vec::new(),
        }
    }
}

/// A reply from a provider that speaks the Anthropic format.
impl From<AnthropicMessagesResponse> for ChatResponse {
    fn from(resp: AnthropicMessagesResponse) -> Self {
        let mut content = String::new();
        let mut tool_calls = Vec::new();
        for block in resp.content {
            match block {
                ContentBlock::Text { text } => content.push_str(&text),
                ContentBlock::ToolUse { id, name, input } => tool_calls.push(ToolCall { id, name, input }),
                ContentBlock::Image { .. } | ContentBlock::ToolResult { .. } => {}
            }
        }

        let stop_reason = match resp.stop_reason.as_deref() {
            Some("end_turn") => StopReason::EndTurn,
            Some("max_tokens") => StopReason::MaxTokens,
            Some("tool_use") => StopReason::ToolUse,
            _ => StopReason::Other,
        };

        ChatResponse {
            id: resp.id,
            model: resp.model,
            content,
            stop_reason,
            tool_calls,
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
        let stop_reason = stop_reason_to_anthropic(resp.stop_reason);

        let mut content = Vec::new();
        if !resp.content.is_empty() {
            content.push(ContentBlock::Text { text: resp.content });
        }
        for tc in resp.tool_calls {
            content.push(ContentBlock::ToolUse { id: tc.id, name: tc.name, input: tc.input });
        }

        AnthropicMessagesResponse {
            id: resp.id,
            response_type: "message".to_string(),
            role: "assistant".to_string(),
            model: resp.model,
            content,
            stop_reason: Some(stop_reason.to_string()),
            usage: AnthropicUsage {
                input_tokens: resp.usage.input_tokens,
                output_tokens: resp.usage.output_tokens,
            },
        }
    }
}

/// Shared between the non-streaming response renderer above and
/// [`render_stream`]'s closing `message_delta` event.
fn stop_reason_to_anthropic(reason: StopReason) -> &'static str {
    match reason {
        StopReason::EndTurn => "end_turn",
        StopReason::MaxTokens => "max_tokens",
        StopReason::ToolUse => "tool_use",
        StopReason::Other => "end_turn",
    }
}

/// A content block currently open in the synthesized Anthropic event stream.
/// Anthropic requires every block to be explicitly started and stopped, and
/// only one may be open at a time — tracked here so [`render_stream`] knows
/// when to close the previous block before opening the next.
enum OpenBlock {
    Text,
    ToolUse,
}

/// Renders a canonical stream as the real Anthropic Messages SSE event
/// sequence (`message_start` / `content_block_*` / `message_delta` /
/// `message_stop`) — the shape Claude Code's streaming parser expects.
pub fn render_stream<S>(mut events: S, model: String) -> ReceiverStream<anyhow::Result<Bytes>>
where
    S: Stream<Item = anyhow::Result<StreamEvent>> + Unpin + Send + 'static,
{
    let (tx, rx) = tokio::sync::mpsc::channel(64);

    tokio::spawn(async move {
        let msg_id = format!(
            "msg_{:x}",
            std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos()
        );

        macro_rules! send {
            ($event_name:expr, $data:expr) => {
                if tx.send(Ok(Bytes::from(format!("event: {}\ndata: {}\n\n", $event_name, $data)))).await.is_err() {
                    return;
                }
            };
        }

        send!(
            "message_start",
            json!({
                "type": "message_start",
                "message": {
                    "id": msg_id,
                    "type": "message",
                    "role": "assistant",
                    "model": model,
                    "content": [],
                    "stop_reason": null,
                    "usage": {"input_tokens": 0, "output_tokens": 0},
                },
            })
        );

        let mut block_index: i64 = -1;
        let mut open_block: Option<OpenBlock> = None;

        while let Some(item) = events.next().await {
            let event = match item {
                Ok(e) => e,
                Err(err) => {
                    tx.send(Err(err)).await.ok();
                    return;
                }
            };

            match event {
                StreamEvent::TextDelta { text } => {
                    if !matches!(open_block, Some(OpenBlock::Text)) {
                        if open_block.is_some() {
                            send!("content_block_stop", json!({"type": "content_block_stop", "index": block_index}));
                        }
                        block_index += 1;
                        open_block = Some(OpenBlock::Text);
                        send!(
                            "content_block_start",
                            json!({"type": "content_block_start", "index": block_index, "content_block": {"type": "text", "text": ""}})
                        );
                    }
                    send!(
                        "content_block_delta",
                        json!({"type": "content_block_delta", "index": block_index, "delta": {"type": "text_delta", "text": text}})
                    );
                }
                StreamEvent::ToolCallStart { id, name } => {
                    if open_block.is_some() {
                        send!("content_block_stop", json!({"type": "content_block_stop", "index": block_index}));
                    }
                    block_index += 1;
                    open_block = Some(OpenBlock::ToolUse);
                    send!(
                        "content_block_start",
                        json!({"type": "content_block_start", "index": block_index, "content_block": {"type": "tool_use", "id": id, "name": name, "input": {}}})
                    );
                }
                StreamEvent::ToolCallDelta { partial_input, .. } => {
                    send!(
                        "content_block_delta",
                        json!({"type": "content_block_delta", "index": block_index, "delta": {"type": "input_json_delta", "partial_json": partial_input}})
                    );
                }
                StreamEvent::Done { stop_reason, usage } => {
                    if open_block.is_some() {
                        send!("content_block_stop", json!({"type": "content_block_stop", "index": block_index}));
                    }
                    send!(
                        "message_delta",
                        json!({
                            "type": "message_delta",
                            "delta": {"stop_reason": stop_reason_to_anthropic(stop_reason), "stop_sequence": null},
                            "usage": {"input_tokens": usage.input_tokens, "output_tokens": usage.output_tokens},
                        })
                    );
                    send!("message_stop", json!({"type": "message_stop"}));
                    return;
                }
            }
        }
    });

    ReceiverStream::new(rx)
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
            tools: Vec::new(),
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
            messages: vec![Message::text(Role::User, "hi")],
            max_tokens: Some(1024),
            temperature: None,
            thinking,
            effort: effort.map(str::to_string),
            task_budget,
            output_schema,
            tools: Vec::new(),
            stream: false,
            plugins: Vec::new(),
            forced_provider: None,
            tags: Vec::new(),
        }
    }

    #[test]
    fn inbound_stream_flag_survives_into_chat_request() {
        let mut req = anthropic_request(None, None, None, None);
        req.stream = true;
        let chat: ChatRequest = req.into();
        assert!(chat.stream);
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

    #[test]
    fn inbound_tool_use_and_tool_result_blocks_round_trip() {
        let req = AnthropicMessagesRequest {
            model: "claude-opus-4-8".to_string(),
            max_tokens: 1024,
            system: None,
            messages: vec![
                AnthropicMessage {
                    role: "assistant".to_string(),
                    content: AnthropicContent::Blocks(vec![ContentBlock::ToolUse {
                        id: "toolu_1".to_string(),
                        name: "get_weather".to_string(),
                        input: json!({"city": "nyc"}),
                    }]),
                },
                AnthropicMessage {
                    role: "user".to_string(),
                    content: AnthropicContent::Blocks(vec![ContentBlock::ToolResult {
                        tool_use_id: "toolu_1".to_string(),
                        content: "sunny".to_string(),
                    }]),
                },
            ],
            temperature: None,
            thinking: None,
            output_config: None,
            tools: Vec::new(),
            stream: false,
            plugins: Vec::new(),
        };

        let chat: ChatRequest = req.into();
        assert_eq!(
            chat.messages[0].content,
            vec![ContentPart::ToolUse {
                id: "toolu_1".to_string(),
                name: "get_weather".to_string(),
                input: json!({"city": "nyc"}),
            }]
        );
        assert_eq!(
            chat.messages[1].content,
            vec![ContentPart::ToolResult { tool_use_id: "toolu_1".to_string(), content: "sunny".to_string() }]
        );
    }

    #[test]
    fn inbound_image_block_maps_to_image_content_part() {
        let req = AnthropicMessagesRequest {
            model: "claude-opus-4-8".to_string(),
            max_tokens: 1024,
            system: None,
            messages: vec![AnthropicMessage {
                role: "user".to_string(),
                content: AnthropicContent::Blocks(vec![ContentBlock::Image {
                    source: ImageSource {
                        source_type: "base64".to_string(),
                        media_type: "image/png".to_string(),
                        data: "abc123".to_string(),
                    },
                }]),
            }],
            temperature: None,
            thinking: None,
            output_config: None,
            tools: Vec::new(),
            stream: false,
            plugins: Vec::new(),
        };

        let chat: ChatRequest = req.into();
        assert_eq!(
            chat.messages[0].content,
            vec![ContentPart::Image { media_type: "image/png".to_string(), data: "abc123".to_string() }]
        );
    }

    #[test]
    fn outbound_single_text_part_collapses_to_bare_string() {
        let chat = chat_request(None, None, None, None);
        let req = AnthropicMessagesRequest::from(&chat);
        match &req.messages[0].content {
            AnthropicContent::Text(text) => assert_eq!(text, "hi"),
            AnthropicContent::Blocks(_) => panic!("expected bare string content"),
        }
    }

    #[test]
    fn outbound_tools_list_maps_to_input_schema() {
        let mut chat = chat_request(None, None, None, None);
        chat.tools = vec![Tool {
            name: "get_weather".to_string(),
            description: Some("Looks up weather".to_string()),
            input_schema: json!({"type": "object"}),
        }];
        let req = AnthropicMessagesRequest::from(&chat);
        assert_eq!(req.tools.len(), 1);
        assert_eq!(req.tools[0].name, "get_weather");
        assert_eq!(req.tools[0].input_schema, json!({"type": "object"}));
    }

    #[test]
    fn response_tool_use_block_parsed_into_tool_calls() {
        let resp = AnthropicMessagesResponse {
            id: "1".to_string(),
            response_type: "message".to_string(),
            role: "assistant".to_string(),
            model: "claude-opus-4-8".to_string(),
            content: vec![ContentBlock::ToolUse {
                id: "toolu_1".to_string(),
                name: "get_weather".to_string(),
                input: json!({"city": "nyc"}),
            }],
            stop_reason: Some("tool_use".to_string()),
            usage: AnthropicUsage::default(),
        };

        let chat: ChatResponse = resp.into();
        assert_eq!(chat.stop_reason, StopReason::ToolUse);
        assert_eq!(chat.tool_calls.len(), 1);
        assert_eq!(chat.tool_calls[0].name, "get_weather");
    }

    fn events_stream(
        events: Vec<StreamEvent>,
    ) -> impl Stream<Item = anyhow::Result<StreamEvent>> + Unpin + Send + 'static {
        tokio_stream::iter(events.into_iter().map(Ok))
    }

    /// Splits a concatenated SSE byte stream into `(event_name, data_json)`
    /// pairs, one per `event: ...\ndata: ...\n\n` block.
    async fn render_to_events(events: Vec<StreamEvent>, model: &str) -> Vec<(String, serde_json::Value)> {
        let mut stream = render_stream(events_stream(events), model.to_string());
        let mut out = String::new();
        while let Some(item) = stream.next().await {
            out.push_str(std::str::from_utf8(&item.unwrap()).unwrap());
        }

        out.split("\n\n")
            .filter(|block| !block.is_empty())
            .map(|block| {
                let mut lines = block.lines();
                let event_name = lines.next().unwrap().trim_start_matches("event: ").to_string();
                let data = lines.next().unwrap().trim_start_matches("data: ");
                (event_name, serde_json::from_str(data).unwrap())
            })
            .collect()
    }

    #[tokio::test]
    async fn render_stream_text_only_opens_and_closes_one_text_block() {
        let events = render_to_events(
            vec![
                StreamEvent::TextDelta { text: "hi".to_string() },
                StreamEvent::TextDelta { text: " there".to_string() },
                StreamEvent::Done { stop_reason: StopReason::EndTurn, usage: Usage { input_tokens: 1, output_tokens: 2 } },
            ],
            "claude-opus-4-8",
        )
        .await;

        let names: Vec<&str> = events.iter().map(|(n, _)| n.as_str()).collect();
        assert_eq!(
            names,
            vec![
                "message_start",
                "content_block_start",
                "content_block_delta",
                "content_block_delta",
                "content_block_stop",
                "message_delta",
                "message_stop",
            ]
        );
        assert_eq!(events[1].1["content_block"]["type"], "text");
        assert_eq!(events[2].1["delta"]["text"], "hi");
        assert_eq!(events[3].1["delta"]["text"], " there");
        assert_eq!(events[5].1["delta"]["stop_reason"], "end_turn");
        assert_eq!(events[5].1["usage"]["output_tokens"], 2);
    }

    #[tokio::test]
    async fn render_stream_tool_use_opens_tool_block_with_id_and_name() {
        let events = render_to_events(
            vec![
                StreamEvent::ToolCallStart { id: "toolu_1".to_string(), name: "get_weather".to_string() },
                StreamEvent::ToolCallDelta { id: "toolu_1".to_string(), partial_input: r#"{"city":"nyc"}"#.to_string() },
                StreamEvent::Done { stop_reason: StopReason::ToolUse, usage: Usage::default() },
            ],
            "claude-opus-4-8",
        )
        .await;

        let names: Vec<&str> = events.iter().map(|(n, _)| n.as_str()).collect();
        assert_eq!(
            names,
            vec!["message_start", "content_block_start", "content_block_delta", "content_block_stop", "message_delta", "message_stop"]
        );
        assert_eq!(events[1].1["content_block"]["type"], "tool_use");
        assert_eq!(events[1].1["content_block"]["id"], "toolu_1");
        assert_eq!(events[1].1["content_block"]["name"], "get_weather");
        assert_eq!(events[2].1["delta"]["type"], "input_json_delta");
        assert_eq!(events[2].1["delta"]["partial_json"], r#"{"city":"nyc"}"#);
        assert_eq!(events[4].1["delta"]["stop_reason"], "tool_use");
    }
}
