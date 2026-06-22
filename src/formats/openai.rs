//! Wire types for the OpenAI `/v1/chat/completions` shape, and conversions
//! to/from the [`canonical`](crate::canonical) representation.
//!
//! These types are used both when the router receives an OpenAI-shaped
//! request from a client, and when it forwards a request to a provider that
//! itself speaks the OpenAI format (e.g. a local `llama-server`).

use std::collections::HashMap;

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

/// OpenAI message content: either a bare string, or an array of typed
/// blocks (used for mixed text/image content).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum OpenAiContent {
    Text(String),
    Blocks(Vec<OpenAiContentBlock>),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum OpenAiContentBlock {
    Text { text: String },
    ImageUrl { image_url: OpenAiImageUrl },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenAiImageUrl {
    pub url: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenAiMessage {
    pub role: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub content: Option<OpenAiContent>,
    /// Present on an assistant message that's requesting tool calls.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<OpenAiToolCall>>,
    /// Present on a `role: "tool"` message, identifying which call this is
    /// the result of.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
}

impl OpenAiMessage {
    fn text(role: &str, text: impl Into<String>) -> Self {
        OpenAiMessage {
            role: role.to_string(),
            content: Some(OpenAiContent::Text(text.into())),
            tool_calls: None,
            tool_call_id: None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenAiToolCall {
    pub id: String,
    #[serde(rename = "type")]
    pub call_type: String,
    pub function: OpenAiFunctionCall,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenAiFunctionCall {
    pub name: String,
    /// JSON-encoded arguments object.
    pub arguments: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenAiTool {
    #[serde(rename = "type")]
    pub tool_type: String,
    pub function: OpenAiFunctionDef,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenAiFunctionDef {
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    pub parameters: serde_json::Value,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct OpenAiChatRequest {
    pub model: String,
    pub messages: Vec<OpenAiMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    /// Structured-outputs config: `{"type": "json_schema", "json_schema": {...}}`.
    /// See <https://developers.openai.com/api/docs/guides/structured-outputs>.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response_format: Option<ResponseFormat>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tools: Vec<OpenAiTool>,
    #[serde(default)]
    pub stream: bool,
    /// Plugins to run for this request, e.g. `[{"id": "response-healing"}]`.
    /// Not part of the standard OpenAI API; ignored by upstream providers
    /// since it's stripped before forwarding.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub plugins: Vec<PluginRequest>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ResponseFormat {
    #[serde(rename = "type")]
    pub format_type: String,
    pub json_schema: JsonSchemaFormat,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct JsonSchemaFormat {
    #[serde(default = "default_schema_name")]
    pub name: String,
    pub schema: serde_json::Value,
    #[serde(default)]
    pub strict: bool,
}

fn default_schema_name() -> String {
    "response".to_string()
}

#[derive(Debug, Clone, Deserialize, Serialize, Default)]
pub struct OpenAiUsage {
    #[serde(default)]
    pub prompt_tokens: u32,
    #[serde(default)]
    pub completion_tokens: u32,
    #[serde(default)]
    pub total_tokens: u32,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct OpenAiChoice {
    pub index: u32,
    pub message: OpenAiMessage,
    pub finish_reason: Option<String>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct OpenAiChatResponse {
    pub id: String,
    pub object: String,
    pub model: String,
    pub choices: Vec<OpenAiChoice>,
    #[serde(default)]
    pub usage: OpenAiUsage,
}

/// `data:<media_type>;base64,<data>` -> `(media_type, data)`. Returns `None`
/// for anything else (e.g. a plain `http(s)://` URL) — the canonical
/// `ContentPart::Image` only carries inline base64 data, so a remote-URL
/// image is dropped rather than fetched.
fn parse_data_url(url: &str) -> Option<(String, String)> {
    let rest = url.strip_prefix("data:")?;
    let (media_type, data) = rest.split_once(";base64,")?;
    Some((media_type.to_string(), data.to_string()))
}

fn data_url(media_type: &str, data: &str) -> String {
    format!("data:{media_type};base64,{data}")
}

fn content_parts_from_openai(content: Option<OpenAiContent>) -> Vec<ContentPart> {
    match content {
        None => Vec::new(),
        Some(OpenAiContent::Text(text)) => vec![ContentPart::Text { text }],
        Some(OpenAiContent::Blocks(blocks)) => blocks
            .into_iter()
            .filter_map(|block| match block {
                OpenAiContentBlock::Text { text } => Some(ContentPart::Text { text }),
                OpenAiContentBlock::ImageUrl { image_url } => parse_data_url(&image_url.url)
                    .map(|(media_type, data)| ContentPart::Image { media_type, data }),
            })
            .collect(),
    }
}

fn openai_content_to_text(content: Option<OpenAiContent>) -> String {
    Message {
        role: Role::User,
        content: content_parts_from_openai(content),
    }
    .text_content()
}

/// Renders one canonical [`Message`] as one or more [`OpenAiMessage`]s — a
/// tool result becomes its own `role: "tool"` message, since OpenAI has no
/// inline tool-result content block the way Anthropic does.
fn openai_messages_from_canonical(msg: &Message) -> Vec<OpenAiMessage> {
    let role = match msg.role {
        Role::User => "user",
        Role::Assistant => "assistant",
    };

    let mut blocks = Vec::new();
    let mut has_image = false;
    let mut text_only = String::new();
    let mut tool_calls = Vec::new();
    let mut tool_results = Vec::new();

    for part in &msg.content {
        match part {
            ContentPart::Text { text } => {
                text_only.push_str(text);
                blocks.push(OpenAiContentBlock::Text { text: text.clone() });
            }
            ContentPart::Image { media_type, data } => {
                has_image = true;
                blocks.push(OpenAiContentBlock::ImageUrl {
                    image_url: OpenAiImageUrl { url: data_url(media_type, data) },
                });
            }
            ContentPart::ToolUse { id, name, input } => tool_calls.push(OpenAiToolCall {
                id: id.clone(),
                call_type: "function".to_string(),
                function: OpenAiFunctionCall {
                    name: name.clone(),
                    arguments: input.to_string(),
                },
            }),
            ContentPart::ToolResult { tool_use_id, content } => {
                tool_results.push((tool_use_id.clone(), content.clone()));
            }
        }
    }

    let mut out = Vec::new();
    if !blocks.is_empty() || !tool_calls.is_empty() {
        let content = if has_image {
            Some(OpenAiContent::Blocks(blocks))
        } else if !text_only.is_empty() {
            Some(OpenAiContent::Text(text_only))
        } else {
            None
        };
        out.push(OpenAiMessage {
            role: role.to_string(),
            content,
            tool_calls: (!tool_calls.is_empty()).then_some(tool_calls),
            tool_call_id: None,
        });
    }
    for (tool_use_id, content) in tool_results {
        out.push(OpenAiMessage {
            role: "tool".to_string(),
            content: Some(OpenAiContent::Text(content)),
            tool_calls: None,
            tool_call_id: Some(tool_use_id),
        });
    }
    out
}

/// An inbound request from a client speaking the OpenAI format.
impl From<OpenAiChatRequest> for ChatRequest {
    fn from(req: OpenAiChatRequest) -> Self {
        let mut system = None;
        let mut messages = Vec::with_capacity(req.messages.len());

        for msg in req.messages {
            match msg.role.as_str() {
                "system" => system = Some(openai_content_to_text(msg.content)),
                "assistant" => {
                    let mut content = content_parts_from_openai(msg.content);
                    for tc in msg.tool_calls.into_iter().flatten() {
                        let input = serde_json::from_str(&tc.function.arguments)
                            .unwrap_or(serde_json::Value::Null);
                        content.push(ContentPart::ToolUse {
                            id: tc.id,
                            name: tc.function.name,
                            input,
                        });
                    }
                    messages.push(Message { role: Role::Assistant, content });
                }
                "tool" => messages.push(Message {
                    role: Role::User,
                    content: vec![ContentPart::ToolResult {
                        tool_use_id: msg.tool_call_id.unwrap_or_default(),
                        content: openai_content_to_text(msg.content),
                    }],
                }),
                // Treat anything else (user, ...) as a user turn.
                _ => messages.push(Message {
                    role: Role::User,
                    content: content_parts_from_openai(msg.content),
                }),
            }
        }

        let output_schema = req
            .response_format
            .filter(|f| f.format_type == "json_schema")
            .map(|f| f.json_schema.schema);

        let tools = req
            .tools
            .into_iter()
            .map(|t| Tool {
                name: t.function.name,
                description: t.function.description,
                input_schema: t.function.parameters,
            })
            .collect();

        ChatRequest {
            model: req.model,
            system,
            messages,
            max_tokens: req.max_tokens,
            temperature: req.temperature,
            thinking: None,
            effort: None,
            task_budget: None,
            output_schema,
            tools,
            stream: req.stream,
            plugins: req.plugins,
            forced_provider: None,
            tags: Vec::new(),
        }
    }
}

/// An outbound request to a provider that speaks the OpenAI format.
impl From<&ChatRequest> for OpenAiChatRequest {
    fn from(req: &ChatRequest) -> Self {
        let mut messages = Vec::with_capacity(req.messages.len() + 1);

        if let Some(system) = &req.system {
            messages.push(OpenAiMessage::text("system", system.clone()));
        }

        for msg in &req.messages {
            messages.extend(openai_messages_from_canonical(msg));
        }

        let response_format = req.output_schema.clone().map(|schema| ResponseFormat {
            format_type: "json_schema".to_string(),
            json_schema: JsonSchemaFormat {
                name: default_schema_name(),
                schema,
                strict: true,
            },
        });

        let tools = req
            .tools
            .iter()
            .map(|t| OpenAiTool {
                tool_type: "function".to_string(),
                function: OpenAiFunctionDef {
                    name: t.name.clone(),
                    description: t.description.clone(),
                    parameters: t.input_schema.clone(),
                },
            })
            .collect();

        OpenAiChatRequest {
            model: req.model.clone(),
            messages,
            max_tokens: req.max_tokens,
            temperature: req.temperature,
            response_format,
            tools,
            stream: req.stream,
            plugins: Vec::new(),
        }
    }
}

/// A reply from a provider that speaks the OpenAI format.
impl From<OpenAiChatResponse> for ChatResponse {
    fn from(resp: OpenAiChatResponse) -> Self {
        let choice = resp.choices.into_iter().next();
        let content = choice
            .as_ref()
            .and_then(|c| c.message.content.clone())
            .map(|c| openai_content_to_text(Some(c)))
            .unwrap_or_default();
        let tool_calls = choice
            .as_ref()
            .and_then(|c| c.message.tool_calls.clone())
            .unwrap_or_default()
            .into_iter()
            .map(|tc| ToolCall {
                id: tc.id,
                name: tc.function.name,
                input: serde_json::from_str(&tc.function.arguments).unwrap_or(serde_json::Value::Null),
            })
            .collect();
        let finish_reason = choice.and_then(|c| c.finish_reason);
        let stop_reason = stop_reason_from_finish_reason(finish_reason.as_deref());

        ChatResponse {
            id: resp.id,
            model: resp.model,
            content,
            stop_reason,
            tool_calls,
            usage: Usage {
                input_tokens: resp.usage.prompt_tokens,
                output_tokens: resp.usage.completion_tokens,
            },
            tags: Vec::new(),
        }
    }
}

/// A reply rendered for a client that speaks the OpenAI format.
impl From<ChatResponse> for OpenAiChatResponse {
    fn from(resp: ChatResponse) -> Self {
        let finish_reason = match resp.stop_reason {
            StopReason::EndTurn => "stop",
            StopReason::MaxTokens => "length",
            StopReason::ToolUse => "tool_calls",
            StopReason::Other => "stop",
        };

        let tool_calls = (!resp.tool_calls.is_empty()).then(|| {
            resp.tool_calls
                .into_iter()
                .map(|tc| OpenAiToolCall {
                    id: tc.id,
                    call_type: "function".to_string(),
                    function: OpenAiFunctionCall {
                        name: tc.name,
                        arguments: tc.input.to_string(),
                    },
                })
                .collect()
        });

        OpenAiChatResponse {
            id: resp.id,
            object: "chat.completion".to_string(),
            model: resp.model,
            choices: vec![OpenAiChoice {
                index: 0,
                message: OpenAiMessage {
                    role: "assistant".to_string(),
                    content: Some(OpenAiContent::Text(resp.content)),
                    tool_calls,
                    tool_call_id: None,
                },
                finish_reason: Some(finish_reason.to_string()),
            }],
            usage: OpenAiUsage {
                prompt_tokens: resp.usage.input_tokens,
                completion_tokens: resp.usage.output_tokens,
                total_tokens: resp.usage.input_tokens + resp.usage.output_tokens,
            },
        }
    }
}

/// Shared between the non-streaming response parser above and
/// [`OpenAiStreamDecoder`] below, since both see the same `finish_reason`
/// strings.
fn stop_reason_from_finish_reason(reason: Option<&str>) -> StopReason {
    match reason {
        Some("length") => StopReason::MaxTokens,
        Some("tool_calls") => StopReason::ToolUse,
        Some("stop") => StopReason::EndTurn,
        _ => StopReason::Other,
    }
}

/// One `chat.completion.chunk` SSE payload from an OpenAI-shaped upstream.
#[derive(Debug, Clone, Default, Deserialize)]
pub struct OpenAiStreamChunk {
    #[serde(default)]
    pub choices: Vec<OpenAiStreamChoice>,
    #[serde(default)]
    pub usage: Option<OpenAiUsage>,
}

#[derive(Debug, Clone, Default, Deserialize)]
pub struct OpenAiStreamChoice {
    #[serde(default)]
    pub delta: OpenAiStreamDelta,
    #[serde(default)]
    pub finish_reason: Option<String>,
}

#[derive(Debug, Clone, Default, Deserialize)]
pub struct OpenAiStreamDelta {
    #[serde(default)]
    pub content: Option<String>,
    #[serde(default)]
    pub tool_calls: Option<Vec<OpenAiStreamToolCall>>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct OpenAiStreamToolCall {
    pub index: u32,
    #[serde(default)]
    pub id: Option<String>,
    #[serde(default)]
    pub function: OpenAiStreamFunctionDelta,
}

#[derive(Debug, Clone, Default, Deserialize)]
pub struct OpenAiStreamFunctionDelta {
    #[serde(default)]
    pub name: Option<String>,
    #[serde(default)]
    pub arguments: Option<String>,
}

/// Reassembles OpenAI's per-chunk streaming deltas into canonical
/// [`StreamEvent`]s. Stateful because a tool call's `id` is only present on
/// the chunk that opens it — later chunks identify it by `index` alone, so
/// this tracks `index -> id` across calls to [`Self::decode`].
#[derive(Default)]
pub struct OpenAiStreamDecoder {
    tool_call_ids: HashMap<u32, String>,
}

impl OpenAiStreamDecoder {
    pub fn decode(&mut self, chunk: OpenAiStreamChunk) -> Vec<StreamEvent> {
        let mut events = Vec::new();
        let Some(choice) = chunk.choices.into_iter().next() else {
            return events;
        };

        if let Some(text) = choice.delta.content {
            if !text.is_empty() {
                events.push(StreamEvent::TextDelta { text });
            }
        }

        for tc in choice.delta.tool_calls.into_iter().flatten() {
            let id = match tc.id {
                Some(id) => {
                    self.tool_call_ids.insert(tc.index, id.clone());
                    id
                }
                None => self.tool_call_ids.get(&tc.index).cloned().unwrap_or_default(),
            };
            if let Some(name) = tc.function.name {
                events.push(StreamEvent::ToolCallStart { id: id.clone(), name });
            }
            if let Some(partial_input) = tc.function.arguments {
                if !partial_input.is_empty() {
                    events.push(StreamEvent::ToolCallDelta { id, partial_input });
                }
            }
        }

        if let Some(reason) = choice.finish_reason {
            let usage = chunk
                .usage
                .map(|u| Usage { input_tokens: u.prompt_tokens, output_tokens: u.completion_tokens })
                .unwrap_or_default();
            events.push(StreamEvent::Done {
                stop_reason: stop_reason_from_finish_reason(Some(&reason)),
                usage,
            });
        }

        events
    }
}

/// Renders a canonical stream as OpenAI `chat.completion.chunk` SSE — the
/// shape `/v1/chat/completions` clients (e.g. Copilot CLI) expect, whether
/// the upstream that produced the events was itself OpenAI- or Ollama-shaped.
pub fn render_stream<S>(mut events: S, model: String) -> ReceiverStream<anyhow::Result<Bytes>>
where
    S: Stream<Item = anyhow::Result<StreamEvent>> + Unpin + Send + 'static,
{
    let (tx, rx) = tokio::sync::mpsc::channel(64);

    tokio::spawn(async move {
        let mut chunk_id: u64 = 0;
        let mut tool_indices: HashMap<String, u32> = HashMap::new();
        let mut next_tool_index: u32 = 0;

        while let Some(item) = events.next().await {
            let event = match item {
                Ok(e) => e,
                Err(err) => {
                    tx.send(Err(err)).await.ok();
                    return;
                }
            };

            chunk_id += 1;
            let mut finish_reason: Option<&'static str> = None;
            let delta = match event {
                StreamEvent::TextDelta { text } => json!({"content": text}),
                StreamEvent::ToolCallStart { id, name } => {
                    let index = *tool_indices.entry(id.clone()).or_insert_with(|| {
                        let i = next_tool_index;
                        next_tool_index += 1;
                        i
                    });
                    json!({"tool_calls": [{
                        "index": index,
                        "id": id,
                        "type": "function",
                        "function": {"name": name, "arguments": ""},
                    }]})
                }
                StreamEvent::ToolCallDelta { id, partial_input } => {
                    let index = *tool_indices.entry(id.clone()).or_insert_with(|| {
                        let i = next_tool_index;
                        next_tool_index += 1;
                        i
                    });
                    json!({"tool_calls": [{"index": index, "function": {"arguments": partial_input}}]})
                }
                StreamEvent::Done { stop_reason, .. } => {
                    finish_reason = Some(match stop_reason {
                        StopReason::EndTurn => "stop",
                        StopReason::MaxTokens => "length",
                        StopReason::ToolUse => "tool_calls",
                        StopReason::Other => "stop",
                    });
                    json!({})
                }
            };

            let sse = json!({
                "id": format!("chatcmpl-{chunk_id}"),
                "object": "chat.completion.chunk",
                "model": &model,
                "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
            });

            if tx.send(Ok(Bytes::from(format!("data: {sse}\n\n")))).await.is_err() {
                return;
            }
        }

        tx.send(Ok(Bytes::from("data: [DONE]\n\n"))).await.ok();
    });

    ReceiverStream::new(rx)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn chat_request(output_schema: Option<serde_json::Value>) -> ChatRequest {
        ChatRequest {
            model: "gpt-4o".to_string(),
            system: None,
            messages: vec![Message::text(Role::User, "hi")],
            max_tokens: None,
            temperature: None,
            thinking: None,
            effort: None,
            task_budget: None,
            output_schema,
            tools: Vec::new(),
            stream: false,
            plugins: Vec::new(),
            forced_provider: None,
            tags: Vec::new(),
        }
    }

    #[test]
    fn inbound_response_format_extracts_schema() {
        let schema = json!({"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"], "additionalProperties": false});
        let req = OpenAiChatRequest {
            model: "gpt-4o".to_string(),
            messages: vec![OpenAiMessage::text("user", "hi")],
            max_tokens: None,
            temperature: None,
            response_format: Some(ResponseFormat {
                format_type: "json_schema".to_string(),
                json_schema: JsonSchemaFormat { name: "contact".to_string(), schema: schema.clone(), strict: true },
            }),
            tools: Vec::new(),
            stream: false,
            plugins: Vec::new(),
        };

        let chat: ChatRequest = req.into();
        assert_eq!(chat.output_schema, Some(schema));
    }

    #[test]
    fn inbound_without_response_format_leaves_output_schema_none() {
        let req = OpenAiChatRequest {
            model: "gpt-4o".to_string(),
            messages: vec![OpenAiMessage::text("user", "hi")],
            max_tokens: None,
            temperature: None,
            response_format: None,
            tools: Vec::new(),
            stream: false,
            plugins: Vec::new(),
        };

        let chat: ChatRequest = req.into();
        assert_eq!(chat.output_schema, None);
    }

    #[test]
    fn outbound_output_schema_wrapped_as_strict_json_schema_response_format() {
        let schema = json!({"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"], "additionalProperties": false});
        let chat = chat_request(Some(schema.clone()));
        let req = OpenAiChatRequest::from(&chat);

        let format = req.response_format.unwrap();
        assert_eq!(format.format_type, "json_schema");
        assert_eq!(format.json_schema.schema, schema);
        assert!(format.json_schema.strict);
        assert_eq!(format.json_schema.name, "response");
    }

    #[test]
    fn outbound_without_output_schema_omits_response_format() {
        let chat = chat_request(None);
        let req = OpenAiChatRequest::from(&chat);
        assert!(req.response_format.is_none());
    }

    #[test]
    fn inbound_tool_role_message_maps_to_tool_result_content_part() {
        let req = OpenAiChatRequest {
            model: "gpt-4o".to_string(),
            messages: vec![OpenAiMessage {
                role: "tool".to_string(),
                content: Some(OpenAiContent::Text("42".to_string())),
                tool_calls: None,
                tool_call_id: Some("call_1".to_string()),
            }],
            max_tokens: None,
            temperature: None,
            response_format: None,
            tools: Vec::new(),
            stream: false,
            plugins: Vec::new(),
        };

        let chat: ChatRequest = req.into();
        assert_eq!(chat.messages.len(), 1);
        assert_eq!(chat.messages[0].role, Role::User);
        assert_eq!(
            chat.messages[0].content,
            vec![ContentPart::ToolResult { tool_use_id: "call_1".to_string(), content: "42".to_string() }]
        );
    }

    #[test]
    fn inbound_assistant_tool_calls_become_tool_use_parts() {
        let req = OpenAiChatRequest {
            model: "gpt-4o".to_string(),
            messages: vec![OpenAiMessage {
                role: "assistant".to_string(),
                content: None,
                tool_calls: Some(vec![OpenAiToolCall {
                    id: "call_1".to_string(),
                    call_type: "function".to_string(),
                    function: OpenAiFunctionCall {
                        name: "get_weather".to_string(),
                        arguments: r#"{"city":"nyc"}"#.to_string(),
                    },
                }]),
                tool_call_id: None,
            }],
            max_tokens: None,
            temperature: None,
            response_format: None,
            tools: Vec::new(),
            stream: false,
            plugins: Vec::new(),
        };

        let chat: ChatRequest = req.into();
        assert_eq!(
            chat.messages[0].content,
            vec![ContentPart::ToolUse {
                id: "call_1".to_string(),
                name: "get_weather".to_string(),
                input: json!({"city": "nyc"}),
            }]
        );
    }

    #[test]
    fn inbound_image_url_block_maps_to_image_content_part() {
        let req = OpenAiChatRequest {
            model: "gpt-4o".to_string(),
            messages: vec![OpenAiMessage {
                role: "user".to_string(),
                content: Some(OpenAiContent::Blocks(vec![OpenAiContentBlock::ImageUrl {
                    image_url: OpenAiImageUrl { url: "data:image/png;base64,abc123".to_string() },
                }])),
                tool_calls: None,
                tool_call_id: None,
            }],
            max_tokens: None,
            temperature: None,
            response_format: None,
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
    fn outbound_tool_result_part_becomes_separate_tool_role_message() {
        let mut chat = chat_request(None);
        chat.messages = vec![Message {
            role: Role::User,
            content: vec![ContentPart::ToolResult { tool_use_id: "call_1".to_string(), content: "42".to_string() }],
        }];
        let req = OpenAiChatRequest::from(&chat);

        assert_eq!(req.messages.len(), 1);
        assert_eq!(req.messages[0].role, "tool");
        assert_eq!(req.messages[0].tool_call_id, Some("call_1".to_string()));
    }

    #[test]
    fn outbound_tools_list_maps_to_function_type_wrapper() {
        let mut chat = chat_request(None);
        chat.tools = vec![Tool {
            name: "get_weather".to_string(),
            description: Some("Looks up weather".to_string()),
            input_schema: json!({"type": "object"}),
        }];
        let req = OpenAiChatRequest::from(&chat);

        assert_eq!(req.tools.len(), 1);
        assert_eq!(req.tools[0].tool_type, "function");
        assert_eq!(req.tools[0].function.name, "get_weather");
    }

    #[test]
    fn outbound_without_tools_omits_tools_field() {
        let chat = chat_request(None);
        let req = OpenAiChatRequest::from(&chat);
        let json = serde_json::to_value(&req).unwrap();
        assert!(json.get("tools").is_none());
    }

    #[test]
    fn response_tool_calls_parsed_into_chat_response() {
        let resp = OpenAiChatResponse {
            id: "1".to_string(),
            object: "chat.completion".to_string(),
            model: "gpt-4o".to_string(),
            choices: vec![OpenAiChoice {
                index: 0,
                message: OpenAiMessage {
                    role: "assistant".to_string(),
                    content: None,
                    tool_calls: Some(vec![OpenAiToolCall {
                        id: "call_1".to_string(),
                        call_type: "function".to_string(),
                        function: OpenAiFunctionCall {
                            name: "get_weather".to_string(),
                            arguments: r#"{"city":"nyc"}"#.to_string(),
                        },
                    }]),
                    tool_call_id: None,
                },
                finish_reason: Some("tool_calls".to_string()),
            }],
            usage: OpenAiUsage::default(),
        };

        let chat: ChatResponse = resp.into();
        assert_eq!(chat.stop_reason, StopReason::ToolUse);
        assert_eq!(chat.tool_calls.len(), 1);
        assert_eq!(chat.tool_calls[0].name, "get_weather");
    }

    #[test]
    fn inbound_malformed_tool_call_arguments_falls_back_to_null() {
        let req = OpenAiChatRequest {
            model: "gpt-4o".to_string(),
            messages: vec![OpenAiMessage {
                role: "assistant".to_string(),
                content: None,
                tool_calls: Some(vec![OpenAiToolCall {
                    id: "call_1".to_string(),
                    call_type: "function".to_string(),
                    function: OpenAiFunctionCall {
                        name: "get_weather".to_string(),
                        arguments: "not valid json".to_string(),
                    },
                }]),
                tool_call_id: None,
            }],
            max_tokens: None,
            temperature: None,
            response_format: None,
            tools: Vec::new(),
            stream: false,
            plugins: Vec::new(),
        };

        let chat: ChatRequest = req.into();
        assert_eq!(
            chat.messages[0].content,
            vec![ContentPart::ToolUse {
                id: "call_1".to_string(),
                name: "get_weather".to_string(),
                input: serde_json::Value::Null,
            }]
        );
    }

    #[test]
    fn inbound_non_data_url_image_is_dropped() {
        let req = OpenAiChatRequest {
            model: "gpt-4o".to_string(),
            messages: vec![OpenAiMessage {
                role: "user".to_string(),
                content: Some(OpenAiContent::Blocks(vec![OpenAiContentBlock::ImageUrl {
                    image_url: OpenAiImageUrl { url: "https://example.com/cat.png".to_string() },
                }])),
                tool_calls: None,
                tool_call_id: None,
            }],
            max_tokens: None,
            temperature: None,
            response_format: None,
            tools: Vec::new(),
            stream: false,
            plugins: Vec::new(),
        };

        let chat: ChatRequest = req.into();
        assert_eq!(chat.messages[0].content, Vec::new());
    }

    #[test]
    fn outbound_mixed_text_image_and_tool_use_message() {
        let mut chat = chat_request(None);
        chat.messages = vec![Message {
            role: Role::Assistant,
            content: vec![
                ContentPart::Text { text: "checking weather".to_string() },
                ContentPart::Image { media_type: "image/png".to_string(), data: "abc123".to_string() },
                ContentPart::ToolUse {
                    id: "call_1".to_string(),
                    name: "get_weather".to_string(),
                    input: json!({"city": "nyc"}),
                },
            ],
        }];

        let req = OpenAiChatRequest::from(&chat);
        assert_eq!(req.messages.len(), 1);
        match &req.messages[0].content {
            Some(OpenAiContent::Blocks(blocks)) => assert_eq!(blocks.len(), 2),
            other => panic!("expected blocks content, got {other:?}"),
        }
        let tool_calls = req.messages[0].tool_calls.as_ref().expect("tool_calls set");
        assert_eq!(tool_calls[0].function.name, "get_weather");
    }

    #[test]
    fn decoder_emits_text_delta_then_done_with_usage() {
        let mut decoder = OpenAiStreamDecoder::default();

        let events = decoder.decode(OpenAiStreamChunk {
            choices: vec![OpenAiStreamChoice {
                delta: OpenAiStreamDelta { content: Some("hi".to_string()), tool_calls: None },
                finish_reason: None,
            }],
            usage: None,
        });
        assert_eq!(events, vec![StreamEvent::TextDelta { text: "hi".to_string() }]);

        let events = decoder.decode(OpenAiStreamChunk {
            choices: vec![OpenAiStreamChoice {
                delta: OpenAiStreamDelta::default(),
                finish_reason: Some("stop".to_string()),
            }],
            usage: Some(OpenAiUsage { prompt_tokens: 3, completion_tokens: 1, total_tokens: 4 }),
        });
        assert_eq!(
            events,
            vec![StreamEvent::Done {
                stop_reason: StopReason::EndTurn,
                usage: Usage { input_tokens: 3, output_tokens: 1 },
            }]
        );
    }

    #[test]
    fn decoder_reassembles_tool_call_id_across_chunks_by_index() {
        let mut decoder = OpenAiStreamDecoder::default();

        let events = decoder.decode(OpenAiStreamChunk {
            choices: vec![OpenAiStreamChoice {
                delta: OpenAiStreamDelta {
                    content: None,
                    tool_calls: Some(vec![OpenAiStreamToolCall {
                        index: 0,
                        id: Some("call_1".to_string()),
                        function: OpenAiStreamFunctionDelta {
                            name: Some("get_weather".to_string()),
                            arguments: Some(String::new()),
                        },
                    }]),
                },
                finish_reason: None,
            }],
            usage: None,
        });
        assert_eq!(
            events,
            vec![StreamEvent::ToolCallStart { id: "call_1".to_string(), name: "get_weather".to_string() }]
        );

        // Later chunks identify the same call by `index` alone, with no `id`.
        let events = decoder.decode(OpenAiStreamChunk {
            choices: vec![OpenAiStreamChoice {
                delta: OpenAiStreamDelta {
                    content: None,
                    tool_calls: Some(vec![OpenAiStreamToolCall {
                        index: 0,
                        id: None,
                        function: OpenAiStreamFunctionDelta {
                            name: None,
                            arguments: Some(r#"{"city":"nyc"}"#.to_string()),
                        },
                    }]),
                },
                finish_reason: None,
            }],
            usage: None,
        });
        assert_eq!(
            events,
            vec![StreamEvent::ToolCallDelta {
                id: "call_1".to_string(),
                partial_input: r#"{"city":"nyc"}"#.to_string(),
            }]
        );
    }

    fn events_stream(
        events: Vec<StreamEvent>,
    ) -> impl Stream<Item = anyhow::Result<StreamEvent>> + Unpin + Send + 'static {
        tokio_stream::iter(events.into_iter().map(Ok))
    }

    async fn render_to_string(events: Vec<StreamEvent>, model: &str) -> String {
        let mut stream = render_stream(events_stream(events), model.to_string());
        let mut out = String::new();
        while let Some(item) = stream.next().await {
            out.push_str(std::str::from_utf8(&item.unwrap()).unwrap());
        }
        out
    }

    #[tokio::test]
    async fn render_stream_emits_text_delta_then_stop_and_done_marker() {
        let out = render_to_string(
            vec![
                StreamEvent::TextDelta { text: "hi".to_string() },
                StreamEvent::Done { stop_reason: StopReason::EndTurn, usage: Usage::default() },
            ],
            "gpt-4o",
        )
        .await;

        let lines: Vec<&str> = out.lines().filter(|l| l.starts_with("data: ")).collect();
        let first: serde_json::Value = serde_json::from_str(lines[0].trim_start_matches("data: ")).unwrap();
        assert_eq!(first["choices"][0]["delta"]["content"], "hi");
        let second: serde_json::Value = serde_json::from_str(lines[1].trim_start_matches("data: ")).unwrap();
        assert_eq!(second["choices"][0]["finish_reason"], "stop");
        assert_eq!(lines[2], "data: [DONE]");
    }

    #[tokio::test]
    async fn render_stream_tool_call_start_then_delta_share_index() {
        let out = render_to_string(
            vec![
                StreamEvent::ToolCallStart { id: "call_1".to_string(), name: "get_weather".to_string() },
                StreamEvent::ToolCallDelta { id: "call_1".to_string(), partial_input: r#"{"city":"nyc"}"#.to_string() },
                StreamEvent::Done { stop_reason: StopReason::ToolUse, usage: Usage::default() },
            ],
            "gpt-4o",
        )
        .await;

        let lines: Vec<&str> = out.lines().filter(|l| l.starts_with("data: ")).collect();
        let start: serde_json::Value = serde_json::from_str(lines[0].trim_start_matches("data: ")).unwrap();
        let start_call = &start["choices"][0]["delta"]["tool_calls"][0];
        assert_eq!(start_call["id"], "call_1");
        assert_eq!(start_call["function"]["name"], "get_weather");
        let index = start_call["index"].clone();

        let delta: serde_json::Value = serde_json::from_str(lines[1].trim_start_matches("data: ")).unwrap();
        let delta_call = &delta["choices"][0]["delta"]["tool_calls"][0];
        assert_eq!(delta_call["index"], index);
        assert_eq!(delta_call["function"]["arguments"], r#"{"city":"nyc"}"#);

        let done: serde_json::Value = serde_json::from_str(lines[2].trim_start_matches("data: ")).unwrap();
        assert_eq!(done["choices"][0]["finish_reason"], "tool_calls");
    }
}
