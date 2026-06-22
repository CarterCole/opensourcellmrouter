//! Wire types for OpenAI's Responses API (`POST /v1/responses`), and
//! conversions to/from the [`canonical`](crate::canonical) representation.
//!
//! Codex CLI/App are the only clients this exists for — OpenAI has removed
//! Chat Completions support from both, so they speak only this format. The
//! router never calls *out* to a Responses-speaking upstream, so unlike
//! `formats::openai`/`formats::anthropic` there's only one direction of
//! conversion here: inbound request -> [`ChatRequest`], and
//! [`ChatResponse`] -> outbound response/stream.
//!
//! Scope is deliberately narrow — only what Codex CLI's own traffic
//! exercises: message input/output, function calls, `instructions`,
//! `tools`, `stream`, `max_output_tokens`. Explicitly not implemented:
//! `previous_response_id` (stateful threading — Codex against a
//! third-party provider replays the full `input` array each turn rather
//! than relying on it), image/file inputs, built-in tools (`web_search`,
//! `code_interpreter`), `reasoning` items, `background` mode.

use bytes::Bytes;
use futures_core::Stream;
use serde::{Deserialize, Serialize};
use serde_json::json;
use tokio_stream::StreamExt;
use tokio_stream::wrappers::ReceiverStream;

use crate::canonical::{ChatRequest, ChatResponse, ContentPart, Message, PluginRequest, Role, StreamEvent, Tool};

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum InputContentPart {
    InputText { text: String },
    OutputText { text: String },
}

/// One entry of the request's `input` array. Unlike Chat Completions /
/// Anthropic Messages, a tool call and its result are siblings of message
/// items in this flat array, not blocks nested inside a message.
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum InputItem {
    Message { role: String, content: Vec<InputContentPart> },
    FunctionCall { call_id: String, name: String, arguments: String },
    FunctionCallOutput { call_id: String, output: String },
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ResponsesTool {
    #[serde(rename = "type")]
    pub tool_type: String,
    pub name: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    pub parameters: serde_json::Value,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ResponsesRequest {
    pub model: String,
    #[serde(default)]
    pub input: Vec<InputItem>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub instructions: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tools: Vec<ResponsesTool>,
    #[serde(default)]
    pub stream: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_output_tokens: Option<u32>,
    /// Plugins to run for this request, e.g. `[{"id": "response-healing"}]`.
    /// Not part of the standard Responses API; stripped before forwarding.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub plugins: Vec<PluginRequest>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum OutputContentPart {
    OutputText { text: String },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum OutputItem {
    Message { id: String, role: String, status: String, content: Vec<OutputContentPart> },
    FunctionCall { id: String, call_id: String, name: String, arguments: String },
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ResponsesUsage {
    #[serde(default)]
    pub input_tokens: u32,
    #[serde(default)]
    pub output_tokens: u32,
    #[serde(default)]
    pub total_tokens: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResponsesResponse {
    pub id: String,
    pub object: String,
    pub model: String,
    pub status: String,
    pub output: Vec<OutputItem>,
    pub usage: ResponsesUsage,
}

/// An inbound request from a client speaking the Responses format.
impl From<ResponsesRequest> for ChatRequest {
    fn from(req: ResponsesRequest) -> Self {
        let mut messages = Vec::with_capacity(req.input.len());

        for item in req.input {
            match item {
                InputItem::Message { role, content } => {
                    let parts = content
                        .into_iter()
                        .map(|c| match c {
                            InputContentPart::InputText { text } => ContentPart::Text { text },
                            InputContentPart::OutputText { text } => ContentPart::Text { text },
                        })
                        .collect();
                    let role = if role == "assistant" { Role::Assistant } else { Role::User };
                    messages.push(Message { role, content: parts });
                }
                InputItem::FunctionCall { call_id, name, arguments } => {
                    let input = serde_json::from_str(&arguments).unwrap_or(serde_json::Value::Null);
                    messages.push(Message {
                        role: Role::Assistant,
                        content: vec![ContentPart::ToolUse { id: call_id, name, input }],
                    });
                }
                InputItem::FunctionCallOutput { call_id, output } => {
                    messages.push(Message {
                        role: Role::User,
                        content: vec![ContentPart::ToolResult { tool_use_id: call_id, content: output }],
                    });
                }
            }
        }

        let tools = req
            .tools
            .into_iter()
            .map(|t| Tool { name: t.name, description: t.description, input_schema: t.parameters })
            .collect();

        ChatRequest {
            model: req.model,
            system: req.instructions,
            messages,
            max_tokens: req.max_output_tokens,
            temperature: None,
            thinking: None,
            effort: None,
            task_budget: None,
            output_schema: None,
            tools,
            stream: req.stream,
            plugins: req.plugins,
            forced_provider: None,
            tags: Vec::new(),
        }
    }
}

/// A reply rendered for a client that speaks the Responses format.
/// `status` is always `"completed"` — truncation due to `max_output_tokens`
/// (which the real API surfaces as `status: "incomplete"`) isn't modeled,
/// since Codex doesn't act differently on it today.
impl From<ChatResponse> for ResponsesResponse {
    fn from(resp: ChatResponse) -> Self {
        let mut output = Vec::new();
        if !resp.content.is_empty() {
            output.push(OutputItem::Message {
                id: format!("msg_{}", resp.id),
                role: "assistant".to_string(),
                status: "completed".to_string(),
                content: vec![OutputContentPart::OutputText { text: resp.content }],
            });
        }
        for tc in resp.tool_calls {
            output.push(OutputItem::FunctionCall {
                id: format!("fc_{}", tc.id),
                call_id: tc.id,
                name: tc.name,
                arguments: tc.input.to_string(),
            });
        }

        ResponsesResponse {
            id: resp.id,
            object: "response".to_string(),
            model: resp.model,
            status: "completed".to_string(),
            output,
            usage: ResponsesUsage {
                input_tokens: resp.usage.input_tokens,
                output_tokens: resp.usage.output_tokens,
                total_tokens: resp.usage.input_tokens + resp.usage.output_tokens,
            },
        }
    }
}

/// A content block currently open in the synthesized Responses event
/// stream — mirrors `formats::anthropic`'s `OpenBlock`, since the Responses
/// API has the same "explicitly open/close one item at a time" structure.
enum OpenItem {
    Message { item_id: String },
    FunctionCall { item_id: String },
}

/// Renders a canonical stream as Responses API SSE
/// (`response.created` / `response.output_item.*` / `response.output_text.delta`
/// / `response.function_call_arguments.delta` / `response.completed`) — the
/// shape Codex CLI/App's streaming parser expects.
pub fn render_stream<S>(mut events: S, model: String) -> ReceiverStream<anyhow::Result<Bytes>>
where
    S: Stream<Item = anyhow::Result<StreamEvent>> + Unpin + Send + 'static,
{
    let (tx, rx) = tokio::sync::mpsc::channel(64);

    tokio::spawn(async move {
        let response_id = format!(
            "resp_{:x}",
            std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos()
        );

        macro_rules! send {
            ($event_name:expr, $data:expr) => {
                if tx.send(Ok(Bytes::from(format!("event: {}\ndata: {}\n\n", $event_name, $data)))).await.is_err() {
                    return;
                }
            };
        }

        send!("response.created", json!({"type": "response.created", "response": {"id": response_id, "model": model, "status": "in_progress"}}));

        let mut item_index: i64 = -1;
        let mut open_item: Option<OpenItem> = None;

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
                    let item_id = match &open_item {
                        Some(OpenItem::Message { item_id }) => item_id.clone(),
                        other => {
                            if other.is_some() {
                                send!("response.output_item.done", json!({"type": "response.output_item.done", "output_index": item_index}));
                            }
                            item_index += 1;
                            let item_id = format!("msg_{item_index}");
                            open_item = Some(OpenItem::Message { item_id: item_id.clone() });
                            send!(
                                "response.output_item.added",
                                json!({
                                    "type": "response.output_item.added",
                                    "output_index": item_index,
                                    "item": {"type": "message", "id": item_id, "role": "assistant", "status": "in_progress", "content": []},
                                })
                            );
                            item_id
                        }
                    };
                    send!(
                        "response.output_text.delta",
                        json!({"type": "response.output_text.delta", "item_id": item_id, "output_index": item_index, "delta": text})
                    );
                }
                StreamEvent::ToolCallStart { id, name } => {
                    if open_item.is_some() {
                        send!("response.output_item.done", json!({"type": "response.output_item.done", "output_index": item_index}));
                    }
                    item_index += 1;
                    let item_id = format!("fc_{item_index}");
                    open_item = Some(OpenItem::FunctionCall { item_id: item_id.clone() });
                    send!(
                        "response.output_item.added",
                        json!({
                            "type": "response.output_item.added",
                            "output_index": item_index,
                            "item": {"type": "function_call", "id": item_id, "call_id": id, "name": name, "arguments": ""},
                        })
                    );
                }
                StreamEvent::ToolCallDelta { id, partial_input } => {
                    let item_id = match &open_item {
                        Some(OpenItem::FunctionCall { item_id, .. }) => item_id.clone(),
                        _ => format!("fc_{item_index}"),
                    };
                    send!(
                        "response.function_call_arguments.delta",
                        json!({"type": "response.function_call_arguments.delta", "item_id": item_id, "output_index": item_index, "call_id": id, "delta": partial_input})
                    );
                }
                StreamEvent::Done { usage, .. } => {
                    if open_item.is_some() {
                        send!("response.output_item.done", json!({"type": "response.output_item.done", "output_index": item_index}));
                    }
                    send!(
                        "response.completed",
                        json!({
                            "type": "response.completed",
                            "response": {
                                "id": response_id,
                                "model": model,
                                "status": "completed",
                                "usage": {
                                    "input_tokens": usage.input_tokens,
                                    "output_tokens": usage.output_tokens,
                                    "total_tokens": usage.input_tokens + usage.output_tokens,
                                },
                            },
                        })
                    );
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
    use crate::canonical::{StopReason, ToolCall, Usage};

    #[test]
    fn inbound_message_items_map_to_canonical_messages() {
        let req = ResponsesRequest {
            model: "gpt-5-codex".to_string(),
            input: vec![InputItem::Message {
                role: "user".to_string(),
                content: vec![InputContentPart::InputText { text: "hi".to_string() }],
            }],
            instructions: Some("be terse".to_string()),
            tools: Vec::new(),
            stream: false,
            max_output_tokens: None,
            plugins: Vec::new(),
        };

        let chat: ChatRequest = req.into();
        assert_eq!(chat.system, Some("be terse".to_string()));
        assert_eq!(chat.messages[0].role, Role::User);
        assert_eq!(chat.messages[0].content, vec![ContentPart::Text { text: "hi".to_string() }]);
    }

    #[test]
    fn inbound_function_call_and_output_items_round_trip() {
        let req = ResponsesRequest {
            model: "gpt-5-codex".to_string(),
            input: vec![
                InputItem::FunctionCall {
                    call_id: "call_1".to_string(),
                    name: "get_weather".to_string(),
                    arguments: r#"{"city":"nyc"}"#.to_string(),
                },
                InputItem::FunctionCallOutput { call_id: "call_1".to_string(), output: "sunny".to_string() },
            ],
            instructions: None,
            tools: Vec::new(),
            stream: false,
            max_output_tokens: None,
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
        assert_eq!(
            chat.messages[1].content,
            vec![ContentPart::ToolResult { tool_use_id: "call_1".to_string(), content: "sunny".to_string() }]
        );
    }

    #[test]
    fn inbound_tools_map_to_canonical_tool() {
        let req = ResponsesRequest {
            model: "gpt-5-codex".to_string(),
            input: Vec::new(),
            instructions: None,
            tools: vec![ResponsesTool {
                tool_type: "function".to_string(),
                name: "get_weather".to_string(),
                description: Some("Looks up weather".to_string()),
                parameters: json!({"type": "object"}),
            }],
            stream: false,
            max_output_tokens: None,
            plugins: Vec::new(),
        };

        let chat: ChatRequest = req.into();
        assert_eq!(chat.tools.len(), 1);
        assert_eq!(chat.tools[0].name, "get_weather");
    }

    fn chat_response(content: &str, tool_calls: Vec<ToolCall>) -> ChatResponse {
        ChatResponse {
            id: "resp_1".to_string(),
            model: "gpt-5-codex".to_string(),
            content: content.to_string(),
            stop_reason: StopReason::EndTurn,
            tool_calls,
            usage: Usage { input_tokens: 3, output_tokens: 5 },
            tags: Vec::new(),
        }
    }

    #[test]
    fn outbound_text_response_becomes_message_output_item() {
        let resp = ResponsesResponse::from(chat_response("hi there", Vec::new()));
        assert_eq!(resp.output.len(), 1);
        match &resp.output[0] {
            OutputItem::Message { role, content, .. } => {
                assert_eq!(role, "assistant");
                assert_eq!(content, &vec![OutputContentPart::OutputText { text: "hi there".to_string() }]);
            }
            other => panic!("expected message output item, got {other:?}"),
        }
        assert_eq!(resp.usage.total_tokens, 8);
    }

    #[test]
    fn outbound_tool_call_becomes_function_call_output_item() {
        let resp = ResponsesResponse::from(chat_response(
            "",
            vec![ToolCall { id: "call_1".to_string(), name: "get_weather".to_string(), input: json!({"city": "nyc"}) }],
        ));
        assert_eq!(resp.output.len(), 1);
        match &resp.output[0] {
            OutputItem::FunctionCall { call_id, name, arguments, .. } => {
                assert_eq!(call_id, "call_1");
                assert_eq!(name, "get_weather");
                assert_eq!(arguments, r#"{"city":"nyc"}"#);
            }
            other => panic!("expected function_call output item, got {other:?}"),
        }
    }

    fn events_stream(
        events: Vec<StreamEvent>,
    ) -> impl Stream<Item = anyhow::Result<StreamEvent>> + Unpin + Send + 'static {
        tokio_stream::iter(events.into_iter().map(Ok))
    }

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
    async fn render_stream_text_then_done_opens_message_item_and_completes() {
        let events = render_to_events(
            vec![
                StreamEvent::TextDelta { text: "hi".to_string() },
                StreamEvent::Done { stop_reason: StopReason::EndTurn, usage: Usage { input_tokens: 1, output_tokens: 2 } },
            ],
            "gpt-5-codex",
        )
        .await;

        let names: Vec<&str> = events.iter().map(|(n, _)| n.as_str()).collect();
        assert_eq!(
            names,
            vec![
                "response.created",
                "response.output_item.added",
                "response.output_text.delta",
                "response.output_item.done",
                "response.completed",
            ]
        );
        assert_eq!(events[2].1["delta"], "hi");
        assert_eq!(events[4].1["response"]["usage"]["output_tokens"], 2);
    }

    #[tokio::test]
    async fn render_stream_function_call_emits_arguments_delta() {
        let events = render_to_events(
            vec![
                StreamEvent::ToolCallStart { id: "call_1".to_string(), name: "get_weather".to_string() },
                StreamEvent::ToolCallDelta { id: "call_1".to_string(), partial_input: r#"{"city":"nyc"}"#.to_string() },
                StreamEvent::Done { stop_reason: StopReason::ToolUse, usage: Usage::default() },
            ],
            "gpt-5-codex",
        )
        .await;

        let names: Vec<&str> = events.iter().map(|(n, _)| n.as_str()).collect();
        assert_eq!(
            names,
            vec![
                "response.created",
                "response.output_item.added",
                "response.function_call_arguments.delta",
                "response.output_item.done",
                "response.completed",
            ]
        );
        assert_eq!(events[1].1["item"]["call_id"], "call_1");
        assert_eq!(events[2].1["delta"], r#"{"city":"nyc"}"#);
    }
}
