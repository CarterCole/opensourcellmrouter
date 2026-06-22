//! Wire types for Ollama's native API (`/api/chat`, `/api/tags`), and
//! conversions to/from the [`canonical`](crate::canonical) representation.
//!
//! Ollama is only ever used as an upstream provider — there is no
//! Ollama-shaped inbound endpoint — so unlike `formats::openai` /
//! `formats::anthropic`, only the outbound (`ChatRequest -> OllamaChatRequest`)
//! and reply (`OllamaChatResponse -> ChatResponse`) directions are needed.

use serde::{Deserialize, Serialize};

use crate::canonical::{ChatRequest, ChatResponse, ContentPart, Message, Role, StopReason, ToolCall, Usage};

#[derive(Debug, Clone, Serialize)]
pub struct OllamaMessage {
    pub role: String,
    pub content: String,
    /// Base64-encoded images, Ollama's native shape (separate from
    /// `content`, unlike OpenAI/Anthropic's inline content blocks).
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub images: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<OllamaToolCall>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OllamaToolCall {
    pub function: OllamaFunctionCall,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OllamaFunctionCall {
    pub name: String,
    pub arguments: serde_json::Value,
}

#[derive(Debug, Clone, Serialize)]
pub struct OllamaTool {
    #[serde(rename = "type")]
    pub tool_type: String,
    pub function: OllamaFunctionDef,
}

#[derive(Debug, Clone, Serialize)]
pub struct OllamaFunctionDef {
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    pub parameters: serde_json::Value,
}

/// Generation parameters. Ollama accepts these under `options` rather than
/// as top-level request fields.
#[derive(Debug, Clone, Default, Serialize)]
pub struct OllamaOptions {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    /// Maximum number of tokens to generate. Ollama's name for this is
    /// `num_predict`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub num_predict: Option<u32>,
}

#[derive(Debug, Clone, Serialize)]
pub struct OllamaChatRequest {
    pub model: String,
    pub messages: Vec<OllamaMessage>,
    pub stream: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub options: Option<OllamaOptions>,
    /// Structured-outputs config: either the literal `"json"` or a full JSON
    /// Schema object. See
    /// <https://docs.ollama.com/capabilities/structured-outputs>.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub format: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub tools: Vec<OllamaTool>,
}

#[derive(Debug, Clone, Default, Deserialize)]
pub struct OllamaResponseMessage {
    #[serde(default)]
    pub content: String,
    #[serde(default)]
    pub tool_calls: Vec<OllamaToolCall>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct OllamaChatResponse {
    pub model: String,
    #[serde(default)]
    pub message: OllamaResponseMessage,
    /// `false` on streaming chunks, `true` on the final chunk.
    #[serde(default)]
    pub done: bool,
    /// Why generation stopped, e.g. `"stop"` or `"length"`. Only present
    /// once `done` is `true`.
    #[serde(default)]
    pub done_reason: Option<String>,
    /// Input token count. Only present once `done` is `true`.
    #[serde(default)]
    pub prompt_eval_count: u32,
    /// Output token count. Only present once `done` is `true`.
    #[serde(default)]
    pub eval_count: u32,
}

/// Response from `GET /api/tags`: the models Ollama currently has pulled.
#[derive(Debug, Clone, Default, Deserialize)]
pub struct OllamaTagsResponse {
    #[serde(default)]
    pub models: Vec<OllamaModelInfo>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct OllamaModelInfo {
    pub name: String,
}

/// Request body for `POST /api/show`.
#[derive(Debug, Clone, Serialize)]
pub struct OllamaShowRequest {
    pub model: String,
}

/// Response from `POST /api/show`. `capabilities` is Ollama's own fixed set
/// (`"completion"`, `"tools"`, `"vision"`, `"embedding"`) — it has no entry
/// for things like "coding", which `implicit_capabilities` infers separately
/// from the model name/family instead.
#[derive(Debug, Clone, Default, Deserialize)]
pub struct OllamaShowResponse {
    #[serde(default)]
    pub capabilities: Vec<String>,
    #[serde(default)]
    pub details: OllamaShowDetails,
}

#[derive(Debug, Clone, Default, Deserialize)]
pub struct OllamaShowDetails {
    #[serde(default)]
    pub family: String,
}

/// Capabilities Ollama doesn't explicitly report via `/api/show`, inferred
/// from the model name/family instead. Currently just "coding" — a
/// code-specialized model (`codellama`, `deepseek-coder`,
/// `qwen2.5-coder`, ...) reports the same bare `["completion"]` as any other
/// text model, since Ollama has no native capability tag for it.
pub fn implicit_capabilities(model: &str, family: &str) -> Vec<String> {
    let haystack = format!("{model} {family}").to_lowercase();
    let mut caps = Vec::new();
    if haystack.contains("code") {
        caps.push("coding".to_string());
    }
    caps
}

/// Renders one canonical [`Message`] as one or more [`OllamaMessage`]s — a
/// tool result becomes its own `role: "tool"` message, mirroring how
/// OpenAI-shaped tool results work.
fn ollama_messages_from_canonical(msg: &Message) -> Vec<OllamaMessage> {
    let role = match msg.role {
        Role::User => "user",
        Role::Assistant => "assistant",
    };

    let mut text = String::new();
    let mut images = Vec::new();
    let mut tool_calls = Vec::new();
    let mut tool_results = Vec::new();

    for part in &msg.content {
        match part {
            ContentPart::Text { text: t } => text.push_str(t),
            // Ollama only accepts inline base64 images (no media-type tag);
            // since our canonical Image is already base64, this is lossless.
            ContentPart::Image { data, .. } => images.push(data.clone()),
            ContentPart::ToolUse { name, input, .. } => tool_calls.push(OllamaToolCall {
                function: OllamaFunctionCall { name: name.clone(), arguments: input.clone() },
            }),
            ContentPart::ToolResult { content, .. } => tool_results.push(content.clone()),
        }
    }

    let mut out = Vec::new();
    if !text.is_empty() || !images.is_empty() || !tool_calls.is_empty() {
        out.push(OllamaMessage {
            role: role.to_string(),
            content: text,
            images,
            tool_calls: (!tool_calls.is_empty()).then_some(tool_calls),
        });
    }
    for content in tool_results {
        out.push(OllamaMessage { role: "tool".to_string(), content, images: Vec::new(), tool_calls: None });
    }
    out
}

/// An outbound request to a provider that speaks Ollama's native API.
impl From<&ChatRequest> for OllamaChatRequest {
    fn from(req: &ChatRequest) -> Self {
        let mut messages = Vec::with_capacity(req.messages.len() + 1);

        if let Some(system) = &req.system {
            messages.push(OllamaMessage {
                role: "system".to_string(),
                content: system.clone(),
                images: Vec::new(),
                tool_calls: None,
            });
        }

        for msg in &req.messages {
            messages.extend(ollama_messages_from_canonical(msg));
        }

        let options = if req.temperature.is_some() || req.max_tokens.is_some() {
            Some(OllamaOptions {
                temperature: req.temperature,
                num_predict: req.max_tokens,
            })
        } else {
            None
        };

        let tools = req
            .tools
            .iter()
            .map(|t| OllamaTool {
                tool_type: "function".to_string(),
                function: OllamaFunctionDef {
                    name: t.name.clone(),
                    description: t.description.clone(),
                    parameters: t.input_schema.clone(),
                },
            })
            .collect();

        OllamaChatRequest {
            model: req.model.clone(),
            messages,
            stream: req.stream,
            options,
            format: req.output_schema.clone(),
            tools,
        }
    }
}

/// A reply from a provider that speaks Ollama's native API.
impl From<OllamaChatResponse> for ChatResponse {
    fn from(resp: OllamaChatResponse) -> Self {
        let tool_calls: Vec<ToolCall> = resp
            .message
            .tool_calls
            .into_iter()
            .enumerate()
            .map(|(i, tc)| ToolCall {
                // Ollama doesn't assign call IDs; synthesize one.
                id: format!("ollama-call-{i}"),
                name: tc.function.name,
                input: tc.function.arguments,
            })
            .collect();

        let stop_reason = if !tool_calls.is_empty() {
            StopReason::ToolUse
        } else {
            match resp.done_reason.as_deref() {
                Some("stop") => StopReason::EndTurn,
                Some("length") => StopReason::MaxTokens,
                _ => StopReason::Other,
            }
        };

        ChatResponse {
            id: format!("ollama-{}", resp.model),
            model: resp.model,
            content: resp.message.content,
            stop_reason,
            tool_calls,
            usage: Usage {
                input_tokens: resp.prompt_eval_count,
                output_tokens: resp.eval_count,
            },
            tags: Vec::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::canonical::Tool;

    fn base_request(messages: Vec<Message>) -> ChatRequest {
        ChatRequest {
            model: "llama3".to_string(),
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
    fn request_includes_system_and_options() {
        let mut req = base_request(vec![Message::text(Role::User, "hi")]);
        req.system = Some("be terse".to_string());
        req.max_tokens = Some(128);
        req.temperature = Some(0.5);

        let ollama_req = OllamaChatRequest::from(&req);
        assert_eq!(ollama_req.model, "llama3");
        assert_eq!(ollama_req.messages[0].role, "system");
        assert_eq!(ollama_req.messages[0].content, "be terse");
        assert_eq!(ollama_req.messages[1].role, "user");
        let options = ollama_req.options.expect("options should be set");
        assert_eq!(options.temperature, Some(0.5));
        assert_eq!(options.num_predict, Some(128));
        assert_eq!(ollama_req.format, None);
    }

    #[test]
    fn request_forwards_output_schema_as_format() {
        let schema = serde_json::json!({"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]});
        let mut req = base_request(vec![Message::text(Role::User, "hi")]);
        req.output_schema = Some(schema.clone());

        let ollama_req = OllamaChatRequest::from(&req);
        assert_eq!(ollama_req.format, Some(schema));
    }

    #[test]
    fn request_includes_images_array_from_image_content_part() {
        let msg = Message {
            role: Role::User,
            content: vec![
                ContentPart::Text { text: "what's this?".to_string() },
                ContentPart::Image { media_type: "image/png".to_string(), data: "abc123".to_string() },
            ],
        };
        let req = base_request(vec![msg]);

        let ollama_req = OllamaChatRequest::from(&req);
        assert_eq!(ollama_req.messages[0].content, "what's this?");
        assert_eq!(ollama_req.messages[0].images, vec!["abc123".to_string()]);
    }

    #[test]
    fn request_includes_tool_calls_and_tools_field() {
        let msg = Message {
            role: Role::Assistant,
            content: vec![ContentPart::ToolUse {
                id: "call_1".to_string(),
                name: "get_weather".to_string(),
                input: serde_json::json!({"city": "nyc"}),
            }],
        };
        let mut req = base_request(vec![msg]);
        req.tools = vec![Tool {
            name: "get_weather".to_string(),
            description: Some("Looks up weather".to_string()),
            input_schema: serde_json::json!({"type": "object"}),
        }];

        let ollama_req = OllamaChatRequest::from(&req);
        assert_eq!(ollama_req.tools.len(), 1);
        let tool_calls = ollama_req.messages[0].tool_calls.as_ref().expect("tool_calls set");
        assert_eq!(tool_calls[0].function.name, "get_weather");
    }

    #[test]
    fn request_splits_tool_result_into_separate_tool_message() {
        let msg = Message {
            role: Role::User,
            content: vec![ContentPart::ToolResult {
                tool_use_id: "call_1".to_string(),
                content: "sunny".to_string(),
            }],
        };
        let req = base_request(vec![msg]);

        let ollama_req = OllamaChatRequest::from(&req);
        assert_eq!(ollama_req.messages.len(), 1);
        assert_eq!(ollama_req.messages[0].role, "tool");
        assert_eq!(ollama_req.messages[0].content, "sunny");
    }

    #[test]
    fn request_includes_mixed_text_image_and_tool_use_on_one_message() {
        let msg = Message {
            role: Role::Assistant,
            content: vec![
                ContentPart::Text { text: "checking weather".to_string() },
                ContentPart::Image { media_type: "image/png".to_string(), data: "abc123".to_string() },
                ContentPart::ToolUse {
                    id: "call_1".to_string(),
                    name: "get_weather".to_string(),
                    input: serde_json::json!({"city": "nyc"}),
                },
            ],
        };
        let req = base_request(vec![msg]);

        let ollama_req = OllamaChatRequest::from(&req);
        assert_eq!(ollama_req.messages.len(), 1);
        assert_eq!(ollama_req.messages[0].content, "checking weather");
        assert_eq!(ollama_req.messages[0].images, vec!["abc123".to_string()]);
        let tool_calls = ollama_req.messages[0].tool_calls.as_ref().expect("tool_calls set");
        assert_eq!(tool_calls[0].function.name, "get_weather");
    }

    #[test]
    fn response_maps_usage_and_stop_reason() {
        let resp = OllamaChatResponse {
            model: "llama3".to_string(),
            message: OllamaResponseMessage {
                content: "hello".to_string(),
                tool_calls: Vec::new(),
            },
            done: true,
            done_reason: Some("stop".to_string()),
            prompt_eval_count: 10,
            eval_count: 5,
        };

        let chat_resp: ChatResponse = resp.into();
        assert_eq!(chat_resp.content, "hello");
        assert_eq!(chat_resp.stop_reason, StopReason::EndTurn);
        assert_eq!(chat_resp.usage.input_tokens, 10);
        assert_eq!(chat_resp.usage.output_tokens, 5);
    }

    #[test]
    fn response_maps_tool_calls_into_chat_response() {
        let resp = OllamaChatResponse {
            model: "llama3".to_string(),
            message: OllamaResponseMessage {
                content: String::new(),
                tool_calls: vec![OllamaToolCall {
                    function: OllamaFunctionCall {
                        name: "get_weather".to_string(),
                        arguments: serde_json::json!({"city": "nyc"}),
                    },
                }],
            },
            done: true,
            done_reason: Some("stop".to_string()),
            prompt_eval_count: 10,
            eval_count: 5,
        };

        let chat_resp: ChatResponse = resp.into();
        assert_eq!(chat_resp.stop_reason, StopReason::ToolUse);
        assert_eq!(chat_resp.tool_calls.len(), 1);
        assert_eq!(chat_resp.tool_calls[0].name, "get_weather");
    }

    #[test]
    fn parses_tags_response() {
        let json = r#"{"models":[{"name":"llama3:8b"},{"name":"mistral:latest"}]}"#;
        let parsed: OllamaTagsResponse = serde_json::from_str(json).unwrap();
        let names: Vec<&str> = parsed.models.iter().map(|m| m.name.as_str()).collect();
        assert_eq!(names, vec!["llama3:8b", "mistral:latest"]);
    }

    #[test]
    fn parses_show_capabilities_response() {
        let json = r#"{"capabilities":["completion","vision"],"details":{"family":"gemma3"}}"#;
        let parsed: OllamaShowResponse = serde_json::from_str(json).unwrap();
        assert_eq!(parsed.capabilities, vec!["completion".to_string(), "vision".to_string()]);
        assert_eq!(parsed.details.family, "gemma3");
    }

    #[test]
    fn implicit_capabilities_detects_code_in_model_or_family() {
        assert_eq!(implicit_capabilities("deepseek-coder:6.7b", ""), vec!["coding".to_string()]);
        assert_eq!(implicit_capabilities("qwen2.5-coder:latest", ""), vec!["coding".to_string()]);
        assert_eq!(implicit_capabilities("custom-model:latest", "codellama"), vec!["coding".to_string()]);
        assert_eq!(implicit_capabilities("gemma3:latest", "gemma3"), Vec::<String>::new());
    }
}
