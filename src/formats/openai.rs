//! Wire types for the OpenAI `/v1/chat/completions` shape, and conversions
//! to/from the [`canonical`](crate::canonical) representation.
//!
//! These types are used both when the router receives an OpenAI-shaped
//! request from a client, and when it forwards a request to a provider that
//! itself speaks the OpenAI format (e.g. a local `llama-server`).

use serde::{Deserialize, Serialize};

use crate::canonical::{ChatRequest, ChatResponse, Message, PluginRequest, Role, StopReason, Usage};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenAiMessage {
    pub role: String,
    pub content: String,
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

/// An inbound request from a client speaking the OpenAI format.
impl From<OpenAiChatRequest> for ChatRequest {
    fn from(req: OpenAiChatRequest) -> Self {
        let mut system = None;
        let mut messages = Vec::with_capacity(req.messages.len());

        for msg in req.messages {
            match msg.role.as_str() {
                "system" => system = Some(msg.content),
                "assistant" => messages.push(Message {
                    role: Role::Assistant,
                    content: msg.content,
                }),
                // Treat anything else (user, tool, ...) as a user turn.
                _ => messages.push(Message {
                    role: Role::User,
                    content: msg.content,
                }),
            }
        }

        let output_schema = req
            .response_format
            .filter(|f| f.format_type == "json_schema")
            .map(|f| f.json_schema.schema);

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
            messages.push(OpenAiMessage {
                role: "system".to_string(),
                content: system.clone(),
            });
        }

        for msg in &req.messages {
            let role = match msg.role {
                Role::User => "user",
                Role::Assistant => "assistant",
            };
            messages.push(OpenAiMessage {
                role: role.to_string(),
                content: msg.content.clone(),
            });
        }

        let response_format = req.output_schema.clone().map(|schema| ResponseFormat {
            format_type: "json_schema".to_string(),
            json_schema: JsonSchemaFormat {
                name: default_schema_name(),
                schema,
                strict: true,
            },
        });

        OpenAiChatRequest {
            model: req.model.clone(),
            messages,
            max_tokens: req.max_tokens,
            temperature: req.temperature,
            response_format,
            stream: false,
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
            .map(|c| c.message.content.clone())
            .unwrap_or_default();
        let stop_reason = match choice.and_then(|c| c.finish_reason) {
            Some(reason) if reason == "length" => StopReason::MaxTokens,
            Some(reason) if reason == "stop" => StopReason::EndTurn,
            _ => StopReason::Other,
        };

        ChatResponse {
            id: resp.id,
            model: resp.model,
            content,
            stop_reason,
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
            StopReason::Other => "stop",
        };

        OpenAiChatResponse {
            id: resp.id,
            object: "chat.completion".to_string(),
            model: resp.model,
            choices: vec![OpenAiChoice {
                index: 0,
                message: OpenAiMessage {
                    role: "assistant".to_string(),
                    content: resp.content,
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

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn chat_request(output_schema: Option<serde_json::Value>) -> ChatRequest {
        ChatRequest {
            model: "gpt-4o".to_string(),
            system: None,
            messages: vec![Message {
                role: Role::User,
                content: "hi".to_string(),
            }],
            max_tokens: None,
            temperature: None,
            thinking: None,
            effort: None,
            task_budget: None,
            output_schema,
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
            messages: vec![OpenAiMessage { role: "user".to_string(), content: "hi".to_string() }],
            max_tokens: None,
            temperature: None,
            response_format: Some(ResponseFormat {
                format_type: "json_schema".to_string(),
                json_schema: JsonSchemaFormat { name: "contact".to_string(), schema: schema.clone(), strict: true },
            }),
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
            messages: vec![OpenAiMessage { role: "user".to_string(), content: "hi".to_string() }],
            max_tokens: None,
            temperature: None,
            response_format: None,
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
}
