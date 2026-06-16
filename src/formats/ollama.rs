//! Wire types for Ollama's native API (`/api/chat`, `/api/tags`), and
//! conversions to/from the [`canonical`](crate::canonical) representation.
//!
//! Ollama is only ever used as an upstream provider — there is no
//! Ollama-shaped inbound endpoint — so unlike `formats::openai` /
//! `formats::anthropic`, only the outbound (`ChatRequest -> OllamaChatRequest`)
//! and reply (`OllamaChatResponse -> ChatResponse`) directions are needed.

use serde::{Deserialize, Serialize};

use crate::canonical::{ChatRequest, ChatResponse, Role, StopReason, Usage};

#[derive(Debug, Clone, Serialize)]
pub struct OllamaMessage {
    pub role: String,
    pub content: String,
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
}

#[derive(Debug, Clone, Default, Deserialize)]
pub struct OllamaResponseMessage {
    #[serde(default)]
    pub content: String,
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

/// An outbound request to a provider that speaks Ollama's native API.
impl From<&ChatRequest> for OllamaChatRequest {
    fn from(req: &ChatRequest) -> Self {
        let mut messages = Vec::with_capacity(req.messages.len() + 1);

        if let Some(system) = &req.system {
            messages.push(OllamaMessage {
                role: "system".to_string(),
                content: system.clone(),
            });
        }

        for msg in &req.messages {
            let role = match msg.role {
                Role::User => "user",
                Role::Assistant => "assistant",
            };
            messages.push(OllamaMessage {
                role: role.to_string(),
                content: msg.content.clone(),
            });
        }

        let options = if req.temperature.is_some() || req.max_tokens.is_some() {
            Some(OllamaOptions {
                temperature: req.temperature,
                num_predict: req.max_tokens,
            })
        } else {
            None
        };

        OllamaChatRequest {
            model: req.model.clone(),
            messages,
            stream: false,
            options,
        }
    }
}

/// A reply from a provider that speaks Ollama's native API.
impl From<OllamaChatResponse> for ChatResponse {
    fn from(resp: OllamaChatResponse) -> Self {
        let stop_reason = match resp.done_reason.as_deref() {
            Some("stop") => StopReason::EndTurn,
            Some("length") => StopReason::MaxTokens,
            _ => StopReason::Other,
        };

        ChatResponse {
            id: format!("ollama-{}", resp.model),
            model: resp.model,
            content: resp.message.content,
            stop_reason,
            usage: Usage {
                input_tokens: resp.prompt_eval_count,
                output_tokens: resp.eval_count,
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::canonical::Message;

    #[test]
    fn request_includes_system_and_options() {
        let req = ChatRequest {
            model: "llama3".to_string(),
            system: Some("be terse".to_string()),
            messages: vec![Message {
                role: Role::User,
                content: "hi".to_string(),
            }],
            max_tokens: Some(128),
            temperature: Some(0.5),
            stream: false,
            plugins: Vec::new(),
            forced_provider: None,
            tags: Vec::new(),
        };

        let ollama_req = OllamaChatRequest::from(&req);
        assert_eq!(ollama_req.model, "llama3");
        assert_eq!(ollama_req.messages[0].role, "system");
        assert_eq!(ollama_req.messages[0].content, "be terse");
        assert_eq!(ollama_req.messages[1].role, "user");
        let options = ollama_req.options.expect("options should be set");
        assert_eq!(options.temperature, Some(0.5));
        assert_eq!(options.num_predict, Some(128));
    }

    #[test]
    fn response_maps_usage_and_stop_reason() {
        let resp = OllamaChatResponse {
            model: "llama3".to_string(),
            message: OllamaResponseMessage {
                content: "hello".to_string(),
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
    fn parses_tags_response() {
        let json = r#"{"models":[{"name":"llama3:8b"},{"name":"mistral:latest"}]}"#;
        let parsed: OllamaTagsResponse = serde_json::from_str(json).unwrap();
        let names: Vec<&str> = parsed.models.iter().map(|m| m.name.as_str()).collect();
        assert_eq!(names, vec!["llama3:8b", "mistral:latest"]);
    }
}
