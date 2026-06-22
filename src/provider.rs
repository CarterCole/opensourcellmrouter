//! A configured upstream LLM backend, and the logic to call it.

use anyhow::{bail, Context};
use reqwest::Client;
use tokio_stream::StreamExt;
use tokio_stream::wrappers::ReceiverStream;
use tracing::Instrument;

use crate::canonical::{ChatRequest, ChatResponse, StreamEvent};
use crate::config::{ProviderConfig, ProviderFormat};
use crate::formats::{anthropic, ollama, openai};

/// A stream of canonical [`StreamEvent`]s from an upstream provider,
/// format-agnostic so the caller can render it into whichever wire format
/// the client actually needs (OpenAI/Anthropic/Responses SSE) — see
/// `server::dispatch_stream` and each format's `render_stream`.
pub type ChunkStream = ReceiverStream<anyhow::Result<StreamEvent>>;

pub struct Provider {
    pub name: String,
    pub format: ProviderFormat,
    base_url: String,
    api_key_env: Option<String>,
}

impl Provider {
    pub fn from_config(config: &ProviderConfig) -> Self {
        Provider {
            name: config.name.clone(),
            format: config.format,
            base_url: config.base_url.trim_end_matches('/').to_string(),
            api_key_env: config.api_key_env.clone(),
        }
    }

    /// Resolves the API key from the configured environment variable, if
    /// any. Deferred to call time so that providers with unset keys don't
    /// prevent the router from starting up if they're never used.
    fn api_key(&self) -> anyhow::Result<Option<String>> {
        match &self.api_key_env {
            Some(var) => {
                let key = std::env::var(var).with_context(|| {
                    format!(
                        "provider '{}' has api_key_env = \"{}\" but that variable is not set",
                        self.name, var
                    )
                })?;
                Ok(Some(key))
            }
            None => Ok(None),
        }
    }

    pub async fn send(&self, client: &Client, req: &ChatRequest) -> anyhow::Result<ChatResponse> {
        match self.format {
            ProviderFormat::OpenAi => self.send_openai(client, req).await,
            ProviderFormat::Anthropic => self.send_anthropic(client, req).await,
            ProviderFormat::Ollama => self.send_ollama(client, req).await,
        }
    }

    /// Opens a streaming connection to the upstream and returns a channel
    /// receiver that yields SSE-formatted `data: {...}\n\n` strings in the
    /// OpenAI chunk format. When the stream ends normally the last item is
    /// `data: [DONE]\n\n`. On error an `Err` item is emitted instead.
    pub async fn send_streaming(&self, client: &Client, req: &ChatRequest) -> anyhow::Result<ChunkStream> {
        match self.format {
            ProviderFormat::OpenAi => self.stream_openai(client, req).await,
            ProviderFormat::Ollama => self.stream_ollama(client, req).await,
            // Real upstream SSE streaming (content_block_delta events) isn't
            // implemented for Anthropic-format providers — a request routed
            // here with stream=true fails with this error rather than
            // silently buffering. Out of scope for now; see docs/plan notes.
            ProviderFormat::Anthropic => bail!("streaming not supported for Anthropic format"),
        }
    }

    /// Parses the upstream's own OpenAI-shaped `text/event-stream` response
    /// into canonical [`StreamEvent`]s via [`openai::OpenAiStreamDecoder`].
    async fn stream_openai(&self, client: &Client, req: &ChatRequest) -> anyhow::Result<ChunkStream> {
        let body = openai::OpenAiChatRequest::from(req);
        let url = format!("{}/chat/completions", self.base_url);

        let mut rb = client.post(&url).json(&body);
        if let Some(key) = self.api_key()? {
            rb = rb.bearer_auth(key);
        }

        let resp = rb
            .send()
            .instrument(tracing::info_span!("provider.http", provider = %self.name, kind = "openai_stream"))
            .await
            .with_context(|| format!("calling provider '{}'", self.name))?;
        let status = resp.status();
        if !status.is_success() {
            let text = resp.text().await?;
            bail!("provider '{}' returned {}: {}", self.name, status, text);
        }

        let (tx, rx) = tokio::sync::mpsc::channel(64);
        tokio::spawn(async move {
            let mut decoder = openai::OpenAiStreamDecoder::default();
            let mut byte_stream = resp.bytes_stream();
            let mut buf = String::new();

            while let Some(chunk) = byte_stream.next().await {
                let bytes = match chunk {
                    Ok(b) => b,
                    Err(e) => { tx.send(Err(anyhow::anyhow!(e))).await.ok(); return; }
                };
                buf.push_str(&String::from_utf8_lossy(&bytes));

                while let Some(nl) = buf.find('\n') {
                    let line = buf[..nl].trim().to_owned();
                    buf.drain(..=nl);
                    let Some(data) = line.strip_prefix("data: ") else { continue };
                    if data == "[DONE]" {
                        return;
                    }

                    let chunk_resp: openai::OpenAiStreamChunk = match serde_json::from_str(data) {
                        Ok(r) => r,
                        Err(e) => {
                            tx.send(Err(anyhow::anyhow!("parsing OpenAI stream chunk: {e}"))).await.ok();
                            return;
                        }
                    };

                    for event in decoder.decode(chunk_resp) {
                        if tx.send(Ok(event)).await.is_err() { return; }
                    }
                }
            }
        });

        Ok(ReceiverStream::new(rx))
    }

    /// Translates Ollama's NDJSON streaming format into canonical
    /// [`StreamEvent`]s. Ollama doesn't stream tool-call deltas today, so
    /// this only ever emits `TextDelta`/`Done`.
    async fn stream_ollama(&self, client: &Client, req: &ChatRequest) -> anyhow::Result<ChunkStream> {
        let body = ollama::OllamaChatRequest::from(req);
        let url = format!("{}/api/chat", self.base_url);

        let mut rb = client.post(&url).json(&body);
        if let Some(key) = self.api_key()? {
            rb = rb.bearer_auth(key);
        }

        let resp = rb
            .send()
            .instrument(tracing::info_span!("provider.http", provider = %self.name, kind = "ollama_stream"))
            .await
            .with_context(|| format!("calling provider '{}'", self.name))?;
        let status = resp.status();
        if !status.is_success() {
            let text = resp.text().await?;
            bail!("provider '{}' returned {}: {}", self.name, status, text);
        }

        let (tx, rx) = tokio::sync::mpsc::channel(64);

        tokio::spawn(async move {
            let mut byte_stream = resp.bytes_stream();
            let mut buf = String::new();

            while let Some(chunk) = byte_stream.next().await {
                let bytes = match chunk {
                    Ok(b) => b,
                    Err(e) => { tx.send(Err(anyhow::anyhow!(e))).await.ok(); return; }
                };
                buf.push_str(&String::from_utf8_lossy(&bytes));

                while let Some(nl) = buf.find('\n') {
                    let line = buf[..nl].trim().to_owned();
                    buf.drain(..=nl);
                    if line.is_empty() { continue; }

                    let chunk_resp: ollama::OllamaChatResponse = match serde_json::from_str(&line) {
                        Ok(r) => r,
                        Err(e) => {
                            tx.send(Err(anyhow::anyhow!("parsing Ollama chunk: {e}"))).await.ok();
                            return;
                        }
                    };

                    if !chunk_resp.message.content.is_empty() {
                        let event = StreamEvent::TextDelta { text: chunk_resp.message.content };
                        if tx.send(Ok(event)).await.is_err() { return; }
                    }

                    if chunk_resp.done {
                        let event = StreamEvent::Done {
                            stop_reason: crate::canonical::StopReason::EndTurn,
                            usage: crate::canonical::Usage {
                                input_tokens: chunk_resp.prompt_eval_count,
                                output_tokens: chunk_resp.eval_count,
                            },
                        };
                        tx.send(Ok(event)).await.ok();
                        return;
                    }
                }
            }
        });

        Ok(ReceiverStream::new(rx))
    }

    /// Lists the models this provider currently has available, for
    /// [`crate::config::RouterRule::Discover`]. Only `ollama`-format
    /// providers support this (via `GET /api/tags`); others return an empty
    /// list.
    pub async fn list_models(&self, client: &Client) -> anyhow::Result<Vec<String>> {
        if self.format != ProviderFormat::Ollama {
            return Ok(Vec::new());
        }

        let url = format!("{}/api/tags", self.base_url);
        let mut rb = client.get(&url);
        if let Some(key) = self.api_key()? {
            rb = rb.bearer_auth(key);
        }

        let resp = rb
            .send()
            .instrument(tracing::info_span!("provider.http", provider = %self.name, kind = "list_models"))
            .await
            .with_context(|| format!("listing models for provider '{}'", self.name))?;
        let status = resp.status();
        let text = resp.text().await?;
        if !status.is_success() {
            bail!("provider '{}' returned {} listing models: {}", self.name, status, text);
        }

        let parsed: ollama::OllamaTagsResponse = serde_json::from_str(&text)
            .with_context(|| format!("parsing model list from provider '{}': {}", self.name, text))?;
        Ok(parsed.models.into_iter().map(|m| m.name).collect())
    }

    /// Capabilities `model` reports via `POST /api/show` (e.g. `"vision"`,
    /// `"tools"`), plus any inferred from its name/family that Ollama
    /// doesn't explicitly tag (e.g. `"coding"`) — see
    /// [`ollama::implicit_capabilities`]. Only `ollama`-format providers
    /// support this; others return an empty list.
    pub async fn model_capabilities(&self, client: &Client, model: &str) -> anyhow::Result<Vec<String>> {
        if self.format != ProviderFormat::Ollama {
            return Ok(Vec::new());
        }

        let url = format!("{}/api/show", self.base_url);
        let body = ollama::OllamaShowRequest { model: model.to_string() };
        let mut rb = client.post(&url).json(&body);
        if let Some(key) = self.api_key()? {
            rb = rb.bearer_auth(key);
        }

        let resp = rb
            .send()
            .instrument(tracing::info_span!("provider.http", provider = %self.name, kind = "model_capabilities"))
            .await
            .with_context(|| format!("fetching capabilities for model '{model}' on provider '{}'", self.name))?;
        let status = resp.status();
        let text = resp.text().await?;
        if !status.is_success() {
            bail!("provider '{}' returned {} fetching capabilities for '{model}': {}", self.name, status, text);
        }

        let parsed: ollama::OllamaShowResponse = serde_json::from_str(&text)
            .with_context(|| format!("parsing capabilities for model '{model}' from provider '{}': {}", self.name, text))?;

        let mut capabilities = parsed.capabilities;
        capabilities.extend(ollama::implicit_capabilities(model, &parsed.details.family));
        Ok(capabilities)
    }

    async fn send_openai(&self, client: &Client, req: &ChatRequest) -> anyhow::Result<ChatResponse> {
        let body = openai::OpenAiChatRequest::from(req);
        let url = format!("{}/chat/completions", self.base_url);

        let mut rb = client.post(&url).json(&body);
        if let Some(key) = self.api_key()? {
            rb = rb.bearer_auth(key);
        }

        let resp = rb
            .send()
            .instrument(tracing::info_span!("provider.http", provider = %self.name, kind = "openai_chat"))
            .await
            .with_context(|| format!("calling provider '{}'", self.name))?;
        let status = resp.status();
        let text = resp.text().await?;
        if !status.is_success() {
            bail!("provider '{}' returned {}: {}", self.name, status, text);
        }

        let parsed: openai::OpenAiChatResponse = serde_json::from_str(&text)
            .with_context(|| format!("parsing response from provider '{}': {}", self.name, text))?;
        Ok(parsed.into())
    }

    async fn send_anthropic(&self, client: &Client, req: &ChatRequest) -> anyhow::Result<ChatResponse> {
        let body = anthropic::AnthropicMessagesRequest::from(req);
        let url = format!("{}/messages", self.base_url);

        let mut rb = client
            .post(&url)
            .header("anthropic-version", "2023-06-01")
            .json(&body);
        if let Some(beta) = anthropic_beta_header(req) {
            rb = rb.header("anthropic-beta", beta);
        }
        if let Some(key) = self.api_key()? {
            rb = rb.header("x-api-key", key);
        }

        let resp = rb
            .send()
            .instrument(tracing::info_span!("provider.http", provider = %self.name, kind = "anthropic_messages"))
            .await
            .with_context(|| format!("calling provider '{}'", self.name))?;
        let status = resp.status();
        let text = resp.text().await?;
        if !status.is_success() {
            bail!("provider '{}' returned {}: {}", self.name, status, text);
        }

        let parsed: anthropic::AnthropicMessagesResponse = serde_json::from_str(&text)
            .with_context(|| format!("parsing response from provider '{}': {}", self.name, text))?;
        Ok(parsed.into())
    }

    async fn send_ollama(&self, client: &Client, req: &ChatRequest) -> anyhow::Result<ChatResponse> {
        let body = ollama::OllamaChatRequest::from(req);
        let url = format!("{}/api/chat", self.base_url);

        let mut rb = client.post(&url).json(&body);
        if let Some(key) = self.api_key()? {
            rb = rb.bearer_auth(key);
        }

        let resp = rb
            .send()
            .instrument(tracing::info_span!("provider.http", provider = %self.name, kind = "ollama_chat"))
            .await
            .with_context(|| format!("calling provider '{}'", self.name))?;
        let status = resp.status();
        let text = resp.text().await?;
        if !status.is_success() {
            bail!("provider '{}' returned {}: {}", self.name, status, text);
        }

        let parsed: ollama::OllamaChatResponse = serde_json::from_str(&text)
            .with_context(|| format!("parsing response from provider '{}': {}", self.name, text))?;
        Ok(parsed.into())
    }
}

/// `anthropic-beta` header value(s) required to forward `req`'s Anthropic-only
/// fields, or `None` if the request uses no beta features. Currently only
/// `task_budget` is beta-gated — `thinking`/`effort` need no header.
fn anthropic_beta_header(req: &ChatRequest) -> Option<&'static str> {
    if req.task_budget.is_some() {
        Some("task-budgets-2026-03-13")
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn request(task_budget: Option<serde_json::Value>) -> ChatRequest {
        ChatRequest {
            model: "claude-opus-4-8".to_string(),
            system: None,
            messages: Vec::new(),
            max_tokens: Some(1024),
            temperature: None,
            thinking: None,
            effort: None,
            task_budget,
            output_schema: None,
            tools: Vec::new(),
            stream: false,
            plugins: Vec::new(),
            forced_provider: None,
            tags: Vec::new(),
        }
    }

    #[test]
    fn no_beta_header_without_task_budget() {
        assert_eq!(anthropic_beta_header(&request(None)), None);
    }

    #[test]
    fn task_budget_requires_beta_header() {
        let req = request(Some(json!({"type": "tokens", "total": 64000})));
        assert_eq!(anthropic_beta_header(&req), Some("task-budgets-2026-03-13"));
    }

    #[test]
    fn stream_flag_flows_into_every_outbound_format() {
        let mut req = request(None);
        req.stream = true;

        assert!(openai::OpenAiChatRequest::from(&req).stream);
        assert!(anthropic::AnthropicMessagesRequest::from(&req).stream);
        assert!(ollama::OllamaChatRequest::from(&req).stream);
    }

    #[test]
    fn stream_false_flows_into_every_outbound_format() {
        let req = request(None);

        assert!(!openai::OpenAiChatRequest::from(&req).stream);
        assert!(!anthropic::AnthropicMessagesRequest::from(&req).stream);
        assert!(!ollama::OllamaChatRequest::from(&req).stream);
    }
}
