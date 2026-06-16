//! A configured upstream LLM backend, and the logic to call it.

use anyhow::{bail, Context};
use bytes::Bytes;
use reqwest::Client;
use tokio_stream::StreamExt;
use tokio_stream::wrappers::ReceiverStream;

use crate::canonical::{ChatRequest, ChatResponse};
use crate::config::{ProviderConfig, ProviderFormat};
use crate::formats::{anthropic, ollama, openai};

/// A stream of SSE-formatted strings in OpenAI chunk format, suitable for
/// proxying directly to the client as `text/event-stream`.
pub type ChunkStream = ReceiverStream<anyhow::Result<Bytes>>;

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
            ProviderFormat::Anthropic => bail!("streaming not supported for Anthropic format"),
        }
    }

    /// Proxies the upstream's own `text/event-stream` response directly.
    async fn stream_openai(&self, client: &Client, req: &ChatRequest) -> anyhow::Result<ChunkStream> {
        let mut body = openai::OpenAiChatRequest::from(req);
        body.stream = true;
        let url = format!("{}/chat/completions", self.base_url);

        let mut rb = client.post(&url).json(&body);
        if let Some(key) = self.api_key()? {
            rb = rb.bearer_auth(key);
        }

        let resp = rb.send().await.with_context(|| format!("calling provider '{}'", self.name))?;
        let status = resp.status();
        if !status.is_success() {
            let text = resp.text().await?;
            bail!("provider '{}' returned {}: {}", self.name, status, text);
        }

        let (tx, rx) = tokio::sync::mpsc::channel(64);
        tokio::spawn(async move {
            let mut stream = resp.bytes_stream();
            while let Some(chunk) = stream.next().await {
                match chunk {
                    Ok(b) => { if tx.send(Ok(b)).await.is_err() { return; } }
                    Err(e) => { tx.send(Err(anyhow::anyhow!(e))).await.ok(); return; }
                }
            }
        });

        Ok(ReceiverStream::new(rx))
    }

    /// Translates Ollama's NDJSON streaming format into OpenAI SSE chunks.
    async fn stream_ollama(&self, client: &Client, req: &ChatRequest) -> anyhow::Result<ChunkStream> {
        let mut body = ollama::OllamaChatRequest::from(req);
        body.stream = true;
        let url = format!("{}/api/chat", self.base_url);

        let mut rb = client.post(&url).json(&body);
        if let Some(key) = self.api_key()? {
            rb = rb.bearer_auth(key);
        }

        let resp = rb.send().await.with_context(|| format!("calling provider '{}'", self.name))?;
        let status = resp.status();
        if !status.is_success() {
            let text = resp.text().await?;
            bail!("provider '{}' returned {}: {}", self.name, status, text);
        }

        let (tx, rx) = tokio::sync::mpsc::channel(64);
        let model_name = req.model.clone();

        tokio::spawn(async move {
            let mut byte_stream = resp.bytes_stream();
            let mut buf = String::new();
            let mut chunk_id: u64 = 0;

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

                    chunk_id += 1;
                    let done = chunk_resp.done;
                    let sse = serde_json::json!({
                        "id": format!("ollama-{model_name}-{chunk_id}"),
                        "object": "chat.completion.chunk",
                        "model": &model_name,
                        "choices": [{
                            "index": 0,
                            "delta": {"content": chunk_resp.message.content},
                            "finish_reason": if done { serde_json::Value::String("stop".to_string()) } else { serde_json::Value::Null },
                        }]
                    });

                    let line_out = format!("data: {sse}\n\n");
                    if tx.send(Ok(Bytes::from(line_out))).await.is_err() { return; }

                    if done {
                        tx.send(Ok(Bytes::from("data: [DONE]\n\n"))).await.ok();
                        return;
                    }
                }
            }

            tx.send(Ok(Bytes::from("data: [DONE]\n\n"))).await.ok();
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

    async fn send_openai(&self, client: &Client, req: &ChatRequest) -> anyhow::Result<ChatResponse> {
        let body = openai::OpenAiChatRequest::from(req);
        let url = format!("{}/chat/completions", self.base_url);

        let mut rb = client.post(&url).json(&body);
        if let Some(key) = self.api_key()? {
            rb = rb.bearer_auth(key);
        }

        let resp = rb
            .send()
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
        if let Some(key) = self.api_key()? {
            rb = rb.header("x-api-key", key);
        }

        let resp = rb
            .send()
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
