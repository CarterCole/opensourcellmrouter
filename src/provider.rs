//! A configured upstream LLM backend, and the logic to call it.

use anyhow::{bail, Context};
use reqwest::Client;

use crate::canonical::{ChatRequest, ChatResponse};
use crate::config::{ProviderConfig, ProviderFormat};
use crate::formats::{anthropic, ollama, openai};

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
