use std::collections::HashMap;
use std::path::Path;

use anyhow::Context;
use serde::Deserialize;

#[derive(Debug, Deserialize)]
pub struct Config {
    #[serde(default)]
    pub server: ServerConfig,
    #[serde(default)]
    pub logging: LoggingConfig,
    pub providers: Vec<ProviderConfig>,
    /// Routing rules, evaluated in order. The first rule that matches for a
    /// given request "wins" and decides the provider; a rule that doesn't
    /// match passes through to the next one. See [`RouterRule`].
    #[serde(default)]
    pub routers: Vec<RouterRule>,
    /// Plugins available to requests, keyed by plugin id (e.g.
    /// `[plugins.response-healing]`). See [`crate::plugins`].
    #[serde(default)]
    pub plugins: HashMap<String, PluginEntryConfig>,
    /// Classifiers run on every request before routing, keyed by classifier
    /// id (e.g. `[classifiers.keyword]`). See [`crate::classifiers`].
    #[serde(default)]
    pub classifiers: HashMap<String, ClassifierEntryConfig>,
}

/// Server-side configuration for one plugin or classifier: whether it's
/// enabled, plus arbitrary settings specific to that plugin/classifier.
#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default)]
pub struct EntryConfig {
    pub enabled: bool,
    #[serde(flatten)]
    pub settings: serde_json::Map<String, serde_json::Value>,
}

/// Whether a plugin runs by default for every request, plus settings merged
/// with (and overridable by) a per-request `plugins` array entry of the same
/// id.
pub type PluginEntryConfig = EntryConfig;

/// Whether a classifier runs on every request, plus its settings (e.g. the
/// `keyword` classifier's tag/keyword lists).
pub type ClassifierEntryConfig = EntryConfig;

/// One entry of the `routers` chain. Each variant either resolves a request
/// to a provider, or "passes through" (returns no decision) so the next
/// rule in the chain gets a chance.
///
/// `providers` lists restrict which providers a rule considers; an empty
/// list (the default) means "every configured provider".
#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum RouterRule {
    /// Matches if the request's `model` starts with `model_prefix`, and
    /// routes to `provider`. If `rewrite_model` is set, it replaces `model`
    /// before forwarding.
    Prefix {
        model_prefix: String,
        provider: String,
        rewrite_model: Option<String>,
    },
    /// Matches if `tag` is among the tags assigned by
    /// [`crate::classifiers`] (e.g. `"vision"`, `"nsfw"`, `"tools"`), and
    /// routes to `provider`. If `rewrite_model` is set, it replaces `model`
    /// before forwarding.
    Tag {
        tag: String,
        provider: String,
        rewrite_model: Option<String>,
    },
    /// Picks the candidate with the lowest `cost_per_1m_tokens`. If
    /// `max_cost_per_1m_tokens` is set, candidates above it are excluded;
    /// if that leaves no candidates, this rule passes through.
    Price {
        #[serde(default)]
        providers: Vec<String>,
        max_cost_per_1m_tokens: Option<f64>,
    },
    /// Picks the candidate with the lowest `latency_ms`. Candidates that
    /// don't declare `latency_ms` are excluded. If `max_latency_ms` is set,
    /// candidates above it are also excluded; if that leaves no candidates,
    /// this rule passes through.
    Latency {
        #[serde(default)]
        providers: Vec<String>,
        max_latency_ms: Option<f64>,
    },
    /// Picks the candidate with the highest `throughput_tokens_per_sec`.
    /// Candidates that don't declare it are excluded. If
    /// `min_tokens_per_sec` is set, candidates below it are also excluded;
    /// if that leaves no candidates, this rule passes through.
    Throughput {
        #[serde(default)]
        providers: Vec<String>,
        min_tokens_per_sec: Option<f64>,
    },
    /// Always matches (assuming at least one candidate exists): ranks
    /// candidates by `score = quality_bias * quality - (1 - quality_bias) *
    /// cost_per_1m_tokens` and picks the highest. A sensible chain terminator.
    Fallback {
        #[serde(default)]
        providers: Vec<String>,
        #[serde(default = "default_quality_bias")]
        quality_bias: f64,
        /// If set, replaces the model name before forwarding (useful when the
        /// fallback provider requires a specific model name).
        rewrite_model: Option<String>,
    },
    /// Matches if `model` is one of the models `provider` reports having
    /// available, as discovered at startup via
    /// [`crate::router::ModelRouter::discover_models`] (currently only
    /// implemented for `ollama`-format providers, via `GET /api/tags`). If
    /// discovery hasn't run, failed, or didn't report `model`, this rule
    /// passes through.
    Discover { provider: String },
    /// Always matches: picks a provider at random from `providers` (or all
    /// configured providers if `providers` is empty). If `rewrite_model` is
    /// set, it replaces `model` before forwarding.
    ///
    /// If `candidates` is non-empty, picks a `{provider, model}` pair at
    /// random from that list instead, ignoring `providers`/`rewrite_model`.
    Random {
        #[serde(default)]
        providers: Vec<String>,
        rewrite_model: Option<String>,
        #[serde(default)]
        candidates: Vec<RandomCandidate>,
    },
}

#[derive(Debug, Clone, Deserialize)]
pub struct RandomCandidate {
    pub provider: String,
    pub model: String,
}

fn default_quality_bias() -> f64 {
    0.5
}

/// Records every request and response handled by the router as a line of
/// JSON, for debugging and auditing.
#[derive(Debug, Deserialize)]
#[serde(default)]
pub struct LoggingConfig {
    pub enabled: bool,
    pub path: String,
}

impl Default for LoggingConfig {
    fn default() -> Self {
        LoggingConfig {
            enabled: false,
            path: "logs/requests.jsonl".to_string(),
        }
    }
}

#[derive(Debug, Deserialize)]
#[serde(default)]
pub struct ServerConfig {
    pub host: String,
    pub port: u16,
    /// Whether to serve the live request dashboard at `/dashboard` (and its
    /// `/dashboard/events` SSE feed).
    pub dashboard: bool,
}

impl Default for ServerConfig {
    fn default() -> Self {
        ServerConfig {
            host: "0.0.0.0".to_string(),
            port: 8090,
            dashboard: false,
        }
    }
}

/// The wire format a provider's API speaks.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ProviderFormat {
    OpenAi,
    Anthropic,
    /// Ollama's native API (`/api/chat`, `/api/tags`). `base_url` is
    /// Ollama's root, e.g. `http://localhost:11434` (no `/v1` suffix).
    Ollama,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ProviderConfig {
    /// Name used to refer to this provider from `routers`.
    pub name: String,
    pub format: ProviderFormat,
    /// Base URL, e.g. `http://localhost:8080/v1` or `https://api.openai.com/v1`.
    pub base_url: String,
    /// Name of an environment variable holding the API key, if required.
    pub api_key_env: Option<String>,
    /// If true, the provider is skipped at startup when `api_key_env` is unset.
    /// If false (default), a warning is logged but the provider is kept — the
    /// first request that hits it will fail instead of startup.
    #[serde(default)]
    pub strict: bool,
    /// Blended cost in USD per 1M tokens. Used by `price` and `fallback` routers.
    #[serde(default)]
    pub cost_per_1m_tokens: f64,
    /// Subjective quality score (e.g. 0-100). Used by `fallback` routers.
    #[serde(default)]
    pub quality: f64,
    /// Typical response latency in milliseconds. Used by `latency` routers;
    /// providers that don't declare this are skipped by such rules.
    pub latency_ms: Option<f64>,
    /// Typical throughput in output tokens/sec. Used by `throughput`
    /// routers; providers that don't declare this are skipped by such rules.
    pub throughput_tokens_per_sec: Option<f64>,
}

impl Config {
    pub fn load(path: &Path) -> anyhow::Result<Self> {
        let text = std::fs::read_to_string(path)
            .with_context(|| format!("reading config file {}", path.display()))?;
        toml::from_str(&text).with_context(|| format!("parsing config file {}", path.display()))
    }
}
