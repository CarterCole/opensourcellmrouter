//! `llm`: asks a chat model — any configured provider's, e.g. Claude Haiku
//! through an `anthropic` provider — which of the configured models should
//! answer a request.
//!
//! Config:
//!
//! ```toml
//! [classifiers.llm]
//! enabled  = true
//! provider = "anthropic"          # the provider to ask
//! model    = "claude-haiku-5-5"   # and which of its models
//!
//! [[classifiers.llm.models]]
//! provider    = "anthropic"
//! model       = "claude-sonnet-5-5"
//! description = "Hard problems: architecture, tricky bugs, long documents"
//!
//! [[classifiers.llm.models]]
//! provider    = "ollama"
//! model       = "llama3.1:8b"
//! description = "General questions, writing, summarizing and conversation"
//!
//! [[routers]]
//! type = "llm"
//! ```
//!
//! For each request, `model` gets the options — name and description — in
//! its system prompt, and the request's system prompt and latest user
//! message, each cut to `max_chars`, as its message, and replies with the
//! name of one option. That pick is attached to the request as
//! [`ChatRequest::route`], and a [`crate::config::RouterRule::Llm`] rule
//! routes on it.
//!
//! Unlike `cev`'s, a pick has no probabilities (its `confidence` is
//! always 1) and no decision of its own to correct through `/v1/feedback`.
//! With `cev` enabled too, it labels cev's decision about the same request
//! instead, which cev learns from (see [`super::cev`]). Every request costs
//! a call to `provider`, and waits for it before being routed.
//!
//! Settings:
//!
//! | Key | Default | |
//! |---|---|---|
//! | `provider` | | Name of the provider to ask. If it has an `api_key_env`, that variable has to be set. |
//! | `model` | | The model to ask, as that provider names it. |
//! | `models` | | The options to choose between; see below. |
//! | `instructions` | "Which model is best suited to answer this request?" | The question. |
//! | `timeout_ms` | `5000` | How long to wait for an answer before routing without one. |
//! | `max_chars` | `2000` | How much of the system prompt and of the message to send. |
//! | `max_tokens` | `64` | Cap on the answer. Raise it for a model that thinks before answering. |
//!
//! `models` are the same as `cev`'s: each has a `provider` and `model` to
//! route to, an optional `description` of what the model is good at, and an
//! optional `name` to pick it by, which defaults to `model`. An option whose
//! provider has an `api_key_env` that isn't set is dropped, the same way
//! the router drops the provider itself.

use std::time::Duration;

use anyhow::{Context, bail};
use async_trait::async_trait;
use indexmap::IndexMap;
use serde::Deserialize;
use serde_json::{Map, Value};

use super::{Classification, Classifier, ClassifierContext, ModelOption, check_models, clip, latest_user_text};
use crate::canonical::{ChatRequest, Message, Role, RouteChoice};
use crate::config::ProviderConfig;
use crate::provider::Provider;

#[derive(Debug, Deserialize)]
#[serde(default, deny_unknown_fields)]
struct Settings {
    provider: String,
    model: String,
    models: Vec<ModelOption>,
    instructions: String,
    timeout_ms: u64,
    max_chars: usize,
    max_tokens: u32,
}

impl Default for Settings {
    fn default() -> Self {
        Settings {
            provider: String::new(),
            model: String::new(),
            models: Vec::new(),
            instructions: "Which model is best suited to answer this request?".to_string(),
            timeout_ms: 5000,
            max_chars: 2000,
            max_tokens: 64,
        }
    }
}

pub struct LlmClassifier {
    provider: Provider,
    client: reqwest::Client,
    model: String,
    /// System prompt listing `models`, the same for every request.
    system: String,
    models: Vec<ModelOption>,
    timeout: Duration,
    max_chars: usize,
    max_tokens: u32,
}

/// Whether `$var` is set to something, as [`crate::router::ModelRouter`]
/// requires of a provider's `api_key_env`.
fn key_is_set(var: &str) -> bool {
    std::env::var(var).is_ok_and(|key| !key.is_empty())
}

impl LlmClassifier {
    /// Builds the classifier from its `[classifiers.llm]` settings.
    /// `providers` are the configured providers, which `provider` and every
    /// model option have to name one of.
    pub fn from_settings(settings: &Map<String, Value>, providers: &[ProviderConfig]) -> anyhow::Result<Self> {
        let mut settings: Settings = serde_json::from_value(Value::Object(settings.clone()))?;
        if settings.provider.is_empty() || settings.model.is_empty() {
            bail!("set `provider` and `model`: which provider to ask, and which of its models");
        }

        let provider = providers
            .iter()
            .find(|p| p.name == settings.provider)
            .with_context(|| format!("`provider` names unknown provider '{}'", settings.provider))?;
        if let Some(var) = &provider.api_key_env {
            if !key_is_set(var) {
                bail!("its provider '{}' needs ${var}, which is not set", provider.name);
            }
        }

        check_models(&settings.models, providers)?;
        settings.models.retain(|option| {
            let config = providers.iter().find(|p| p.name == option.provider);
            match config.and_then(|p| p.api_key_env.as_deref()) {
                Some(var) if !key_is_set(var) => {
                    tracing::warn!(
                        "classifier 'llm' won't pick model '{}': provider '{}' is skipped, ${var} is not set",
                        option.name(),
                        option.provider
                    );
                    false
                }
                _ => true,
            }
        });

        Self::new(Provider::from_config(provider), settings)
    }

    fn new(provider: Provider, settings: Settings) -> anyhow::Result<Self> {
        if settings.models.len() < 2 {
            bail!("needs at least two `models` to choose between");
        }

        Ok(LlmClassifier {
            provider,
            client: reqwest::Client::new(),
            model: settings.model,
            system: system_prompt(&settings.instructions, &settings.models),
            models: settings.models,
            timeout: Duration::from_millis(settings.timeout_ms),
            max_chars: settings.max_chars,
            max_tokens: settings.max_tokens,
        })
    }

    /// The provider and model this classifier asks.
    pub fn asked(&self) -> (&str, &str) {
        (&self.provider.name, &self.model)
    }

    /// The options it picks between.
    pub fn models(&self) -> &[ModelOption] {
        &self.models
    }

    /// What the model is shown of `req`: the system prompt and the latest
    /// user message, each cut to `max_chars`.
    fn message(&self, req: &ChatRequest) -> String {
        let mut message = String::new();
        if let Some(system) = &req.system {
            message.push_str(&format!("<system_prompt>\n{}\n</system_prompt>\n", clip(system, self.max_chars)));
        }
        let request = latest_user_text(req);
        message.push_str(&format!("<request>\n{}\n</request>", clip(&request, self.max_chars)));
        message
    }

    /// The option `reply` names: all of it, give or take quotes and
    /// punctuation, or failing that the only option it mentions. A
    /// reasoning model's `<think>…</think>` is skipped.
    fn parse(&self, reply: &str) -> Option<&ModelOption> {
        let reply = reply.rsplit_once("</think>").map_or(reply, |(_, answer)| answer);
        let answer = reply.trim().trim_matches(|c: char| matches!(c, '"' | '\'' | '`' | '*' | '.')).trim();
        if let Some(option) = self.models.iter().find(|m| m.name().eq_ignore_ascii_case(answer)) {
            return Some(option);
        }

        let reply = reply.to_lowercase();
        let mentioned: Vec<String> = self
            .models
            .iter()
            .map(|m| m.name().to_lowercase())
            .filter(|name| reply.contains(name.as_str()))
            .collect();
        // A name only mentioned as part of a longer one ("llama3.1:8b" in
        // "llama3.1:8b-instruct") doesn't count.
        let mut named = self.models.iter().filter(|m| {
            let name = m.name().to_lowercase();
            mentioned.contains(&name) && !mentioned.iter().any(|other| other.len() > name.len() && other.contains(&name))
        });
        match (named.next(), named.next()) {
            (Some(only), None) => Some(only),
            _ => None,
        }
    }
}

/// Tells the model what it's choosing between, and how to answer.
fn system_prompt(instructions: &str, models: &[ModelOption]) -> String {
    let mut prompt = format!(
        "You route requests to the model best suited to answer them; you never answer them yourself. \
         {instructions}\n\nModels:\n"
    );
    for option in models {
        match &option.description {
            Some(description) => prompt.push_str(&format!("- {}: {description}\n", option.name())),
            None => prompt.push_str(&format!("- {}\n", option.name())),
        }
    }
    prompt.push_str(
        "\nThe request is in <request>, after its system prompt in <system_prompt> if it has one. \
         Reply with the name of one model, exactly as written above, and nothing else.",
    );
    prompt
}

#[async_trait]
impl Classifier for LlmClassifier {
    fn id(&self) -> &'static str {
        "llm"
    }

    async fn classify(&self, _ctx: &ClassifierContext, req: &ChatRequest) -> anyhow::Result<Classification> {
        let question = ChatRequest {
            model: self.model.clone(),
            system: Some(self.system.clone()),
            messages: vec![Message::text(Role::User, self.message(req))],
            max_tokens: Some(self.max_tokens),
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
            routes: Vec::new(),
        };

        let reply = tokio::time::timeout(self.timeout, self.provider.send(&self.client, &question))
            .await
            .with_context(|| format!("no answer from provider '{}' within {:?}", self.provider.name, self.timeout))??;
        let picked = self.parse(&reply.content).with_context(|| {
            format!("'{}' answered {:?}, which doesn't name one of `models`", self.model, reply.content)
        })?;

        Ok(Classification {
            tags: Vec::new(),
            route: Some(RouteChoice {
                classifier: "llm".to_string(),
                decision_id: None,
                choice: picked.name().to_string(),
                provider: picked.provider.clone(),
                model: picked.model.clone(),
                confidence: 1.0,
                probabilities: IndexMap::new(),
                adapted: false,
            }),
        })
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use std::sync::{Arc, Mutex};

    use axum::routing::post;
    use axum::{Json, Router};
    use serde_json::json;

    use super::*;
    use crate::config::Config;

    /// Serves an OpenAI-compatible provider that answers every request with
    /// `answer`, and returns its base URL and the requests it has received.
    pub(crate) async fn mock_provider(answer: &'static str) -> (String, Arc<Mutex<Vec<Value>>>) {
        let received = Arc::new(Mutex::new(Vec::new()));
        let log = received.clone();
        let app = Router::new().route(
            "/v1/chat/completions",
            post(move |Json(body): Json<Value>| async move {
                log.lock().unwrap().push(body.clone());
                Json(json!({
                    "id": "chatcmpl-1",
                    "object": "chat.completion",
                    "model": body["model"],
                    "choices": [{
                        "index": 0,
                        "message": {"role": "assistant", "content": answer},
                        "finish_reason": "stop",
                    }],
                }))
            }),
        );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let url = format!("http://{}/v1", listener.local_addr().unwrap());
        tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        (url, received)
    }

    fn providers(router_url: &str, extra: &str) -> Vec<ProviderConfig> {
        toml::from_str::<Config>(&format!(
            r#"
            [[providers]]
            name = "router"
            format = "openai"
            base_url = "{router_url}"

            [[providers]]
            name = "ollama"
            format = "ollama"
            base_url = "http://localhost:11434"
            {extra}
            "#
        ))
        .unwrap()
        .providers
    }

    fn settings(value: Value) -> Map<String, Value> {
        value.as_object().unwrap().clone()
    }

    fn two_models() -> Value {
        json!([
            {"provider": "ollama", "model": "deepseek-r1:latest", "description": "Programming and debugging"},
            {"provider": "ollama", "model": "llama3.1:8b", "name": "general", "description": "Conversation"},
        ])
    }

    /// Asks `router-model` at `router_url` to pick between the same models
    /// as [`super::super::cev::tests::mock_classifier`].
    pub(crate) fn classifier(router_url: &str) -> LlmClassifier {
        let settings = settings(json!({"provider": "router", "model": "router-model", "models": two_models()}));
        LlmClassifier::from_settings(&settings, &providers(router_url, "")).unwrap()
    }

    fn request(system: Option<&str>, messages: Vec<Message>) -> ChatRequest {
        ChatRequest {
            model: "gpt-4".to_string(),
            system: system.map(str::to_string),
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
            routes: Vec::new(),
        }
    }

    #[tokio::test]
    async fn asks_the_provider_and_routes_to_the_model_it_names() {
        let (url, received) = mock_provider("general").await;
        let classifier = classifier(&url);
        let ctx = ClassifierContext { settings: Map::new() };

        let req = request(Some("Be brief"), vec![Message::text(Role::User, "Any recipes for my travel?")]);
        let route = classifier.classify(&ctx, &req).await.unwrap().route.unwrap();
        assert_eq!(route.classifier, "llm");
        assert_eq!(route.choice, "general");
        assert_eq!((route.provider.as_str(), route.model.as_str()), ("ollama", "llama3.1:8b"));
        assert_eq!(route.decision_id, None);

        // The question went to the configured model, with the options in
        // the system prompt and the request in the message.
        let asked = received.lock().unwrap()[0].clone();
        assert_eq!(asked["model"], "router-model");
        assert_eq!(asked["max_tokens"], 64);
        let system = asked["messages"][0]["content"].as_str().unwrap();
        assert!(system.contains("- deepseek-r1:latest: Programming and debugging\n- general: Conversation\n"), "{system}");
        assert_eq!(
            asked["messages"][1]["content"],
            "<system_prompt>\nBe brief\n</system_prompt>\n<request>\nAny recipes for my travel?\n</request>"
        );
    }

    #[tokio::test]
    async fn an_answer_that_names_no_model_is_an_error() {
        let (url, _) = mock_provider("I'd suggest something with more parameters.").await;
        let classifier = classifier(&url);
        let ctx = ClassifierContext { settings: Map::new() };

        let req = request(None, vec![Message::text(Role::User, "hi")]);
        let err = classifier.classify(&ctx, &req).await.unwrap_err();
        assert!(err.to_string().contains("doesn't name one of `models`"), "{err}");
    }

    #[test]
    fn parses_a_name_out_of_a_chatty_answer() {
        let classifier = classifier("http://localhost:1/v1");
        let parse = |reply| classifier.parse(reply).map(|m| m.name());

        assert_eq!(parse("general"), Some("general"));
        assert_eq!(parse("  `General`.\n"), Some("general"));
        assert_eq!(parse("\"deepseek-r1:latest\""), Some("deepseek-r1:latest"));
        assert_eq!(parse("I'd pick deepseek-r1:latest for this."), Some("deepseek-r1:latest"));
        assert_eq!(parse("<think>general, or maybe deepseek-r1:latest?</think>\ngeneral"), Some("general"));
        // Naming both, or neither, is no answer.
        assert_eq!(parse("general or deepseek-r1:latest"), None);
        assert_eq!(parse("gpt-4"), None);
    }

    #[test]
    fn a_name_inside_a_longer_one_is_not_a_mention() {
        let settings = settings(json!({"provider": "router", "model": "m", "models": [
            {"provider": "ollama", "model": "llama3.1:8b"},
            {"provider": "ollama", "model": "llama3.1:8b-instruct"},
        ]}));
        let classifier = LlmClassifier::from_settings(&settings, &providers("http://localhost:1/v1", "")).unwrap();

        assert_eq!(classifier.parse("use llama3.1:8b-instruct").map(|m| m.name()), Some("llama3.1:8b-instruct"));
        assert_eq!(classifier.parse("use llama3.1:8b").map(|m| m.name()), Some("llama3.1:8b"));
    }

    #[test]
    fn the_latest_user_message_with_text_is_classified() {
        let classifier = classifier("http://localhost:1/v1");
        let mut req = request(None, vec![Message::text(Role::User, "Fix the failing test")]);
        req.messages.push(Message::text(Role::Assistant, "Running it"));
        // A turn that only carries a tool result has no text of its own.
        req.messages.push(Message { role: Role::User, content: Vec::new() });

        assert_eq!(classifier.message(&req), "<request>\nFix the failing test\n</request>");
    }

    #[test]
    fn needs_a_usable_provider_and_two_models() {
        const UNSET: &str = "OPENSOURCELLMROUTER_TEST_UNSET_KEY";
        let build = |value: Value, extra: &str| {
            LlmClassifier::from_settings(&settings(value), &providers("http://localhost:1/v1", extra)).map(|_| ())
        };
        let locked = format!(
            r#"
            [[providers]]
            name = "locked"
            format = "anthropic"
            base_url = "https://api.anthropic.com/v1"
            api_key_env = "{UNSET}"
            "#
        );

        let err = build(json!({"model": "m", "models": two_models()}), "").unwrap_err();
        assert!(err.to_string().contains("set `provider` and `model`"), "{err}");

        let err = build(json!({"provider": "nope", "model": "m", "models": two_models()}), "").unwrap_err();
        assert!(err.to_string().contains("unknown provider 'nope'"), "{err}");

        let err = build(json!({"provider": "locked", "model": "m", "models": two_models()}), &locked).unwrap_err();
        assert!(err.to_string().contains(&format!("${UNSET}")), "{err}");

        // An option on a provider without its key is dropped, which here
        // leaves too few to choose between.
        let mut models = two_models();
        models[1]["provider"] = json!("locked");
        let err = build(json!({"provider": "router", "model": "m", "models": models}), &locked).unwrap_err();
        assert!(err.to_string().contains("at least two"), "{err}");
    }
}
