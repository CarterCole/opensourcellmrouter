//! `cev`: asks a [cev](https://github.com/CarterCole/rust-cev) model which
//! of the configured models should answer a request, and learns from
//! corrections.
//!
//! Config:
//!
//! ```toml
//! [classifiers.cev]
//! enabled = true
//! url     = "http://127.0.0.1:8081"   # a running `cev` server
//!
//! [[classifiers.cev.models]]
//! provider    = "ollama"
//! model       = "deepseek-r1:latest"
//! description = "Programming, debugging, math and step-by-step reasoning"
//!
//! [[classifiers.cev.models]]
//! provider    = "ollama"
//! model       = "llama3.1:8b"
//! description = "General questions, writing, summarizing and conversation"
//!
//! [[routers]]
//! type = "cev"
//! ```
//!
//! Each request becomes one cev *choice* question — the same request
//! `cev::Cev::choose` sends, with the options read from config instead of
//! derived from an enum — about the system prompt and the latest user
//! message. The answer is one of `models`, with a probability for each; it's
//! attached to the request as [`ChatRequest::route`], and a
//! [`crate::config::RouterRule::Cev`] rule routes on it.
//!
//! Every answer has a decision id, returned to the client in the
//! `x-cev-decision-id` response header and recorded in the request log.
//! Posting it back to `/v1/feedback` with the model that should have been
//! picked (see [`CevClassifier::feedback`]) labels the decision, and cev's
//! online adapter for `task` learns from the label straight away.
//!
//! Settings:
//!
//! | Key | Default | |
//! |---|---|---|
//! | `url` | | Base URL of a `cev` server. Set this or `model`. |
//! | `api_key_env` | | Environment variable holding the server's bearer token. |
//! | `model` | | Qwen3 checkpoint (Hugging Face repo id or directory) to run in-process instead of `url`. |
//! | `db` | `cev.db` | With `model`: SQLite file for decisions, feedback and adapters. |
//! | `models` | | The options to choose between; see below. |
//! | `instructions` | "Which model is best suited to answer this request?" | The question. |
//! | `task` | `llmrouter.route` | Key of the online adapter that learns from feedback. |
//! | `timeout_ms` | `5000` | How long to wait for cev before routing without it. |
//! | `max_chars` | `2000` | How much of the system prompt and of the message to send. |
//!
//! Each of `models` has a `provider` and `model` to route to, an optional
//! `description` of what the model is good at (cev sees it next to the
//! name), and an optional `name` to pick and label it by, which defaults to
//! `model`.
//!
//! `model` needs a build with `--features cev-local`, which runs it on the
//! GPU on macOS (add `--features cuda` for an NVIDIA one); `url` works in
//! any build.

use std::time::Duration;

use anyhow::{Context, bail};
use async_trait::async_trait;
use cev::{Cev, FeedbackResponse, Label, SystemOneRequest};
use cev_core::{Answer, Question, QuestionType};
use indexmap::IndexMap;
use serde::Deserialize;
use serde_json::{Map, Value};

use super::{Classification, Classifier, ClassifierContext, ModelOption, check_models, clip, latest_user_text};
use crate::canonical::{ChatRequest, RouteChoice};
use crate::config::ProviderConfig;

/// Id of the one question in each request to cev.
const QUESTION: &str = "route";

#[derive(Debug, Deserialize)]
#[serde(default, deny_unknown_fields)]
struct Settings {
    url: Option<String>,
    api_key_env: Option<String>,
    model: Option<String>,
    db: String,
    models: Vec<ModelOption>,
    instructions: String,
    task: String,
    timeout_ms: u64,
    max_chars: usize,
}

impl Default for Settings {
    fn default() -> Self {
        Settings {
            url: None,
            api_key_env: None,
            model: None,
            db: "cev.db".to_string(),
            models: Vec::new(),
            instructions: "Which model is best suited to answer this request?".to_string(),
            task: "llmrouter.route".to_string(),
            timeout_ms: 5000,
            max_chars: 2000,
        }
    }
}

pub struct CevClassifier {
    cev: Cev,
    /// The question asked about every request.
    question: Question,
    models: Vec<ModelOption>,
    timeout: Duration,
    max_chars: usize,
}

impl CevClassifier {
    /// Builds the classifier from its `[classifiers.cev]` settings.
    /// `providers` are the configured providers, which every model option
    /// has to name one of.
    pub fn from_settings(settings: &Map<String, Value>, providers: &[ProviderConfig]) -> anyhow::Result<Self> {
        let settings: Settings = serde_json::from_value(Value::Object(settings.clone()))?;

        let cev = match (&settings.url, &settings.model) {
            (Some(url), None) => {
                let cev = Cev::http(url.as_str());
                match &settings.api_key_env {
                    Some(var) => {
                        let key = std::env::var(var).with_context(|| format!("${var} is not set"))?;
                        cev.with_api_key(key)
                    }
                    None => cev,
                }
            }
            (None, Some(model)) => local(model, &settings.db)?,
            _ => bail!("set either `url` (a running cev server) or `model` (to run one in-process)"),
        };

        check_models(&settings.models, providers)?;
        Self::new(cev, settings)
    }

    fn new(cev: Cev, settings: Settings) -> anyhow::Result<Self> {
        if settings.models.len() < 2 {
            bail!("needs at least two `models` to choose between");
        }

        let criteria = settings
            .models
            .iter()
            .map(|m| (m.name().to_string(), m.description.clone().map_or(Value::Null, Value::String)))
            .collect();

        Ok(CevClassifier {
            cev,
            question: Question {
                kind: QuestionType::Choice,
                instructions: settings.instructions,
                criteria: Some(Value::Object(criteria)),
                task: Some(settings.task),
            },
            models: settings.models,
            timeout: Duration::from_millis(settings.timeout_ms),
            max_chars: settings.max_chars,
        })
    }

    /// What cev is shown of `req`: the system prompt and the latest user
    /// message, each cut to `max_chars`.
    fn state(&self, req: &ChatRequest) -> Value {
        let message = latest_user_text(req);

        let mut state = Map::new();
        if let Some(system) = &req.system {
            state.insert("system".to_string(), clip(system, self.max_chars).into());
        }
        state.insert("request".to_string(), clip(&message, self.max_chars).into());
        Value::Object(state)
    }

    /// Tells cev which model should have answered the request behind
    /// `decision_id` — `label` is that model's option name, which may well
    /// be the one that was picked. The label is stored with the decision
    /// and, unless online learning is off, updates the adapter right away.
    pub async fn feedback(
        &self,
        decision_id: &str,
        label: &str,
        comment: Option<String>,
    ) -> cev::Result<FeedbackResponse> {
        let feedback = self.cev.feedback(decision_id, Label::Text(label.to_string()), comment);
        tokio::time::timeout(self.timeout, feedback)
            .await
            .map_err(|_| cev::Error::Transport(format!("no answer from cev within {:?}", self.timeout)))?
    }
}

#[cfg(feature = "cev-local")]
fn local(model: &str, db: &str) -> anyhow::Result<Cev> {
    use cev_model::{Engine, EngineOptions};
    use cev_runtime::{RuntimeConfig, Store};

    let engine = Engine::load(EngineOptions {
        model: model.to_string(),
        ..Default::default()
    })?;
    let store = Store::open(db)?;
    let runtime = cev_runtime::Cev::new(std::sync::Arc::new(engine), store, RuntimeConfig::default())?;
    Ok(Cev::local(runtime))
}

#[cfg(not(feature = "cev-local"))]
fn local(_model: &str, _db: &str) -> anyhow::Result<Cev> {
    bail!(
        "`model` runs cev in-process, which needs a build with `--features cev-local` \
         (or set `url` to use a cev server)"
    )
}

#[async_trait]
impl Classifier for CevClassifier {
    fn id(&self) -> &'static str {
        "cev"
    }

    async fn classify(&self, _ctx: &ClassifierContext, req: &ChatRequest) -> anyhow::Result<Classification> {
        let request = SystemOneRequest {
            state: self.state(req),
            model: None,
            questions: IndexMap::from([(QUESTION.to_string(), self.question.clone())]),
            no_store: false,
            debias: None,
        };

        let response = tokio::time::timeout(self.timeout, self.cev.system_one(request))
            .await
            .with_context(|| format!("no answer from cev within {:?}", self.timeout))??;

        let Some(Answer::Choice {
            choice,
            confidence,
            probabilities,
            x_cev,
        }) = response.answers.get(QUESTION)
        else {
            bail!("cev didn't answer with a choice");
        };
        let picked = self
            .models
            .iter()
            .find(|m| m.name() == choice)
            .with_context(|| format!("cev picked '{choice}', which isn't one of `models`"))?;
        let meta = x_cev.as_ref().context("cev's answer has no decision id")?;

        Ok(Classification {
            tags: Vec::new(),
            route: Some(RouteChoice {
                classifier: "cev".to_string(),
                decision_id: Some(meta.decision_id.clone()),
                choice: choice.clone(),
                provider: picked.provider.clone(),
                model: picked.model.clone(),
                confidence: *confidence,
                probabilities: probabilities.clone(),
                adapted: meta.adapted,
            }),
        })
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use std::sync::Arc;

    use cev_core::MockBackend;
    use cev_runtime::{RuntimeConfig, Store};
    use serde_json::json;

    use super::*;
    use crate::canonical::{Message, Role};

    fn settings(value: Value) -> Map<String, Value> {
        value.as_object().unwrap().clone()
    }

    fn providers() -> Vec<ProviderConfig> {
        toml::from_str::<crate::config::Config>(
            r#"
            [[providers]]
            name = "ollama"
            format = "ollama"
            base_url = "http://localhost:11434"
            "#,
        )
        .unwrap()
        .providers
    }

    /// A classifier choosing between a coding model and a general one,
    /// backed by cev's keyword-matching mock model rather than a server.
    pub(crate) fn mock_classifier() -> CevClassifier {
        let settings: Settings = serde_json::from_value(json!({"models": [
            {
                "provider": "ollama",
                "model": "deepseek-r1:latest",
                "description": "Programming, debugging and algorithms",
            },
            {
                "provider": "ollama",
                "model": "llama3.1:8b",
                "name": "general",
                "description": "Recipes, travel and conversation",
            },
        ]}))
        .unwrap();
        let store = Store::in_memory().unwrap();
        let runtime = cev_runtime::Cev::new(Arc::new(MockBackend::default()), store, RuntimeConfig::default()).unwrap();
        CevClassifier::new(Cev::local(runtime), settings).unwrap()
    }

    fn request(content: &str) -> ChatRequest {
        ChatRequest {
            model: "gpt-4".to_string(),
            system: None,
            messages: vec![Message::text(Role::User, content)],
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
            route: None,
        }
    }

    async fn route(classifier: &CevClassifier, content: &str) -> RouteChoice {
        let ctx = ClassifierContext {
            settings: Default::default(),
        };
        classifier.classify(&ctx, &request(content)).await.unwrap().route.unwrap()
    }

    #[tokio::test]
    async fn picks_the_model_whose_description_fits() {
        let classifier = mock_classifier();

        let picked = route(&classifier, "I need help debugging these algorithms").await;
        assert_eq!(picked.choice, "deepseek-r1:latest");
        assert_eq!((picked.provider.as_str(), picked.model.as_str()), ("ollama", "deepseek-r1:latest"));
        assert!(picked.decision_id.as_deref().is_some_and(|id| !id.is_empty()));
        assert_eq!(picked.probabilities.keys().collect::<Vec<_>>(), ["deepseek-r1:latest", "general"]);

        // An option is picked and labeled by its `name`, and routes to its `model`.
        let picked = route(&classifier, "Any recipes for my travel?").await;
        assert_eq!(picked.choice, "general");
        assert_eq!(picked.model, "llama3.1:8b");
    }

    #[tokio::test]
    async fn feedback_teaches_it_a_different_pick() {
        let classifier = mock_classifier();
        let ask = |i: usize| format!("Plan my travel to a programming conference #{i}");

        // The mock model can't tell: the prompt has one keyword of each option.
        let before = route(&classifier, &ask(0)).await;
        assert!(before.confidence < 0.1, "confidence {}", before.confidence);
        assert!(!before.adapted);

        // cev trusts an adapter once it has 8 labels and has beaten the
        // base model on them.
        let mut learned = None;
        for i in 0..8 {
            let decision = route(&classifier, &ask(i)).await;
            learned = Some(classifier.feedback(decision.decision_id.as_deref().unwrap(), "general", None).await.unwrap());
        }
        let learned = learned.unwrap();
        assert!(learned.learned);
        assert_eq!(learned.task, "llmrouter.route");
        assert_eq!(learned.task_examples, 8);
        assert!(learned.adapter_active);

        let after = route(&classifier, &ask(8)).await;
        assert_eq!(after.choice, "general");
        assert!(after.adapted);
        assert!(after.confidence > 0.5, "confidence {}", after.confidence);
    }

    #[tokio::test]
    async fn feedback_rejects_a_label_that_is_not_an_option() {
        let classifier = mock_classifier();
        let decision = route(&classifier, "hello").await;

        let err = classifier.feedback(decision.decision_id.as_deref().unwrap(), "gpt-4", None).await.unwrap_err();
        assert!(matches!(err, cev::Error::Api { status: 400, .. }), "{err}");

        let err = classifier.feedback("dec_missing", "general", None).await.unwrap_err();
        assert!(matches!(err, cev::Error::Api { status: 404, .. }), "{err}");
    }

    #[test]
    fn state_is_the_system_prompt_and_latest_user_message_clipped() {
        let mut classifier = mock_classifier();
        classifier.max_chars = 5;

        let mut req = request("first question");
        req.system = Some("You are héllo".to_string());
        req.messages.push(Message::text(Role::Assistant, "an answer"));
        req.messages.push(Message::text(Role::User, "naïve follow-up"));

        assert_eq!(classifier.state(&req), json!({"system": "You a", "request": "naïve"}));
    }

    #[test]
    fn rejects_bad_settings() {
        let a = json!({"provider": "ollama", "model": "a"});
        let b = json!({"provider": "ollama", "model": "b"});
        let stray = json!({"provider": "nope", "model": "c"});
        let build = |value: Value| CevClassifier::from_settings(&settings(value), &providers());
        let error = |value: Value| build(value).err().expect("settings should be rejected").to_string();

        assert!(build(json!({"url": "http://x", "models": [a, b]})).is_ok());

        assert!(error(json!({"models": [a, b]})).contains("either `url`"));
        assert!(error(json!({"url": "http://x", "model": "Qwen/Qwen3-0.6B", "models": [a, b]})).contains("either `url`"));
        assert!(error(json!({"url": "http://x", "models": [a]})).contains("at least two"));
        assert!(error(json!({"url": "http://x", "models": [a, stray]})).contains("unknown provider 'nope'"));
        assert!(error(json!({"url": "http://x", "models": [a, a]})).contains("two models are named 'a'"));
        assert!(error(json!({"url": "http://x", "timeout": 5, "models": [a, b]})).contains("unknown field `timeout`"));
    }
}
