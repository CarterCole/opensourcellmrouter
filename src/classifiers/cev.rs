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
//! cev decides every request whether or not a rule routes on its pick, so
//! it can also learn while another classifier routes. With the `llm`
//! classifier enabled too, which takes `mode = "teacher"` on it, each of
//! its picks labels cev's decision about the same request (see
//! [`CevClassifier::learn_from_llm`]), and a correction sent to
//! `/v1/feedback` later takes that label's place.
//! Labels are stored with a `metadata.source` of `"feedback"` or `"llm"`,
//! so a training set exported from cev can tell them apart.
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
//! | `llm_label_weight` | `1.0` | Weight of a label taken from an `llm` pick, against 1 for a correction. `0` takes none. |
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
use cev_core::{Answer, FeedbackRequest, Question, QuestionType};
use indexmap::IndexMap;
use serde::Deserialize;
use serde_json::{Map, Value, json};

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
    llm_label_weight: f64,
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
            llm_label_weight: 1.0,
        }
    }
}

pub struct CevClassifier {
    cev: Cev,
    labels: Labels,
    /// The question asked about every request.
    question: Question,
    models: Vec<ModelOption>,
    timeout: Duration,
    max_chars: usize,
    /// Weight of a label taken from an `llm` pick; `None` takes none.
    llm_label_weight: Option<f64>,
}

/// Where labels are sent. [`Cev::feedback`] can't attach a label's
/// `weight` or `metadata`, so labels take the same way it sends them, with
/// those added.
enum Labels {
    Http {
        url: String,
        key: Option<String>,
        client: reqwest::Client,
    },
    #[cfg(any(test, feature = "cev-local"))]
    Local(std::sync::Arc<cev_runtime::Cev>),
}

impl Labels {
    async fn send(&self, request: FeedbackRequest) -> cev::Result<FeedbackResponse> {
        let transport = |err: &dyn std::fmt::Display| cev::Error::Transport(err.to_string());
        match self {
            Labels::Http { url, key, client } => {
                let mut post = client.post(format!("{}/v1/feedback", url.trim_end_matches('/'))).json(&request);
                if let Some(key) = key {
                    post = post.bearer_auth(key);
                }
                let response = post.send().await.map_err(|err| transport(&err))?;
                let status = response.status();
                let body = response.text().await.map_err(|err| transport(&err))?;
                if !status.is_success() {
                    let message = serde_json::from_str::<Value>(&body)
                        .ok()
                        .and_then(|v| v["error"]["message"].as_str().map(str::to_string))
                        .unwrap_or(body);
                    return Err(cev::Error::Api {
                        status: status.as_u16(),
                        message,
                    });
                }
                serde_json::from_str(&body).map_err(|err| cev::Error::Protocol(err.to_string()))
            }
            #[cfg(any(test, feature = "cev-local"))]
            Labels::Local(runtime) => {
                use cev_runtime::CevError;

                let runtime = runtime.clone();
                let result = tokio::task::spawn_blocking(move || runtime.feedback(&request))
                    .await
                    .map_err(|err| transport(&err))?;
                result.map_err(|err| {
                    let status = match err {
                        CevError::BadRequest(_) => 400,
                        CevError::NotFound(_) => 404,
                        CevError::Internal(_) => 500,
                    };
                    cev::Error::Api {
                        status,
                        message: err.to_string(),
                    }
                })
            }
        }
    }
}

impl CevClassifier {
    /// Builds the classifier from its `[classifiers.cev]` settings.
    /// `providers` are the configured providers, which every model option
    /// has to name one of.
    pub fn from_settings(settings: &Map<String, Value>, providers: &[ProviderConfig]) -> anyhow::Result<Self> {
        let settings: Settings = serde_json::from_value(Value::Object(settings.clone()))?;

        let (cev, labels) = match (&settings.url, &settings.model) {
            (Some(url), None) => {
                let key = match &settings.api_key_env {
                    Some(var) => Some(std::env::var(var).with_context(|| format!("${var} is not set"))?),
                    None => None,
                };
                let cev = match &key {
                    Some(key) => Cev::http(url.as_str()).with_api_key(key.as_str()),
                    None => Cev::http(url.as_str()),
                };
                let labels = Labels::Http {
                    url: url.clone(),
                    key,
                    client: reqwest::Client::new(),
                };
                (cev, labels)
            }
            (None, Some(model)) => local(model, &settings.db)?,
            _ => bail!("set either `url` (a running cev server) or `model` (to run one in-process)"),
        };

        check_models(&settings.models, providers)?;
        Self::new(cev, labels, settings)
    }

    fn new(cev: Cev, labels: Labels, settings: Settings) -> anyhow::Result<Self> {
        if settings.models.len() < 2 {
            bail!("needs at least two `models` to choose between");
        }
        if !(settings.llm_label_weight.is_finite() && settings.llm_label_weight >= 0.0) {
            bail!("`llm_label_weight` has to be 0 or more");
        }

        let criteria = settings
            .models
            .iter()
            .map(|m| (m.name().to_string(), m.description.clone().map_or(Value::Null, Value::String)))
            .collect();

        Ok(CevClassifier {
            cev,
            labels,
            question: Question {
                kind: QuestionType::Choice,
                instructions: settings.instructions,
                criteria: Some(Value::Object(criteria)),
                task: Some(settings.task),
            },
            models: settings.models,
            timeout: Duration::from_millis(settings.timeout_ms),
            max_chars: settings.max_chars,
            llm_label_weight: Some(settings.llm_label_weight).filter(|weight| *weight > 0.0),
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
    /// and, unless online learning is off, updates the adapter right away,
    /// replacing any label it learned for the decision before.
    pub async fn feedback(
        &self,
        decision_id: &str,
        label: &str,
        comment: Option<String>,
    ) -> cev::Result<FeedbackResponse> {
        self.label(decision_id, label, None, comment, json!({"source": "feedback"})).await
    }

    /// The name of the option that routes to `provider` and `model`, if one
    /// does.
    pub fn option_for(&self, provider: &str, model: &str) -> Option<&str> {
        self.models
            .iter()
            .find(|m| m.provider == provider && m.model == model)
            .map(ModelOption::name)
    }

    /// Whether [`Self::learn_from_llm`] sends anything.
    pub fn learns_from_llm(&self) -> bool {
        self.llm_label_weight.is_some()
    }

    /// Labels the decision behind `decision_id` with the model the `llm`
    /// classifier picked for the same request: the option that routes where
    /// `pick` does, weighted by `llm_label_weight`. `asked` says which
    /// provider and model the `llm` classifier asked, and is stored with
    /// the label. Returns `None`, sending nothing, if no option routes there
    /// or `llm_label_weight` is 0.
    pub async fn learn_from_llm(
        &self,
        decision_id: &str,
        pick: &RouteChoice,
        asked: (&str, &str),
    ) -> cev::Result<Option<FeedbackResponse>> {
        let (Some(weight), Some(label)) = (self.llm_label_weight, self.option_for(&pick.provider, &pick.model)) else {
            return Ok(None);
        };
        let (provider, model) = asked;
        let metadata = json!({"source": "llm", "provider": provider, "model": model});
        self.label(decision_id, label, Some(weight), None, metadata).await.map(Some)
    }

    async fn label(
        &self,
        decision_id: &str,
        label: &str,
        weight: Option<f64>,
        comment: Option<String>,
        metadata: Value,
    ) -> cev::Result<FeedbackResponse> {
        let request = FeedbackRequest {
            decision_id: Some(decision_id.to_string()),
            request_id: None,
            question_id: None,
            label: Some(Label::Text(label.to_string())),
            weight,
            comment,
            metadata: Some(metadata),
        };
        tokio::time::timeout(self.timeout, self.labels.send(request))
            .await
            .map_err(|_| cev::Error::Transport(format!("no answer from cev within {:?}", self.timeout)))?
    }
}

#[cfg(feature = "cev-local")]
fn local(model: &str, db: &str) -> anyhow::Result<(Cev, Labels)> {
    use std::sync::Arc;

    use cev_model::{Engine, EngineOptions};
    use cev_runtime::{RuntimeConfig, Store};

    let engine = Engine::load(EngineOptions {
        model: model.to_string(),
        ..Default::default()
    })?;
    let store = Store::open(db)?;
    let runtime = Arc::new(cev_runtime::Cev::new(Arc::new(engine), store, RuntimeConfig::default())?);
    Ok((Cev::from_runtime(runtime.clone()), Labels::Local(runtime)))
}

#[cfg(not(feature = "cev-local"))]
fn local(_model: &str, _db: &str) -> anyhow::Result<(Cev, Labels)> {
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
        mock_classifier_and_runtime().0
    }

    /// [`mock_classifier`], and the runtime behind it, to look at what it
    /// stored.
    pub(crate) fn mock_classifier_and_runtime() -> (CevClassifier, Arc<cev_runtime::Cev>) {
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
        let runtime = Arc::new(runtime);
        let classifier = CevClassifier::new(Cev::from_runtime(runtime.clone()), Labels::Local(runtime.clone()), settings);
        (classifier.unwrap(), runtime)
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
            routes: Vec::new(),
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
    async fn learns_from_an_llm_pick_of_the_same_model() {
        let (classifier, runtime) = mock_classifier_and_runtime();
        let decision = route(&classifier, "Plan my travel to a programming conference").await;
        let decision_id = decision.decision_id.as_deref().unwrap();

        // The llm classifier names its options its own way; what counts is
        // where they route.
        let pick = RouteChoice {
            classifier: "llm".to_string(),
            decision_id: None,
            choice: "chat".to_string(),
            provider: "ollama".to_string(),
            model: "llama3.1:8b".to_string(),
            confidence: 1.0,
            probabilities: IndexMap::new(),
            adapted: false,
        };
        let learned = classifier.learn_from_llm(decision_id, &pick, ("anthropic", "claude-haiku-5-5")).await.unwrap();
        assert_eq!(learned.unwrap().task_examples, 1);

        // A correction is stored next to it, so an export has both.
        classifier.feedback(decision_id, "deepseek-r1:latest", None).await.unwrap();
        let (_, labels) = runtime.decision(decision_id).unwrap().unwrap();
        let stored: Vec<_> = labels
            .iter()
            .map(|l| json!({"label": l.label, "weight": l.weight, "metadata": l.metadata}))
            .collect();
        assert_eq!(
            stored,
            [
                json!({
                    "label": "general",
                    "weight": 1.0,
                    "metadata": {"source": "llm", "provider": "anthropic", "model": "claude-haiku-5-5"},
                }),
                json!({"label": "deepseek-r1:latest", "weight": 1.0, "metadata": {"source": "feedback"}}),
            ]
        );

        // A pick of a model cev doesn't choose between labels nothing.
        let elsewhere = RouteChoice {
            model: "llama3.1:70b".to_string(),
            ..pick
        };
        assert!(classifier.learn_from_llm(decision_id, &elsewhere, ("anthropic", "m")).await.unwrap().is_none());
    }

    #[tokio::test]
    async fn labels_reach_a_cev_server_with_their_weight_and_source() {
        use std::sync::Mutex;

        use axum::http::{HeaderMap, StatusCode};
        use axum::{Json, routing::post};

        let received = Arc::new(Mutex::new(Vec::new()));
        let log = received.clone();
        let app = axum::Router::new().route(
            "/v1/feedback",
            post(move |headers: HeaderMap, Json(body): Json<Value>| async move {
                log.lock().unwrap().push((headers["authorization"].clone(), body.clone()));
                if body["decision_id"] == "dec_missing" {
                    return (StatusCode::NOT_FOUND, Json(json!({"error": {"message": "decision not found"}})));
                }
                let learned = json!({
                    "feedback_id": "fb_1",
                    "decision_id": body["decision_id"],
                    "task": "llmrouter.route",
                    "learned": true,
                    "task_examples": 3,
                    "adapter_active": false,
                });
                (StatusCode::OK, Json(learned))
            }),
        );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let url = format!("http://{}/", listener.local_addr().unwrap());
        tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });

        let labels = Labels::Http {
            url,
            key: Some("secret".to_string()),
            client: reqwest::Client::new(),
        };
        let label = |decision_id: &str| FeedbackRequest {
            decision_id: Some(decision_id.to_string()),
            request_id: None,
            question_id: None,
            label: Some(Label::Text("general".to_string())),
            weight: Some(0.5),
            comment: None,
            metadata: Some(json!({"source": "llm"})),
        };

        let learned = labels.send(label("dec_1")).await.unwrap();
        assert_eq!(learned.task_examples, 3);
        let (auth, body) = received.lock().unwrap()[0].clone();
        assert_eq!(auth, "Bearer secret");
        assert_eq!(body, json!({"decision_id": "dec_1", "label": "general", "weight": 0.5, "metadata": {"source": "llm"}}));

        // The server's errors come back as cev's would.
        let err = labels.send(label("dec_missing")).await.unwrap_err();
        assert!(matches!(&err, cev::Error::Api { status: 404, message } if message == "decision not found"), "{err}");
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
        assert!(error(json!({"url": "http://x", "llm_label_weight": -1, "models": [a, b]})).contains("0 or more"));

        let learns = |weight: f64| build(json!({"url": "http://x", "llm_label_weight": weight, "models": [a, b]})).unwrap().learns_from_llm();
        assert!(learns(0.5));
        assert!(!learns(0.0));
    }
}
