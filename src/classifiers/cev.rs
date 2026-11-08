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
//! If cev's model takes images or audio (d1, d3 and Qwen3.5 take images,
//! d1-omni takes either), it also gets those of the latest user turn: up to
//! `max_images` PNG, JPEG, WebP or GIF images, and a WAV clip. What the
//! model takes is asked of cev once, the first time a request has any.
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
//! | `model` | | Model to run in-process instead of `url`: a short name such as `d1` or `kev:4b`, a Hugging Face repo id, or a directory. |
//! | `db` | `cev.db` | With `model`: SQLite file for decisions, feedback and adapters. |
//! | `models` | | The options to choose between; see below. |
//! | `instructions` | "Which model is best suited to answer this request?" | The question. |
//! | `task` | `llmrouter.route` | Key of the online adapter that learns from feedback. |
//! | `timeout_ms` | `5000` | How long to wait for cev before routing without it. |
//! | `max_chars` | `2000` | How much of the system prompt and of the message to send. |
//! | `max_images` | `4` | Most images to send, for a model that takes them. `0` sends none. |
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
use serde::de::DeserializeOwned;
use serde_json::{Map, Value, json};
use tokio::sync::OnceCell;

use super::{Classification, Classifier, ClassifierContext, ModelOption, check_models, clip, latest_user_text};
use crate::canonical::{ChatRequest, ContentPart, Message, Role, RouteChoice};
use crate::config::ProviderConfig;

/// Id of the one question in each request to cev.
const QUESTION: &str = "route";

/// Image types cev decodes.
const IMAGE_TYPES: [&str; 5] = ["image/png", "image/jpeg", "image/jpg", "image/webp", "image/gif"];

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
    max_images: usize,
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
            max_images: 4,
            llm_label_weight: 1.0,
        }
    }
}

pub struct CevClassifier {
    cev: Cev,
    direct: Direct,
    /// What cev's model takes besides text, once asked.
    takes: OnceCell<Media>,
    /// The question asked about every request.
    question: Question,
    models: Vec<ModelOption>,
    timeout: Duration,
    max_chars: usize,
    max_images: usize,
    /// Weight of a label taken from an `llm` pick; `None` takes none.
    llm_label_weight: Option<f64>,
}

/// What a cev model takes besides text.
#[derive(Debug, Clone, Copy, PartialEq)]
struct Media {
    images: bool,
    audio: bool,
    /// Takes images or audio in one request, not both (d1-omni).
    one_kind: bool,
}

impl Media {
    /// From a model's `modalities` and `family`, as cev's `/v1/models`
    /// reports them.
    fn of(modalities: &[String], family: &str) -> Self {
        let takes = |modality: &str| modalities.iter().any(|m| m == modality);
        Media {
            images: takes("image"),
            audio: takes("audio"),
            one_kind: family == "d1-omni",
        }
    }
}

/// cev's `GET /v1/models`. Servers before cev 0.2 report neither field, and
/// take text only.
#[derive(Deserialize)]
struct ModelList {
    data: Vec<ModelEntry>,
}

#[derive(Deserialize)]
struct ModelEntry {
    #[serde(default)]
    modalities: Vec<String>,
    #[serde(default)]
    family: String,
}

/// What the classifier asks of cev that [`Cev`] can't: labels with a
/// `weight` and `metadata` ([`Cev::feedback`] sends neither), and what the
/// model takes.
enum Direct {
    Http {
        url: String,
        key: Option<String>,
        client: reqwest::Client,
    },
    #[cfg(any(test, feature = "cev-local"))]
    Local(std::sync::Arc<cev_runtime::Cev>),
}

impl Direct {
    async fn media(&self) -> cev::Result<Media> {
        match self {
            Direct::Http { url, key, client } => {
                let list: ModelList = call(client.get(endpoint(url, "models")), key).await?;
                let model = list.data.first().ok_or_else(|| cev::Error::Protocol("cev lists no model".to_string()))?;
                Ok(Media::of(&model.modalities, &model.family))
            }
            #[cfg(any(test, feature = "cev-local"))]
            Direct::Local(runtime) => {
                let info = runtime.model_info();
                Ok(Media::of(&info.modalities, &info.family))
            }
        }
    }

    async fn feedback(&self, request: FeedbackRequest) -> cev::Result<FeedbackResponse> {
        match self {
            Direct::Http { url, key, client } => call(client.post(endpoint(url, "feedback")).json(&request), key).await,
            #[cfg(any(test, feature = "cev-local"))]
            Direct::Local(runtime) => {
                let transport = |err: &dyn std::fmt::Display| cev::Error::Transport(err.to_string());
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

fn endpoint(url: &str, path: &str) -> String {
    format!("{}/v1/{path}", url.trim_end_matches('/'))
}

/// Sends `request` to a cev server, with `key` as its bearer token, and
/// reads the answer, failing the way [`Cev`] does.
async fn call<T: DeserializeOwned>(mut request: reqwest::RequestBuilder, key: &Option<String>) -> cev::Result<T> {
    let transport = |err: reqwest::Error| cev::Error::Transport(err.to_string());
    if let Some(key) = key {
        request = request.bearer_auth(key);
    }
    let response = request.send().await.map_err(transport)?;
    let status = response.status();
    let body = response.text().await.map_err(transport)?;
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

/// The latest user message with text, an image or audio in it: the turn
/// whose images and audio cev is shown. Turns of tool results alone are
/// skipped, as [`latest_user_text`] skips them.
fn latest_turn(req: &ChatRequest) -> Option<&Message> {
    req.messages.iter().rev().filter(|m| m.role == Role::User).find(|m| {
        m.content.iter().any(|part| match part {
            ContentPart::Text { text } => !text.is_empty(),
            ContentPart::Image { .. } | ContentPart::Audio { .. } => true,
            ContentPart::ToolUse { .. } | ContentPart::ToolResult { .. } => false,
        })
    })
}

impl CevClassifier {
    /// Builds the classifier from its `[classifiers.cev]` settings.
    /// `providers` are the configured providers, which every model option
    /// has to name one of.
    pub fn from_settings(settings: &Map<String, Value>, providers: &[ProviderConfig]) -> anyhow::Result<Self> {
        let settings: Settings = serde_json::from_value(Value::Object(settings.clone()))?;

        let (cev, direct) = match (&settings.url, &settings.model) {
            (Some(url), None) => {
                let key = match &settings.api_key_env {
                    Some(var) => Some(std::env::var(var).with_context(|| format!("${var} is not set"))?),
                    None => None,
                };
                let cev = match &key {
                    Some(key) => Cev::http(url.as_str()).with_api_key(key.as_str()),
                    None => Cev::http(url.as_str()),
                };
                let direct = Direct::Http {
                    url: url.clone(),
                    key,
                    client: reqwest::Client::new(),
                };
                (cev, direct)
            }
            (None, Some(model)) => local(model, &settings.db)?,
            _ => bail!("set either `url` (a running cev server) or `model` (to run one in-process)"),
        };

        check_models(&settings.models, providers)?;
        Self::new(cev, direct, settings)
    }

    fn new(cev: Cev, direct: Direct, settings: Settings) -> anyhow::Result<Self> {
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
            direct,
            takes: OnceCell::new(),
            question: Question {
                kind: QuestionType::Choice,
                instructions: settings.instructions,
                criteria: Some(Value::Object(criteria)),
                task: Some(settings.task),
            },
            models: settings.models,
            timeout: Duration::from_millis(settings.timeout_ms),
            max_chars: settings.max_chars,
            max_images: settings.max_images,
            llm_label_weight: Some(settings.llm_label_weight).filter(|weight| *weight > 0.0),
        })
    }

    /// The images and audio of the latest user turn that cev's model takes:
    /// up to `max_images` images it can decode, and a WAV clip. A model
    /// that takes one kind at a time gets the clip.
    async fn media(&self, req: &ChatRequest) -> (Vec<String>, Option<String>) {
        let parts = latest_turn(req).map_or(&[][..], |turn| &turn.content[..]);
        let mut images: Vec<&String> = parts
            .iter()
            .filter_map(|part| match part {
                ContentPart::Image { media_type, data }
                    if IMAGE_TYPES.iter().any(|t| t.eq_ignore_ascii_case(media_type)) =>
                {
                    Some(data)
                }
                _ => None,
            })
            .take(self.max_images)
            .collect();
        let mut audio = parts.iter().find_map(|part| match part {
            ContentPart::Audio { format, data } if format.eq_ignore_ascii_case("wav") => Some(data),
            _ => None,
        });
        if images.is_empty() && audio.is_none() {
            return (Vec::new(), None);
        }

        let takes = match self.takes.get_or_try_init(|| self.direct.media()).await {
            Ok(takes) => *takes,
            Err(err) => {
                tracing::warn!("sending cev text only: can't tell whether its model takes images or audio: {err}");
                return (Vec::new(), None);
            }
        };
        if !takes.audio {
            audio = None;
        }
        if !takes.images || (takes.one_kind && audio.is_some()) {
            images.clear();
        }
        (images.into_iter().cloned().collect(), audio.cloned())
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
        tokio::time::timeout(self.timeout, self.direct.feedback(request))
            .await
            .map_err(|_| cev::Error::Transport(format!("no answer from cev within {:?}", self.timeout)))?
    }
}

#[cfg(feature = "cev-local")]
fn local(model: &str, db: &str) -> anyhow::Result<(Cev, Direct)> {
    use std::sync::Arc;

    use cev_model::EngineOptions;
    use cev_runtime::{RuntimeConfig, Store};

    // Any model cev runs; which kind it is is told from its files.
    let backend = cev_model::load(EngineOptions {
        model: model.to_string(),
        ..Default::default()
    })?;
    let store = Store::open(db)?;
    let runtime = Arc::new(cev_runtime::Cev::new(backend, store, RuntimeConfig::default())?);

    // A model's first decision can be slow: d1-omni's first on a Mac that
    // hadn't run it took 22 s, past any sensible `timeout_ms`. Make it now
    // rather than on a request; once warm it takes milliseconds.
    let started = std::time::Instant::now();
    runtime.decide(&serde_json::from_value(json!({
        "state": "warm-up",
        "questions": {"q": {"type": "choice", "instructions": "Which?", "criteria": {"a": null, "b": null}}},
        "no_store": true,
    }))?)?;

    let info = runtime.model_info();
    tracing::info!(
        model = %info.id,
        family = %info.family,
        modalities = ?info.modalities,
        warm_up_ms = started.elapsed().as_millis() as u64,
        "cev model loaded"
    );
    Ok((Cev::from_runtime(runtime.clone()), Direct::Local(runtime)))
}

#[cfg(not(feature = "cev-local"))]
fn local(_model: &str, _db: &str) -> anyhow::Result<(Cev, Direct)> {
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
        let decide = async {
            let (images, audio) = self.media(req).await;
            let request = SystemOneRequest {
                state: self.state(req),
                model: None,
                questions: IndexMap::from([(QUESTION.to_string(), self.question.clone())]),
                no_store: false,
                debias: None,
                namespace: None,
                images,
                audio,
            };
            self.cev.system_one(request).await
        };

        let response = tokio::time::timeout(self.timeout, decide)
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
        mock_with(MockBackend::default())
    }

    fn mock_with(backend: MockBackend) -> (CevClassifier, Arc<cev_runtime::Cev>) {
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
        let runtime = cev_runtime::Cev::new(Arc::new(backend), store, RuntimeConfig::default()).unwrap();
        let runtime = Arc::new(runtime);
        let classifier = CevClassifier::new(Cev::from_runtime(runtime.clone()), Direct::Local(runtime.clone()), settings);
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
        route_request(classifier, &request(content)).await
    }

    async fn route_request(classifier: &CevClassifier, req: &ChatRequest) -> RouteChoice {
        let ctx = ClassifierContext {
            settings: Default::default(),
        };
        classifier.classify(&ctx, req).await.unwrap().route.unwrap()
    }

    fn image(media_type: &str, data: &str) -> ContentPart {
        ContentPart::Image {
            media_type: media_type.to_string(),
            data: data.to_string(),
        }
    }

    fn audio(format: &str, data: &str) -> ContentPart {
        ContentPart::Audio {
            format: format.to_string(),
            data: data.to_string(),
        }
    }

    /// A request whose latest user turn has two PNGs, a HEIC cev can't
    /// decode, and a WAV clip.
    fn media_request() -> ChatRequest {
        let mut req = request("Is this code or a recipe?");
        req.messages[0].content.extend([
            image("image/png", "iVBORw0KGgo="),
            image("image/heic", "AAAAGGZ0eXA="),
            image("image/PNG", "iVBORw0KGgoA"),
            audio("mp3", "SUQz"),
            audio("wav", "UklGRg=="),
        ]);
        req
    }

    #[tokio::test]
    async fn sends_media_only_to_a_model_that_takes_it() {
        let both = (vec!["iVBORw0KGgo=".to_string(), "iVBORw0KGgoA".to_string()], Some("UklGRg==".to_string()));

        // cev would refuse the request if they were sent.
        let (text_only, _) = mock_with(MockBackend::default());
        assert_eq!(text_only.media(&media_request()).await, (Vec::new(), None));
        route_request(&text_only, &media_request()).await;

        let (takes_media, _) = mock_with(MockBackend::default().with_media());
        assert_eq!(takes_media.media(&media_request()).await, both);
        route_request(&takes_media, &media_request()).await;

        let with = |takes: Media, max_images: usize| {
            let (mut classifier, _) = mock_with(MockBackend::default());
            classifier.takes = OnceCell::new_with(Some(takes));
            classifier.max_images = max_images;
            classifier
        };
        let images = Media {
            images: true,
            audio: false,
            one_kind: false,
        };
        assert_eq!(with(images, 4).media(&media_request()).await, (both.0.clone(), None));
        assert_eq!(with(images, 1).media(&media_request()).await, (vec![both.0[0].clone()], None));
        assert_eq!(with(images, 0).media(&media_request()).await, (Vec::new(), None));

        // d1-omni takes one kind at a time, and gets the clip.
        let d1_omni = Media {
            images: true,
            audio: true,
            one_kind: true,
        };
        assert_eq!(with(d1_omni, 4).media(&media_request()).await, (Vec::new(), both.1.clone()));
        let mut no_clip = media_request();
        no_clip.messages[0].content.pop();
        assert_eq!(with(d1_omni, 4).media(&no_clip).await, (both.0.clone(), None));
    }

    #[tokio::test]
    async fn sends_the_media_of_the_latest_user_turn() {
        let (classifier, _) = mock_with(MockBackend::default().with_media());
        let png = || image("image/png", "iVBORw0KGgo=");

        // A tool result isn't a turn of its own; a later question is.
        let mut req = request("What is in this picture?");
        req.messages[0].content.push(png());
        req.messages.push(Message {
            role: Role::User,
            content: vec![ContentPart::ToolResult {
                tool_use_id: "call_1".to_string(),
                content: "ok".to_string(),
            }],
        });
        assert_eq!(classifier.media(&req).await.0.len(), 1);
        req.messages.push(Message::text(Role::Assistant, "A cat."));
        req.messages.push(Message::text(Role::User, "What breed?"));
        assert_eq!(classifier.media(&req).await.0.len(), 0);

        // A picture sent on its own is still the latest turn.
        req.messages.push(Message {
            role: Role::User,
            content: vec![png()],
        });
        assert_eq!(classifier.media(&req).await.0.len(), 1);
        assert_eq!(classifier.state(&req)["request"], "What breed?");
    }

    #[tokio::test]
    async fn asks_a_cev_server_once_what_its_model_takes() {
        use std::sync::atomic::{AtomicUsize, Ordering};

        use axum::{Json, routing::get};

        async fn server(model: Value) -> (CevClassifier, Arc<AtomicUsize>) {
            let asked = Arc::new(AtomicUsize::new(0));
            let count = asked.clone();
            let app = axum::Router::new().route(
                "/v1/models",
                get(move || async move {
                    count.fetch_add(1, Ordering::SeqCst);
                    Json(json!({"object": "list", "data": [model]}))
                }),
            );
            let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
            let url = format!("http://{}", listener.local_addr().unwrap());
            tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });

            let (mock, _) = mock_classifier_and_runtime();
            let direct = Direct::Http {
                url: url.clone(),
                key: None,
                client: reqwest::Client::new(),
            };
            let settings = Settings {
                models: mock.models,
                ..Default::default()
            };
            (CevClassifier::new(Cev::http(url), direct, settings).unwrap(), asked)
        }

        let d1_omni = json!({"id": "cev-d1-omni-600m", "family": "d1-omni", "modalities": ["text", "image", "audio"]});
        let (classifier, asked) = server(d1_omni).await;
        assert_eq!(classifier.media(&request("no media here")).await, (Vec::new(), None));
        assert_eq!(asked.load(Ordering::SeqCst), 0);
        for _ in 0..2 {
            assert_eq!(classifier.media(&media_request()).await, (Vec::new(), Some("UklGRg==".to_string())));
        }
        assert_eq!(asked.load(Ordering::SeqCst), 1);

        // A server from before cev 0.2 says neither, and takes text only.
        let (classifier, _) = server(json!({"id": "cev-qwen3-1.7b", "object": "model"})).await;
        assert_eq!(classifier.media(&media_request()).await, (Vec::new(), None));
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

        let direct = Direct::Http {
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

        let learned = direct.feedback(label("dec_1")).await.unwrap();
        assert_eq!(learned.task_examples, 3);
        let (auth, body) = received.lock().unwrap()[0].clone();
        assert_eq!(auth, "Bearer secret");
        assert_eq!(body, json!({"decision_id": "dec_1", "label": "general", "weight": 0.5, "metadata": {"source": "llm"}}));

        // The server's errors come back as cev's would.
        let err = direct.feedback(label("dec_missing")).await.unwrap_err();
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
