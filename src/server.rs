//! HTTP surface: an OpenAI-compatible `/v1/chat/completions` endpoint and an
//! Anthropic-compatible `/v1/messages` endpoint, both backed by the same
//! [`ModelRouter`].

use std::convert::Infallible;
use std::sync::Arc;
use std::time::{Duration, Instant};

use axum::extract::State;
use axum::http::StatusCode;
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{Html, IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router as AxumRouter};
use futures_core::Stream;
use serde_json::{json, Map, Value};
use tokio::sync::broadcast;
use tokio_stream::StreamExt;
use tokio_stream::wrappers::BroadcastStream;

use crate::canonical::{ChatRequest, ChatResponse};
use crate::classifiers::ClassifierRegistry;
use crate::formats::{anthropic, openai};
use crate::logging::{LogEntry, RequestLogger};
use crate::plugins::{Flow, Plugin, PluginContext, PluginRegistry, Stage};
use crate::router::ModelRouter;

/// Embedded dashboard page, served at `/dashboard` when enabled. Connects
/// to `/dashboard/events` via SSE to show requests as they're handled.
const DASHBOARD_HTML: &str = include_str!("../static/dashboard.html");

#[derive(Clone)]
pub struct AppState {
    pub router: Arc<ModelRouter>,
    pub client: reqwest::Client,
    pub logger: Option<Arc<RequestLogger>>,
    pub plugins: Arc<PluginRegistry>,
    pub classifiers: Arc<ClassifierRegistry>,
    /// Broadcasts a JSON [`LogEntry`] line for every handled request, for
    /// the `/dashboard/events` SSE feed. Sending is a no-op if there are no
    /// subscribers.
    pub events: broadcast::Sender<Arc<str>>,
}

pub fn build_app(state: AppState, dashboard: bool) -> AxumRouter {
    let mut router = AxumRouter::new()
        .route("/health", get(health))
        .route("/v1/chat/completions", post(chat_completions))
        .route("/v1/messages", post(messages));

    if dashboard {
        router = router
            .route("/dashboard", get(dashboard_page))
            .route("/dashboard/events", get(dashboard_events));
    }

    router.with_state(state)
}

async fn health() -> &'static str {
    "ok"
}

async fn dashboard_page() -> Html<&'static str> {
    Html(DASHBOARD_HTML)
}

/// Streams a JSON [`LogEntry`] line as an SSE `data:` event for every
/// request handled from this point on.
async fn dashboard_events(
    State(state): State<AppState>,
) -> Sse<impl Stream<Item = Result<Event, Infallible>>> {
    let stream = BroadcastStream::new(state.events.subscribe())
        .filter_map(|msg| msg.ok().map(|line| Ok(Event::default().data(line.to_string()))));

    Sse::new(stream).keep_alive(KeepAlive::new().interval(Duration::from_secs(15)))
}

/// Error type for the request handlers, rendered as a JSON error body.
enum ApiError {
    /// No route or default provider matches the requested model.
    NoProvider(String),
    /// The chosen provider returned an error or an unparsable response.
    Upstream(anyhow::Error),
    /// A plugin's `on_start`/`pre_request` hook failed.
    Plugin(&'static str, anyhow::Error),
    /// The pipeline finished with no response: a plugin returned
    /// [`Flow::Stop`] before routing without writing one into `resp`.
    NoResponse,
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let (status, message) = match self {
            ApiError::NoProvider(model) => (
                StatusCode::BAD_REQUEST,
                format!("no provider configured for model '{model}'"),
            ),
            ApiError::Upstream(err) => (StatusCode::BAD_GATEWAY, err.to_string()),
            ApiError::Plugin(id, err) => (
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("plugin '{id}' failed: {err}"),
            ),
            ApiError::NoResponse => (
                StatusCode::INTERNAL_SERVER_ERROR,
                "pipeline stopped without producing a response".to_string(),
            ),
        };

        (
            status,
            Json(json!({
                "error": {
                    "type": "router_error",
                    "message": message,
                }
            })),
        )
            .into_response()
    }
}

type ResolvedPlugins = Vec<(Arc<dyn Plugin>, Map<String, Value>)>;

/// Runs every resolved plugin's hook for `stage`, in order, until one
/// returns [`Flow::Stop`]. Pushes the id of any plugin that returns
/// [`Flow::Modified`] or [`Flow::Stop`] into `active` — those are the
/// only ones worth surfacing in logs/UI.
///
/// Errors from [`Stage::Start`]/[`Stage::PreRouting`] hooks abort the
/// request (`ApiError::Plugin`). Errors from [`Stage::PostResponse`]/
/// [`Stage::End`] hooks are logged and treated as [`Flow::Continue`], since
/// a response is already available by then.
async fn run_stage(
    plugins: &ResolvedPlugins,
    client: &reqwest::Client,
    stage: Stage,
    req: &mut ChatRequest,
    resp: &mut Option<ChatResponse>,
    active: &mut Vec<String>,
) -> Result<Flow, ApiError> {
    for (plugin, settings) in plugins {
        let ctx = PluginContext {
            client: client.clone(),
            settings: settings.clone(),
        };

        let result = match stage {
            Stage::Start => plugin.on_start(&ctx, req, resp).await,
            Stage::PreRouting => plugin.pre_request(&ctx, req, resp).await,
            Stage::PostResponse => plugin.post_response(&ctx, req, resp).await,
            Stage::End => plugin.on_end(&ctx, req, resp).await,
        };

        match result {
            Ok(Flow::Continue) => continue,
            Ok(Flow::Modified) => {
                active.push(plugin.id().to_string());
            }
            Ok(Flow::Stop) => {
                active.push(plugin.id().to_string());
                return Ok(Flow::Stop);
            }
            Err(err) => match stage {
                Stage::Start | Stage::PreRouting => {
                    return Err(ApiError::Plugin(plugin.id(), err));
                }
                Stage::PostResponse | Stage::End => {
                    tracing::warn!("plugin '{}' {stage:?} hook failed: {err}", plugin.id());
                }
            },
        }
    }

    Ok(Flow::Continue)
}

/// Serializes `entry` once and both appends it to the request log file (if
/// enabled) and broadcasts it to any `/dashboard/events` subscribers.
fn record(state: &AppState, entry: &LogEntry) {
    let line = match serde_json::to_string(entry) {
        Ok(line) => line,
        Err(err) => {
            tracing::warn!("failed to serialize log entry: {err}");
            return;
        }
    };

    if let Some(logger) = &state.logger {
        logger.log_line(&line);
    }

    // Err just means no dashboard is currently listening.
    let _ = state.events.send(Arc::from(line));
}

/// Runs the full pipeline for one request:
///
/// ```text
/// Start -> classifiers -> PreRouting -> routers -> provider -> PostResponse -> End -> logging
/// ```
///
/// A plugin can stop the pipeline early at `Start` or `PreRouting` (skipping
/// routing and the provider call) by writing a response into `resp` and
/// returning [`Flow::Stop`]; `PostResponse`/`End` always run afterwards.
async fn dispatch(state: &AppState, mut req: ChatRequest) -> Result<ChatResponse, ApiError> {
    let resolved_plugins = state.plugins.resolve(&req);
    let mut resp: Option<ChatResponse> = None;
    let mut active_plugins: Vec<String> = Vec::new();

    let requested_model = req.model.clone();
    let mut sent_model = req.model.clone();
    let mut provider_name = "plugin".to_string();

    let started = Instant::now();

    if run_stage(&resolved_plugins, &state.client, Stage::Start, &mut req, &mut resp, &mut active_plugins).await? == Flow::Continue {
        req.tags = state.classifiers.classify(&req).await;

        let routing_flow =
            run_stage(&resolved_plugins, &state.client, Stage::PreRouting, &mut req, &mut resp, &mut active_plugins).await?;

        if routing_flow == Flow::Continue && resp.is_none() {
            let (provider, target_model) = match &req.forced_provider {
                Some(name) => {
                    let provider = state
                        .router
                        .provider(name)
                        .ok_or_else(|| ApiError::NoProvider(req.model.clone()))?;
                    (provider, req.model.clone())
                }
                None => state
                    .router
                    .resolve(&req.model, &req.tags)
                    .ok_or_else(|| ApiError::NoProvider(req.model.clone()))?,
            };

            sent_model = target_model.clone();
            provider_name = provider.name.clone();
            req.model = target_model;

            match provider.send(&state.client, &req).await {
                Ok(r) => resp = Some(r),
                Err(err) => {
                    record(
                        state,
                        &LogEntry {
                            ts_ms: LogEntry::now_ms(),
                            provider: provider_name,
                            requested_model,
                            sent_model,
                            duration_ms: started.elapsed().as_millis(),
                            tags: req.tags,
                            plugins: active_plugins,
                            system: req.system,
                            messages: req.messages,
                            response: None,
                            error: Some(err.to_string()),
                        },
                    );
                    return Err(ApiError::Upstream(err));
                }
            }
        }
    }

    if run_stage(&resolved_plugins, &state.client, Stage::PostResponse, &mut req, &mut resp, &mut active_plugins).await?
        == Flow::Continue
    {
        run_stage(&resolved_plugins, &state.client, Stage::End, &mut req, &mut resp, &mut active_plugins).await?;
    }

    record(
        state,
        &LogEntry {
            ts_ms: LogEntry::now_ms(),
            provider: provider_name,
            requested_model,
            sent_model,
            duration_ms: started.elapsed().as_millis(),
            tags: req.tags,
            plugins: active_plugins,
            system: req.system,
            messages: req.messages,
            response: resp.clone(),
            error: None,
        },
    );

    resp.ok_or(ApiError::NoResponse)
}

async fn chat_completions(
    State(state): State<AppState>,
    Json(body): Json<openai::OpenAiChatRequest>,
) -> Result<Json<openai::OpenAiChatResponse>, ApiError> {
    let resp = dispatch(&state, body.into()).await?;
    Ok(Json(resp.into()))
}

async fn messages(
    State(state): State<AppState>,
    Json(body): Json<anthropic::AnthropicMessagesRequest>,
) -> Result<Json<anthropic::AnthropicMessagesResponse>, ApiError> {
    let resp = dispatch(&state, body.into()).await?;
    Ok(Json(resp.into()))
}
