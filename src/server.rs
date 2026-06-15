//! HTTP surface: an OpenAI-compatible `/v1/chat/completions` endpoint and an
//! Anthropic-compatible `/v1/messages` endpoint, both backed by the same
//! [`ModelRouter`].

use std::sync::Arc;
use std::time::Instant;

use axum::extract::State;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router as AxumRouter};
use serde_json::{json, Map, Value};

use crate::canonical::{ChatRequest, ChatResponse};
use crate::classifiers::ClassifierRegistry;
use crate::formats::{anthropic, openai};
use crate::logging::{LogEntry, RequestLogger};
use crate::plugins::{Flow, Plugin, PluginContext, PluginRegistry, Stage};
use crate::router::ModelRouter;

#[derive(Clone)]
pub struct AppState {
    pub router: Arc<ModelRouter>,
    pub client: reqwest::Client,
    pub logger: Option<Arc<RequestLogger>>,
    pub plugins: Arc<PluginRegistry>,
    pub classifiers: Arc<ClassifierRegistry>,
}

pub fn build_app(state: AppState) -> AxumRouter {
    AxumRouter::new()
        .route("/health", get(health))
        .route("/v1/chat/completions", post(chat_completions))
        .route("/v1/messages", post(messages))
        .with_state(state)
}

async fn health() -> &'static str {
    "ok"
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
/// returns [`Flow::Stop`].
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
            Ok(Flow::Stop) => return Ok(Flow::Stop),
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

    let requested_model = req.model.clone();
    let mut sent_model = req.model.clone();
    let mut provider_name = "plugin".to_string();

    let started = Instant::now();

    if run_stage(&resolved_plugins, &state.client, Stage::Start, &mut req, &mut resp).await? == Flow::Continue {
        req.tags = state.classifiers.classify(&req).await;

        let routing_flow =
            run_stage(&resolved_plugins, &state.client, Stage::PreRouting, &mut req, &mut resp).await?;

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
                    if let Some(logger) = &state.logger {
                        logger.log(&LogEntry {
                            ts_ms: LogEntry::now_ms(),
                            provider: provider_name,
                            requested_model,
                            sent_model,
                            duration_ms: started.elapsed().as_millis(),
                            tags: req.tags,
                            plugins: resolved_plugins.iter().map(|(p, _)| p.id()).collect(),
                            system: req.system,
                            messages: req.messages,
                            response: None,
                            error: Some(err.to_string()),
                        });
                    }
                    return Err(ApiError::Upstream(err));
                }
            }
        }
    }

    if run_stage(&resolved_plugins, &state.client, Stage::PostResponse, &mut req, &mut resp).await?
        == Flow::Continue
    {
        run_stage(&resolved_plugins, &state.client, Stage::End, &mut req, &mut resp).await?;
    }

    if let Some(logger) = &state.logger {
        logger.log(&LogEntry {
            ts_ms: LogEntry::now_ms(),
            provider: provider_name,
            requested_model,
            sent_model,
            duration_ms: started.elapsed().as_millis(),
            tags: req.tags,
            plugins: resolved_plugins.iter().map(|(p, _)| p.id()).collect(),
            system: req.system,
            messages: req.messages,
            response: resp.clone(),
            error: None,
        });
    }

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
