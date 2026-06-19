//! HTTP surface: an OpenAI-compatible `/v1/chat/completions` endpoint and an
//! Anthropic-compatible `/v1/messages` endpoint, both backed by the same
//! [`ModelRouter`].

use std::convert::Infallible;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use axum::body::Body;
use axum::extract::State;
use axum::http::{StatusCode, header};
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{Html, IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router as AxumRouter};
use futures_core::Stream;
use serde_json::{json, Map, Value};
use tokio::sync::broadcast;
use tokio_stream::StreamExt;
use tokio_stream::wrappers::BroadcastStream;

use crate::canonical::{ChatRequest, ChatResponse, Usage};
use crate::classifiers::{ClassifierRegistry, ResponseClassifierRegistry};
use crate::formats::{anthropic, openai};
use crate::logging::{LogEntry, RequestLogger, RouterEvent};
use crate::plugins::{Flow, Plugin, PluginContext, PluginRegistry, Stage};
use crate::router::ModelRouter;

/// Embedded dashboard page, served at `/dashboard` when enabled.
const DASHBOARD_HTML: &str = include_str!("../static/dashboard.html");

#[derive(Clone)]
pub struct AppState {
    pub router: Arc<ModelRouter>,
    pub client: reqwest::Client,
    pub logger: Option<Arc<RequestLogger>>,
    pub plugins: Arc<PluginRegistry>,
    pub classifiers: Arc<ClassifierRegistry>,
    /// Tags responses after the provider replies (e.g. `"refusal"`). See
    /// [`crate::classifiers::ResponseClassifier`].
    pub response_classifiers: Arc<ResponseClassifierRegistry>,
    /// Broadcasts serialized [`RouterEvent`] JSON for the SSE feed.
    pub events: broadcast::Sender<Arc<str>>,
    /// Number of requests currently inside `dispatch`.
    pub in_flight: Arc<AtomicU64>,
    /// Monotonically increasing request id, used to correlate Start/Complete.
    pub next_id: Arc<AtomicU64>,
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

/// Streams [`RouterEvent`] JSON as SSE `data:` events for every request.
async fn dashboard_events(
    State(state): State<AppState>,
) -> Sse<impl Stream<Item = Result<Event, Infallible>>> {
    let stream = BroadcastStream::new(state.events.subscribe())
        .filter_map(|msg| msg.ok().map(|line| Ok(Event::default().data(line.to_string()))));

    Sse::new(stream).keep_alive(KeepAlive::new().interval(Duration::from_secs(15)))
}

/// Error type for the request handlers.
enum ApiError {
    NoProvider(String),
    Upstream(anyhow::Error),
    Plugin(&'static str, anyhow::Error),
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

/// Decrements `in_flight` when dropped. Used as a RAII guard in `dispatch`
/// so the counter stays accurate even when the function returns early via `?`.
struct InFlightGuard(Arc<AtomicU64>);
impl Drop for InFlightGuard {
    fn drop(&mut self) {
        self.0.fetch_sub(1, Ordering::Relaxed);
    }
}

type ResolvedPlugins = Vec<(Arc<dyn Plugin>, Map<String, Value>)>;

/// Runs every resolved plugin's hook for `stage`, pushing the id of any that
/// returned [`Flow::Modified`] or [`Flow::Stop`] into `active`.
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

/// Serializes `event`, writes `Complete` variants to the log file, and
/// broadcasts to all `/dashboard/events` subscribers.
fn broadcast(state: &AppState, event: &RouterEvent) {
    let line = match serde_json::to_string(event) {
        Ok(l) => l,
        Err(err) => {
            tracing::warn!("failed to serialize event: {err}");
            return;
        }
    };

    if let Some(logger) = &state.logger {
        if matches!(event, RouterEvent::Complete { .. }) {
            logger.log_line(&line);
        }
    }

    let _ = state.events.send(Arc::from(line));
}

/// Everything [`dispatch`] learns about a request that isn't part of the
/// `ChatResponse` body itself, for callers to surface as response headers.
struct DispatchOutcome {
    response: ChatResponse,
    /// The request's classifier tags, assigned before routing (see
    /// [`crate::classifiers::ClassifierRegistry`]). Kept separate from
    /// `response.tags` (assigned by response classifiers afterward).
    request_tags: Vec<String>,
    /// Name of the provider that produced `response` (or `"plugin"` if a
    /// plugin answered the request directly, bypassing routing).
    provider: String,
    /// Model name actually sent to that provider (after any `rewrite_model`).
    model: String,
}

/// Runs the full pipeline for one request:
///
/// ```text
/// Start → classifiers → PreRouting → routers → provider → PostResponse → End → logging
/// ```
///
/// Emits a [`RouterEvent::Start`] immediately so the UI can show in-flight
/// requests before a response is ready, then a [`RouterEvent::Complete`] once
/// the pipeline finishes (successfully or with an error).
async fn dispatch(state: &AppState, mut req: ChatRequest) -> Result<DispatchOutcome, ApiError> {
    let id = state.next_id.fetch_add(1, Ordering::Relaxed);
    let in_flight = state.in_flight.fetch_add(1, Ordering::Relaxed) + 1;
    // Decrement in_flight when this function returns, even on early error exit.
    let _guard = InFlightGuard(state.in_flight.clone());

    let requested_model = req.model.clone();
    let started = Instant::now();

    broadcast(state, &RouterEvent::Start {
        id,
        ts_ms: LogEntry::now_ms(),
        model: requested_model.clone(),
        in_flight,
    });

    let resolved_plugins = state.plugins.resolve(&req);
    let mut resp: Option<ChatResponse> = None;
    let mut active_plugins: Vec<String> = Vec::new();
    let mut sent_model = req.model.clone();
    let mut provider_name = "plugin".to_string();

    if run_stage(&resolved_plugins, &state.client, Stage::Start, &mut req, &mut resp, &mut active_plugins).await? == Flow::Continue {
        req.tags = state.classifiers.classify(&req).await;
        broadcast(state, &RouterEvent::Classified {
            id,
            ts_ms: LogEntry::now_ms(),
            tags: req.tags.clone(),
        });

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
            broadcast(state, &RouterEvent::Routed {
                id,
                ts_ms: LogEntry::now_ms(),
                provider: provider_name.clone(),
                model: target_model.clone(),
            });
            req.model = target_model;

            match provider.send(&state.client, &req).await {
                Ok(r) => resp = Some(r),
                Err(err) => {
                    broadcast(state, &RouterEvent::Complete {
                        id,
                        entry: LogEntry {
                            ts_ms: LogEntry::now_ms(),
                            provider: provider_name,
                            requested_model,
                            sent_model,
                            duration_ms: started.elapsed().as_millis() as u64,
                            tags: req.tags,
                            plugins: active_plugins,
                            system: req.system,
                            messages: req.messages,
                            response: None,
                            error: Some(err.to_string()),
                        },
                    });
                    return Err(ApiError::Upstream(err));
                }
            }
        }
    }

    if run_stage(&resolved_plugins, &state.client, Stage::PostResponse, &mut req, &mut resp, &mut active_plugins).await?
        == Flow::Continue
    {
        if let Some(r) = &mut resp {
            r.tags = state.response_classifiers.classify(&req, r).await;
        }

        run_stage(&resolved_plugins, &state.client, Stage::End, &mut req, &mut resp, &mut active_plugins).await?;
    }

    let request_tags = req.tags.clone();
    let provider = provider_name.clone();
    let model = sent_model.clone();
    broadcast(state, &RouterEvent::Complete {
        id,
        entry: LogEntry {
            ts_ms: LogEntry::now_ms(),
            provider: provider_name,
            requested_model,
            sent_model,
            duration_ms: started.elapsed().as_millis() as u64,
            tags: req.tags,
            plugins: active_plugins,
            system: req.system,
            messages: req.messages,
            response: resp.clone(),
            error: None,
        },
    });

    resp.ok_or(ApiError::NoResponse).map(|response| DispatchOutcome {
        response,
        request_tags,
        provider,
        model,
    })
}

/// Sets `header_name` to a comma-separated tag list (e.g. `vision,nsfw`),
/// omitted entirely when `tags` is empty. Used to carry classifier metadata
/// to the client without touching the OpenAI/Anthropic response body, so the
/// wire formats stay unmodified — see [`X_ROUTER_REQUEST_TAGS`] and
/// [`X_ROUTER_RESPONSE_TAGS`].
fn apply_tags_header(response: &mut Response, header_name: &'static str, tags: &[String]) {
    if tags.is_empty() {
        return;
    }
    if let Ok(value) = header::HeaderValue::from_str(&tags.join(",")) {
        response.headers_mut().insert(header_name, value);
    }
}

/// Sets `header_name` to `value`. Used for the single-value headers
/// ([`X_ROUTER_PROVIDER`], [`X_ROUTER_MODEL`], [`X_ROUTER_INPUT_TOKENS`],
/// [`X_ROUTER_OUTPUT_TOKENS`]), which — unlike the tag headers — are always
/// present once a response exists.
fn apply_header(response: &mut Response, header_name: &'static str, value: &str) {
    if let Ok(v) = header::HeaderValue::from_str(value) {
        response.headers_mut().insert(header_name, v);
    }
}

/// Tags assigned by [`crate::classifiers`]' pre-routing classifiers (e.g.
/// `"vision"`, `"nsfw"`).
const X_ROUTER_REQUEST_TAGS: &str = "x-router-request-tags";
/// Tags assigned by [`crate::classifiers::ResponseClassifier`]s after the
/// provider replies (e.g. `"refusal"`).
const X_ROUTER_RESPONSE_TAGS: &str = "x-router-response-tags";
/// Name of the provider that handled the request (`DispatchOutcome::provider`).
const X_ROUTER_PROVIDER: &str = "x-router-provider";
/// Model name actually sent to that provider (`DispatchOutcome::model`).
const X_ROUTER_MODEL: &str = "x-router-model";
/// `ChatResponse.usage.input_tokens` for this request.
const X_ROUTER_INPUT_TOKENS: &str = "x-router-input-tokens";
/// `ChatResponse.usage.output_tokens` for this request.
const X_ROUTER_OUTPUT_TOKENS: &str = "x-router-output-tokens";

/// Sets all six `X-Router-*` headers.
fn apply_router_headers(
    response: &mut Response,
    request_tags: &[String],
    response_tags: &[String],
    provider: &str,
    model: &str,
    usage: &Usage,
) {
    apply_tags_header(response, X_ROUTER_REQUEST_TAGS, request_tags);
    apply_tags_header(response, X_ROUTER_RESPONSE_TAGS, response_tags);
    apply_header(response, X_ROUTER_PROVIDER, provider);
    apply_header(response, X_ROUTER_MODEL, model);
    apply_header(response, X_ROUTER_INPUT_TOKENS, &usage.input_tokens.to_string());
    apply_header(response, X_ROUTER_OUTPUT_TOKENS, &usage.output_tokens.to_string());
}

async fn chat_completions(
    State(state): State<AppState>,
    Json(body): Json<openai::OpenAiChatRequest>,
) -> Response {
    let streaming = body.stream;
    if streaming {
        dispatch_stream(state, body.into()).await
    } else {
        match dispatch(&state, body.into()).await {
            Ok(outcome) => {
                let response_tags = outcome.response.tags.clone();
                let usage = outcome.response.usage.clone();
                let mut response = Json(openai::OpenAiChatResponse::from(outcome.response)).into_response();
                apply_router_headers(&mut response, &outcome.request_tags, &response_tags, &outcome.provider, &outcome.model, &usage);
                response
            }
            Err(e) => e.into_response(),
        }
    }
}

/// Streaming variant of dispatch: classifies + routes the request, then
/// proxies the provider's SSE stream directly to the client. Plugins are
/// skipped (they operate on complete responses). Emits `RouterEvent::Start`
/// immediately and `RouterEvent::Complete` when the stream finishes.
async fn dispatch_stream(state: AppState, mut req: ChatRequest) -> Response {
    let id = state.next_id.fetch_add(1, Ordering::Relaxed);
    let in_flight = state.in_flight.fetch_add(1, Ordering::Relaxed) + 1;

    let requested_model = req.model.clone();
    let started = Instant::now();

    broadcast(&state, &RouterEvent::Start {
        id,
        ts_ms: LogEntry::now_ms(),
        model: requested_model.clone(),
        in_flight,
    });

    // Classify + route without running plugins.
    req.tags = state.classifiers.classify(&req).await;
    broadcast(&state, &RouterEvent::Classified {
        id,
        ts_ms: LogEntry::now_ms(),
        tags: req.tags.clone(),
    });

    let (provider, target_model) = match state.router.resolve(&req.model, &req.tags) {
        Some(p) => p,
        None => {
            state.in_flight.fetch_sub(1, Ordering::Relaxed);
            return ApiError::NoProvider(req.model.clone()).into_response();
        }
    };

    let provider_name = provider.name.clone();
    let sent_model = target_model.clone();
    let tags = req.tags.clone();
    broadcast(&state, &RouterEvent::Routed {
        id,
        ts_ms: LogEntry::now_ms(),
        provider: provider_name.clone(),
        model: target_model.clone(),
    });
    req.model = target_model;

    let chunk_stream = match provider.send_streaming(&state.client, &req).await {
        Ok(s) => s,
        Err(e) => {
            state.in_flight.fetch_sub(1, Ordering::Relaxed);
            return ApiError::Upstream(e).into_response();
        }
    };

    // Proxy the SSE stream in a spawned task; emit Complete when done.
    let (tx, rx) = tokio::sync::mpsc::channel::<anyhow::Result<bytes::Bytes>>(64);
    let state_clone = state.clone();
    let in_flight_arc = state.in_flight.clone();

    tokio::spawn(async move {
        let _guard = InFlightGuard(in_flight_arc);
        let mut stream = chunk_stream;

        while let Some(item) = stream.next().await {
            if tx.send(item).await.is_err() {
                return; // client disconnected
            }
        }

        broadcast(&state_clone, &RouterEvent::Complete {
            id,
            entry: LogEntry {
                ts_ms: LogEntry::now_ms(),
                provider: provider_name,
                requested_model,
                sent_model,
                duration_ms: started.elapsed().as_millis() as u64,
                tags,
                plugins: Vec::new(),
                system: req.system,
                messages: req.messages,
                response: None,
                error: None,
            },
        });
    });

    let body = Body::from_stream(tokio_stream::wrappers::ReceiverStream::new(rx));
    Response::builder()
        .header(header::CONTENT_TYPE, "text/event-stream")
        .header(header::CACHE_CONTROL, "no-cache")
        .header(header::CONNECTION, "keep-alive")
        .body(body)
        .unwrap()
}

async fn messages(
    State(state): State<AppState>,
    Json(body): Json<anthropic::AnthropicMessagesRequest>,
) -> Response {
    match dispatch(&state, body.into()).await {
        Ok(outcome) => {
            let response_tags = outcome.response.tags.clone();
            let usage = outcome.response.usage.clone();
            let mut response = Json(anthropic::AnthropicMessagesResponse::from(outcome.response)).into_response();
            apply_router_headers(&mut response, &outcome.request_tags, &response_tags, &outcome.provider, &outcome.model, &usage);
            response
        }
        Err(e) => e.into_response(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn header_value<'a>(response: &'a Response, name: &str) -> Option<&'a str> {
        response.headers().get(name).and_then(|v| v.to_str().ok())
    }

    #[test]
    fn provider_model_and_token_headers_always_set() {
        let mut response = StatusCode::OK.into_response();
        let usage = Usage { input_tokens: 12, output_tokens: 34 };
        apply_router_headers(&mut response, &[], &[], "ollama", "llama3.1:8b", &usage);

        assert_eq!(header_value(&response, X_ROUTER_PROVIDER), Some("ollama"));
        assert_eq!(header_value(&response, X_ROUTER_MODEL), Some("llama3.1:8b"));
        assert_eq!(header_value(&response, X_ROUTER_INPUT_TOKENS), Some("12"));
        assert_eq!(header_value(&response, X_ROUTER_OUTPUT_TOKENS), Some("34"));
    }

    #[test]
    fn tag_headers_omitted_when_empty() {
        let mut response = StatusCode::OK.into_response();
        apply_router_headers(&mut response, &[], &[], "openai", "gpt-4o", &Usage::default());

        assert_eq!(header_value(&response, X_ROUTER_REQUEST_TAGS), None);
        assert_eq!(header_value(&response, X_ROUTER_RESPONSE_TAGS), None);
    }

    #[test]
    fn tag_headers_comma_joined_when_present() {
        let mut response = StatusCode::OK.into_response();
        let request_tags = vec!["vision".to_string(), "code".to_string()];
        let response_tags = vec!["refusal".to_string()];
        apply_router_headers(&mut response, &request_tags, &response_tags, "openai", "gpt-4o", &Usage::default());

        assert_eq!(header_value(&response, X_ROUTER_REQUEST_TAGS), Some("vision,code"));
        assert_eq!(header_value(&response, X_ROUTER_RESPONSE_TAGS), Some("refusal"));
    }
}
