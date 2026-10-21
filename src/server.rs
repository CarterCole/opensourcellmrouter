//! HTTP surface: an OpenAI-compatible `/v1/chat/completions` endpoint and an
//! Anthropic-compatible `/v1/messages` endpoint, both backed by the same
//! [`ModelRouter`], plus `/v1/feedback` for correcting the `cev`
//! classifier's routing picks.

use std::convert::Infallible;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use axum::body::Body;
use axum::extract::{Request, State};
use axum::http::{HeaderMap, StatusCode, header};
use axum::middleware::{self, Next};
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{Html, IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router as AxumRouter};
use futures_core::Stream;
use serde::Deserialize;
use serde_json::{json, Map, Value};
use tokio::sync::broadcast;
use tokio_stream::StreamExt;
use tokio_stream::wrappers::BroadcastStream;
use tracing::Instrument;

use crate::canonical::{ChatRequest, ChatResponse, RouteChoice, Usage};
use crate::classifiers::{ClassifierRegistry, ResponseClassifierRegistry};
use crate::formats::{anthropic, openai, responses};
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
    /// Key clients must present to reach any route other than `/health`, if
    /// `[server] api_key_env` is set — see [`require_api_key`]. `None` means
    /// the server is unauthenticated.
    pub api_key: Option<String>,
}

pub fn build_app(state: AppState, dashboard: bool) -> AxumRouter {
    let mut protected = AxumRouter::new()
        .route("/v1/chat/completions", post(chat_completions))
        .route("/v1/messages", post(messages))
        .route("/v1/responses", post(responses_endpoint))
        .route("/v1/feedback", post(feedback));

    if dashboard {
        protected = protected
            .route("/dashboard", get(dashboard_page))
            .route("/dashboard/events", get(dashboard_events));
    }

    let protected = protected.route_layer(middleware::from_fn_with_state(state.clone(), require_api_key));

    AxumRouter::new()
        .route("/health", get(health))
        .merge(protected)
        .with_state(state)
        .layer(tower_http::trace::TraceLayer::new_for_http())
}

/// Pulls the client-presented key out of `Authorization: Bearer <key>`,
/// `x-api-key: <key>`, or a `?api_key=<key>` query parameter (the last is
/// for `EventSource`, which can't set custom headers — see
/// `static/dashboard.html`'s `location.search` passthrough).
fn presented_key(headers: &HeaderMap, query: &str) -> Option<String> {
    if let Some(value) = headers.get(header::AUTHORIZATION).and_then(|v| v.to_str().ok()) {
        if let Some(key) = value.strip_prefix("Bearer ") {
            return Some(key.to_string());
        }
    }
    if let Some(value) = headers.get("x-api-key").and_then(|v| v.to_str().ok()) {
        return Some(value.to_string());
    }
    query
        .split('&')
        .filter_map(|pair| pair.split_once('='))
        .find(|(k, _)| *k == "api_key")
        .map(|(_, v)| v.to_string())
}

/// Rejects any request that doesn't present `state.api_key`, when one is
/// configured. A no-op when `[server] api_key_env` is unset.
async fn require_api_key(State(state): State<AppState>, req: Request, next: Next) -> Response {
    let Some(expected) = &state.api_key else {
        return next.run(req).await;
    };

    let query = req.uri().query().unwrap_or("");
    match presented_key(req.headers(), query) {
        Some(key) if key == *expected => next.run(req).await,
        _ => ApiError::Unauthorized.into_response(),
    }
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
    Unauthorized,
    /// Feedback for the `cev` classifier couldn't be recorded.
    Feedback(StatusCode, String),
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
            ApiError::Unauthorized => (
                StatusCode::UNAUTHORIZED,
                "missing or invalid API key".to_string(),
            ),
            ApiError::Feedback(status, message) => (status, message),
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

/// The OTel trace id of the current `tracing` span, if telemetry is enabled
/// (i.e. an OTel layer is registered and produced a valid trace context).
/// `None` when telemetry is disabled.
fn current_trace_id() -> Option<String> {
    use opentelemetry::trace::TraceContextExt;
    use tracing_opentelemetry::OpenTelemetrySpanExt;

    let context = tracing::Span::current().context();
    let trace_id = context.span().span_context().trace_id();
    if trace_id == opentelemetry::trace::TraceId::INVALID {
        None
    } else {
        Some(trace_id.to_string())
    }
}

/// Records request-count, duration, and token metrics via the globally
/// registered OTel `MeterProvider`. A no-op when telemetry is disabled (the
/// global default meter provider is a no-op implementation).
fn record_metrics(provider: &str, model: &str, duration: Duration, usage: Option<&Usage>, error: bool) {
    let meter = opentelemetry::global::meter("opensourcellmrouter");
    let attrs = [
        opentelemetry::KeyValue::new("provider", provider.to_string()),
        opentelemetry::KeyValue::new("model", model.to_string()),
        opentelemetry::KeyValue::new("error", error),
    ];
    meter.u64_counter("router.requests").build().add(1, &attrs);
    meter
        .f64_histogram("router.request.duration_ms")
        .build()
        .record(duration.as_millis() as f64, &attrs);
    if let Some(usage) = usage {
        meter.u64_counter("router.tokens.input").build().add(usage.input_tokens as u64, &attrs);
        meter.u64_counter("router.tokens.output").build().add(usage.output_tokens as u64, &attrs);
    }
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
    /// The model a classifier picked for the request, if one did (see
    /// [`crate::classifiers::cev`] and [`crate::classifiers::llm`]).
    route: Option<RouteChoice>,
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
#[tracing::instrument(
    name = "dispatch",
    skip(state, req),
    fields(request_id = tracing::field::Empty, provider = tracing::field::Empty,
           model = tracing::field::Empty, tags = tracing::field::Empty)
)]
async fn dispatch(state: &AppState, mut req: ChatRequest) -> Result<DispatchOutcome, ApiError> {
    let id = state.next_id.fetch_add(1, Ordering::Relaxed);
    tracing::Span::current().record("request_id", id);
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
        let classification = state.classifiers.classify(&req).await;
        req.tags = classification.tags;
        req.route = classification.route;
        tracing::Span::current().record("tags", format!("{:?}", req.tags));
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
                    .resolve(&req.model, &req.tags, req.route.as_ref(), &req.needed_capabilities())
                    .ok_or_else(|| ApiError::NoProvider(req.model.clone()))?,
            };

            sent_model = target_model.clone();
            provider_name = provider.name.clone();
            tracing::Span::current().record("provider", &provider_name);
            tracing::Span::current().record("model", &sent_model);
            broadcast(state, &RouterEvent::Routed {
                id,
                ts_ms: LogEntry::now_ms(),
                provider: provider_name.clone(),
                model: target_model.clone(),
            });
            req.model = target_model;

            let send_span = tracing::info_span!("provider.send", provider = %provider_name, model = %sent_model);
            match provider.send(&state.client, &req).instrument(send_span).await {
                Ok(r) => resp = Some(r),
                Err(err) => {
                    let duration = started.elapsed();
                    record_metrics(&provider_name, &sent_model, duration, None, true);
                    broadcast(state, &RouterEvent::Complete {
                        id,
                        entry: LogEntry {
                            ts_ms: LogEntry::now_ms(),
                            provider: provider_name,
                            requested_model,
                            sent_model,
                            duration_ms: duration.as_millis() as u64,
                            tags: req.tags,
                            route: req.route,
                            plugins: active_plugins,
                            system: req.system,
                            messages: req.messages,
                            response: None,
                            error: Some(err.to_string()),
                            trace_id: current_trace_id(),
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
    let route = req.route.clone();
    let provider = provider_name.clone();
    let model = sent_model.clone();
    let duration = started.elapsed();
    record_metrics(&provider_name, &sent_model, duration, resp.as_ref().map(|r| &r.usage), false);
    broadcast(state, &RouterEvent::Complete {
        id,
        entry: LogEntry {
            ts_ms: LogEntry::now_ms(),
            provider: provider_name,
            requested_model,
            sent_model,
            duration_ms: duration.as_millis() as u64,
            tags: req.tags,
            route: req.route,
            plugins: active_plugins,
            system: req.system,
            messages: req.messages,
            response: resp.clone(),
            error: None,
            trace_id: current_trace_id(),
        },
    });

    resp.ok_or(ApiError::NoResponse).map(|response| DispatchOutcome {
        response,
        request_tags,
        provider,
        model,
        route,
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

/// Response header naming the `cev` classifier's decision for a request
/// (`RouteChoice::decision_id`), so the client can correct it later via
/// `POST /v1/feedback`.
const X_CEV_DECISION_ID: &str = "x-cev-decision-id";
/// Name of the option the `cev` classifier picked (`RouteChoice::choice`).
const X_CEV_CHOICE: &str = "x-cev-choice";

/// Sets [`X_CEV_DECISION_ID`] and [`X_CEV_CHOICE`] when the `cev`
/// classifier picked the model for this request; omitted otherwise
/// (including for `llm` picks, which have no decision to correct).
fn apply_route_headers(response: &mut Response, route: Option<&RouteChoice>) {
    if let Some(route) = route {
        if let Some(decision_id) = &route.decision_id {
            apply_header(response, X_CEV_DECISION_ID, decision_id);
            apply_header(response, X_CEV_CHOICE, &route.choice);
        }
    }
}

/// Which wire format the client that hit a streaming endpoint expects its
/// SSE response rendered in. Deliberately distinct from
/// [`crate::config::ProviderFormat`] — that describes the *upstream*, and
/// conflating the two was the root cause of the bug this type fixes (an
/// Anthropic client receiving OpenAI-shaped chunks because the proxy never
/// distinguished them).
#[derive(Clone, Copy)]
enum ClientFormat {
    OpenAi,
    Anthropic,
    Responses,
}

async fn chat_completions(
    State(state): State<AppState>,
    Json(body): Json<openai::OpenAiChatRequest>,
) -> Response {
    let streaming = body.stream;
    if streaming {
        dispatch_stream(state, body.into(), ClientFormat::OpenAi).await
    } else {
        match dispatch(&state, body.into()).await {
            Ok(outcome) => {
                let response_tags = outcome.response.tags.clone();
                let usage = outcome.response.usage.clone();
                let mut response = Json(openai::OpenAiChatResponse::from(outcome.response)).into_response();
                apply_router_headers(&mut response, &outcome.request_tags, &response_tags, &outcome.provider, &outcome.model, &usage);
                apply_route_headers(&mut response, outcome.route.as_ref());
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
#[tracing::instrument(
    name = "dispatch_stream",
    skip(state, req, client_format),
    fields(request_id = tracing::field::Empty, provider = tracing::field::Empty,
           model = tracing::field::Empty, tags = tracing::field::Empty)
)]
async fn dispatch_stream(state: AppState, mut req: ChatRequest, client_format: ClientFormat) -> Response {
    let id = state.next_id.fetch_add(1, Ordering::Relaxed);
    tracing::Span::current().record("request_id", id);
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
    let classification = state.classifiers.classify(&req).await;
    req.tags = classification.tags;
    req.route = classification.route;
    tracing::Span::current().record("tags", format!("{:?}", req.tags));
    broadcast(&state, &RouterEvent::Classified {
        id,
        ts_ms: LogEntry::now_ms(),
        tags: req.tags.clone(),
    });

    let (provider, target_model) = match state.router.resolve(&req.model, &req.tags, req.route.as_ref(), &req.needed_capabilities()) {
        Some(p) => p,
        None => {
            state.in_flight.fetch_sub(1, Ordering::Relaxed);
            return ApiError::NoProvider(req.model.clone()).into_response();
        }
    };

    let provider_name = provider.name.clone();
    let sent_model = target_model.clone();
    let tags = req.tags.clone();
    let route = req.route.clone();
    tracing::Span::current().record("provider", &provider_name);
    tracing::Span::current().record("model", &sent_model);
    broadcast(&state, &RouterEvent::Routed {
        id,
        ts_ms: LogEntry::now_ms(),
        provider: provider_name.clone(),
        model: target_model.clone(),
    });
    req.model = target_model;

    let send_span = tracing::info_span!("provider.send", provider = %provider_name, model = %sent_model);
    let chunk_stream = match provider.send_streaming(&state.client, &req).instrument(send_span).await {
        Ok(s) => s,
        Err(e) => {
            state.in_flight.fetch_sub(1, Ordering::Relaxed);
            return ApiError::Upstream(e).into_response();
        }
    };

    // Render the canonical event stream into whichever wire format the
    // client that hit this endpoint expects.
    let rendered_stream = match client_format {
        ClientFormat::OpenAi => openai::render_stream(chunk_stream, req.model.clone()),
        ClientFormat::Anthropic => anthropic::render_stream(chunk_stream, req.model.clone()),
        ClientFormat::Responses => responses::render_stream(chunk_stream, req.model.clone()),
    };

    // Proxy the rendered SSE stream in a spawned task; emit Complete when done.
    let (tx, rx) = tokio::sync::mpsc::channel::<anyhow::Result<bytes::Bytes>>(64);
    let state_clone = state.clone();
    let in_flight_arc = state.in_flight.clone();

    let stream_span = tracing::info_span!("dispatch_stream.body", request_id = id, provider = %provider_name, model = %sent_model);
    tokio::spawn(
        async move {
            let _guard = InFlightGuard(in_flight_arc);
            let mut stream = rendered_stream;

            while let Some(item) = stream.next().await {
                if tx.send(item).await.is_err() {
                    return; // client disconnected
                }
            }

            let duration = started.elapsed();
            record_metrics(&provider_name, &sent_model, duration, None, false);
            broadcast(&state_clone, &RouterEvent::Complete {
                id,
                entry: LogEntry {
                    ts_ms: LogEntry::now_ms(),
                    provider: provider_name,
                    requested_model,
                    sent_model,
                    duration_ms: duration.as_millis() as u64,
                    tags,
                    route: req.route,
                    plugins: Vec::new(),
                    system: req.system,
                    messages: req.messages,
                    response: None,
                    error: None,
                    trace_id: current_trace_id(),
                },
            });
        }
        .instrument(stream_span),
    );

    let body = Body::from_stream(tokio_stream::wrappers::ReceiverStream::new(rx));
    let mut response = Response::builder()
        .header(header::CONTENT_TYPE, "text/event-stream")
        .header(header::CACHE_CONTROL, "no-cache")
        .header(header::CONNECTION, "keep-alive")
        .body(body)
        .unwrap();
    apply_route_headers(&mut response, route.as_ref());
    response
}

async fn messages(
    State(state): State<AppState>,
    Json(body): Json<anthropic::AnthropicMessagesRequest>,
) -> Response {
    let streaming = body.stream;
    if streaming {
        dispatch_stream(state, body.into(), ClientFormat::Anthropic).await
    } else {
        match dispatch(&state, body.into()).await {
            Ok(outcome) => {
                let response_tags = outcome.response.tags.clone();
                let usage = outcome.response.usage.clone();
                let mut response = Json(anthropic::AnthropicMessagesResponse::from(outcome.response)).into_response();
                apply_router_headers(&mut response, &outcome.request_tags, &response_tags, &outcome.provider, &outcome.model, &usage);
                apply_route_headers(&mut response, outcome.route.as_ref());
                response
            }
            Err(e) => e.into_response(),
        }
    }
}

async fn responses_endpoint(
    State(state): State<AppState>,
    Json(body): Json<responses::ResponsesRequest>,
) -> Response {
    let streaming = body.stream;
    if streaming {
        dispatch_stream(state, body.into(), ClientFormat::Responses).await
    } else {
        match dispatch(&state, body.into()).await {
            Ok(outcome) => {
                let response_tags = outcome.response.tags.clone();
                let usage = outcome.response.usage.clone();
                let mut response = Json(responses::ResponsesResponse::from(outcome.response)).into_response();
                apply_router_headers(&mut response, &outcome.request_tags, &response_tags, &outcome.provider, &outcome.model, &usage);
                apply_route_headers(&mut response, outcome.route.as_ref());
                response
            }
            Err(e) => e.into_response(),
        }
    }
}

#[derive(Deserialize)]
struct FeedbackRequest {
    /// The `x-cev-decision-id` of the request being corrected.
    decision_id: String,
    /// Name of the model that should have been picked for it.
    label: String,
    comment: Option<String>,
}

/// Tells the `cev` classifier which model it should have picked for an
/// earlier request, which it learns from (see
/// [`crate::classifiers::cev::CevClassifier::feedback`]).
async fn feedback(
    State(state): State<AppState>,
    Json(body): Json<FeedbackRequest>,
) -> Result<Json<cev::FeedbackResponse>, ApiError> {
    let classifier = state
        .classifiers
        .cev()
        .ok_or_else(|| ApiError::Feedback(StatusCode::NOT_FOUND, "the 'cev' classifier is not enabled".to_string()))?;

    match classifier.feedback(&body.decision_id, &body.label, body.comment).await {
        Ok(learned) => Ok(Json(learned)),
        Err(cev::Error::Api { status, message }) => Err(ApiError::Feedback(
            StatusCode::from_u16(status).unwrap_or(StatusCode::BAD_GATEWAY),
            message,
        )),
        Err(err) => Err(ApiError::Feedback(StatusCode::BAD_GATEWAY, err.to_string())),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::classifiers::cev::tests::mock_classifier;
    use crate::classifiers::llm::tests::mock_provider;
    use crate::config::Config;

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

    fn headers_with(name: &str, value: &str) -> HeaderMap {
        let mut headers = HeaderMap::new();
        headers.insert(header::HeaderName::from_bytes(name.as_bytes()).unwrap(), value.parse().unwrap());
        headers
    }

    #[test]
    fn presented_key_reads_bearer_auth_header() {
        let headers = headers_with("authorization", "Bearer secret123");
        assert_eq!(presented_key(&headers, ""), Some("secret123".to_string()));
    }

    #[test]
    fn presented_key_reads_x_api_key_header() {
        let headers = headers_with("x-api-key", "secret123");
        assert_eq!(presented_key(&headers, ""), Some("secret123".to_string()));
    }

    #[test]
    fn presented_key_reads_query_param() {
        let headers = HeaderMap::new();
        assert_eq!(presented_key(&headers, "foo=bar&api_key=secret123"), Some("secret123".to_string()));
    }

    #[test]
    fn presented_key_none_when_absent() {
        let headers = HeaderMap::new();
        assert_eq!(presented_key(&headers, ""), None);
    }

    #[test]
    fn presented_key_prefers_header_over_query() {
        let headers = headers_with("x-api-key", "from-header");
        assert_eq!(presented_key(&headers, "api_key=from-query"), Some("from-header".to_string()));
    }

    /// Serves `app` on a free local port and returns its base URL.
    async fn serve(app: AxumRouter) -> String {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let url = format!("http://{}", listener.local_addr().unwrap());
        tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        url
    }

    #[tokio::test]
    async fn cev_pick_routes_the_request_and_takes_feedback() {
        // A provider that answers with the name of the model it was asked for.
        let provider = serve(AxumRouter::new().route(
            "/v1/chat/completions",
            post(|Json(body): Json<Value>| async move {
                Json(json!({
                    "id": "chatcmpl-1",
                    "object": "chat.completion",
                    "model": body["model"],
                    "choices": [{
                        "index": 0,
                        "message": {"role": "assistant", "content": body["model"]},
                        "finish_reason": "stop",
                    }],
                }))
            }),
        ))
        .await;

        let config: Config = toml::from_str(&format!(
            r#"
            [[providers]]
            name = "ollama"
            format = "openai"
            base_url = "{provider}/v1"

            [[routers]]
            type = "cev"

            [[routers]]
            type = "fallback"
            "#
        ))
        .unwrap();

        let (events, mut logged) = broadcast::channel(8);
        let state = AppState {
            router: Arc::new(ModelRouter::from_config(&config).unwrap()),
            client: reqwest::Client::new(),
            logger: None,
            plugins: Arc::new(PluginRegistry::from_config(&config)),
            classifiers: Arc::new(ClassifierRegistry::with_cev(mock_classifier())),
            response_classifiers: Arc::new(ResponseClassifierRegistry::from_config(&config)),
            events,
            in_flight: Arc::new(AtomicU64::new(0)),
            next_id: Arc::new(AtomicU64::new(0)),
            api_key: None,
        };
        let router = serve(build_app(state, false)).await;
        let client = reqwest::Client::new();

        // The request goes to the model cev picked, not the "gpt-4" it named,
        // and the response says which decision that was.
        let resp = client
            .post(format!("{router}/v1/chat/completions"))
            .json(&json!({
                "model": "gpt-4",
                "messages": [{"role": "user", "content": "Help me with debugging these algorithms"}],
            }))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        assert_eq!(resp.headers()["x-cev-choice"], "deepseek-r1:latest");
        let decision_id = resp.headers()["x-cev-decision-id"].to_str().unwrap().to_string();
        let body: Value = resp.json().await.unwrap();
        assert_eq!(body["choices"][0]["message"]["content"], "deepseek-r1:latest");

        // The same decision is in the request log, for the dashboard.
        let entry = loop {
            let event: Value = serde_json::from_str(&logged.recv().await.unwrap()).unwrap();
            if event["type"] == "complete" {
                break event;
            }
        };
        assert_eq!(entry["sent_model"], "deepseek-r1:latest");
        assert_eq!(entry["route"]["decision_id"], decision_id.as_str());
        assert_eq!(entry["route"]["choice"], "deepseek-r1:latest");

        // Correcting it by decision id reaches cev, which learns from it.
        let correct = |label: &'static str| {
            client
                .post(format!("{router}/v1/feedback"))
                .json(&json!({"decision_id": decision_id, "label": label}))
                .send()
        };
        let resp = correct("general").await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let learned: Value = resp.json().await.unwrap();
        assert_eq!(learned["learned"], true);
        assert_eq!(learned["task_examples"], 1);

        // A label that isn't one of the models is the caller's mistake.
        let resp = correct("gpt-4").await.unwrap();
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
        let error: Value = resp.json().await.unwrap();
        assert_eq!(error["error"]["type"], "router_error");
    }

    #[tokio::test]
    async fn llm_pick_routes_the_request() {
        // The provider the `llm` classifier asks, which always says "general",
        // and the one that answers.
        let (asked, _) = mock_provider("general").await;
        let (answering, received) = mock_provider("an answer").await;

        let config: Config = toml::from_str(&format!(
            r#"
            [[providers]]
            name = "router"
            format = "openai"
            base_url = "{asked}"

            [[providers]]
            name = "local"
            format = "openai"
            base_url = "{answering}"

            [classifiers.llm]
            enabled = true
            provider = "router"
            model = "router-model"

            [[classifiers.llm.models]]
            provider = "local"
            model = "deepseek-r1:latest"

            [[classifiers.llm.models]]
            provider = "local"
            model = "llama3.1:8b"
            name = "general"

            [[routers]]
            type = "llm"

            [[routers]]
            type = "fallback"
            providers = ["router"]
            "#
        ))
        .unwrap();

        let (events, mut logged) = broadcast::channel(8);
        let state = AppState {
            router: Arc::new(ModelRouter::from_config(&config).unwrap()),
            client: reqwest::Client::new(),
            logger: None,
            plugins: Arc::new(PluginRegistry::from_config(&config)),
            classifiers: Arc::new(ClassifierRegistry::from_config(&config).unwrap()),
            response_classifiers: Arc::new(ResponseClassifierRegistry::from_config(&config)),
            events,
            in_flight: Arc::new(AtomicU64::new(0)),
            next_id: Arc::new(AtomicU64::new(0)),
            api_key: None,
        };
        let router = serve(build_app(state, false)).await;

        let resp = reqwest::Client::new()
            .post(format!("{router}/v1/messages"))
            .json(&json!({
                "model": "claude-sonnet-5-5",
                "max_tokens": 100,
                "messages": [{"role": "user", "content": "Any recipes for my travel?"}],
            }))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        assert_eq!(resp.headers()[X_ROUTER_PROVIDER], "local");
        assert_eq!(resp.headers()[X_ROUTER_MODEL], "llama3.1:8b");
        // There's no cev decision to correct.
        assert!(resp.headers().get(X_CEV_DECISION_ID).is_none());
        assert_eq!(received.lock().unwrap()[0]["model"], "llama3.1:8b");

        let entry = loop {
            let event: Value = serde_json::from_str(&logged.recv().await.unwrap()).unwrap();
            if event["type"] == "complete" {
                break event;
            }
        };
        assert_eq!(entry["route"]["classifier"], "llm");
        assert_eq!(entry["route"]["choice"], "general");
    }
}
