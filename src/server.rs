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
use serde_json::{Map, Value, json};
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
        protected =
            protected.route("/dashboard", get(dashboard_page)).route("/dashboard/events", get(dashboard_events));
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
    query.split('&').filter_map(|pair| pair.split_once('=')).find(|(k, _)| *k == "api_key").map(|(_, v)| v.to_string())
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
async fn dashboard_events(State(state): State<AppState>) -> Sse<impl Stream<Item = Result<Event, Infallible>>> {
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
            ApiError::NoProvider(model) => {
                (StatusCode::BAD_REQUEST, format!("no provider configured for model '{model}'"))
            }
            ApiError::Upstream(err) => (StatusCode::BAD_GATEWAY, err.to_string()),
            ApiError::Plugin(id, err) => (StatusCode::INTERNAL_SERVER_ERROR, format!("plugin '{id}' failed: {err}")),
            ApiError::NoResponse => {
                (StatusCode::INTERNAL_SERVER_ERROR, "pipeline stopped without producing a response".to_string())
            }
            ApiError::Unauthorized => (StatusCode::UNAUTHORIZED, "missing or invalid API key".to_string()),
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
        let ctx = PluginContext { client: client.clone(), settings: settings.clone() };

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
    if trace_id == opentelemetry::trace::TraceId::INVALID { None } else { Some(trace_id.to_string()) }
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
    meter.f64_histogram("router.request.duration_ms").build().record(duration.as_millis() as f64, &attrs);
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
    /// The models classifiers picked for the request (see
    /// [`crate::classifiers::cev`] and [`crate::classifiers::llm`]).
    routes: Vec<RouteChoice>,
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

    broadcast(state, &RouterEvent::Start { id, ts_ms: LogEntry::now_ms(), model: requested_model.clone(), in_flight });

    let resolved_plugins = state.plugins.resolve(&req);
    let mut resp: Option<ChatResponse> = None;
    let mut active_plugins: Vec<String> = Vec::new();
    let mut sent_model = req.model.clone();
    let mut provider_name = "plugin".to_string();

    if run_stage(&resolved_plugins, &state.client, Stage::Start, &mut req, &mut resp, &mut active_plugins).await?
        == Flow::Continue
    {
        let classified = state.classifiers.classify(&req).await;
        req.tags = classified.tags;
        req.routes = classified.routes;
        tracing::Span::current().record("tags", format!("{:?}", req.tags));
        broadcast(state, &RouterEvent::Classified { id, ts_ms: LogEntry::now_ms(), tags: req.tags.clone() });

        let routing_flow =
            run_stage(&resolved_plugins, &state.client, Stage::PreRouting, &mut req, &mut resp, &mut active_plugins)
                .await?;

        if routing_flow == Flow::Continue && resp.is_none() {
            let (provider, target_model) = match &req.forced_provider {
                Some(name) => {
                    let provider =
                        state.router.provider(name).ok_or_else(|| ApiError::NoProvider(req.model.clone()))?;
                    (provider, req.model.clone())
                }
                None => state
                    .router
                    .resolve(&req.model, &req.tags, &req.routes, &req.needed_capabilities())
                    .ok_or_else(|| ApiError::NoProvider(req.model.clone()))?,
            };

            sent_model = target_model.clone();
            provider_name = provider.name.clone();
            tracing::Span::current().record("provider", &provider_name);
            tracing::Span::current().record("model", &sent_model);
            broadcast(
                state,
                &RouterEvent::Routed {
                    id,
                    ts_ms: LogEntry::now_ms(),
                    provider: provider_name.clone(),
                    model: target_model.clone(),
                },
            );
            req.model = target_model;

            let send_span = tracing::info_span!("provider.send", provider = %provider_name, model = %sent_model);
            match provider.send(&state.client, &req).instrument(send_span).await {
                Ok(r) => resp = Some(r),
                Err(err) => {
                    let duration = started.elapsed();
                    record_metrics(&provider_name, &sent_model, duration, None, true);
                    broadcast(
                        state,
                        &RouterEvent::Complete {
                            id,
                            entry: LogEntry {
                                ts_ms: LogEntry::now_ms(),
                                provider: provider_name,
                                requested_model,
                                sent_model,
                                duration_ms: duration.as_millis() as u64,
                                tags: req.tags,
                                routes: req.routes,
                                plugins: active_plugins,
                                system: req.system,
                                messages: req.messages,
                                response: None,
                                error: Some(err.to_string()),
                                trace_id: current_trace_id(),
                            },
                        },
                    );
                    return Err(ApiError::Upstream(err));
                }
            }
        }
    }

    if run_stage(&resolved_plugins, &state.client, Stage::PostResponse, &mut req, &mut resp, &mut active_plugins)
        .await?
        == Flow::Continue
    {
        if let Some(r) = &mut resp {
            r.tags = state.response_classifiers.classify(&req, r).await;
        }

        run_stage(&resolved_plugins, &state.client, Stage::End, &mut req, &mut resp, &mut active_plugins).await?;
    }

    let request_tags = req.tags.clone();
    let routes = req.routes.clone();
    let provider = provider_name.clone();
    let model = sent_model.clone();
    let duration = started.elapsed();
    record_metrics(&provider_name, &sent_model, duration, resp.as_ref().map(|r| &r.usage), false);
    broadcast(
        state,
        &RouterEvent::Complete {
            id,
            entry: LogEntry {
                ts_ms: LogEntry::now_ms(),
                provider: provider_name,
                requested_model,
                sent_model,
                duration_ms: duration.as_millis() as u64,
                tags: req.tags,
                routes: req.routes,
                plugins: active_plugins,
                system: req.system,
                messages: req.messages,
                response: resp.clone(),
                error: None,
                trace_id: current_trace_id(),
            },
        },
    );

    resp.ok_or(ApiError::NoResponse).map(|response| DispatchOutcome { response, request_tags, provider, model, routes })
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
/// classifier decided this request, whether or not its pick routed it;
/// omitted otherwise (an `llm` pick has no decision of its own).
fn apply_route_headers(response: &mut Response, routes: &[RouteChoice]) {
    for route in routes {
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

async fn chat_completions(State(state): State<AppState>, Json(body): Json<openai::OpenAiChatRequest>) -> Response {
    let streaming = body.stream;
    if streaming {
        dispatch_stream(state, body.into(), ClientFormat::OpenAi).await
    } else {
        match dispatch(&state, body.into()).await {
            Ok(outcome) => {
                let response_tags = outcome.response.tags.clone();
                let usage = outcome.response.usage.clone();
                let mut response = Json(openai::OpenAiChatResponse::from(outcome.response)).into_response();
                apply_router_headers(
                    &mut response,
                    &outcome.request_tags,
                    &response_tags,
                    &outcome.provider,
                    &outcome.model,
                    &usage,
                );
                apply_route_headers(&mut response, &outcome.routes);
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

    broadcast(&state, &RouterEvent::Start { id, ts_ms: LogEntry::now_ms(), model: requested_model.clone(), in_flight });

    // Classify + route without running plugins.
    let classified = state.classifiers.classify(&req).await;
    req.tags = classified.tags;
    req.routes = classified.routes;
    tracing::Span::current().record("tags", format!("{:?}", req.tags));
    broadcast(&state, &RouterEvent::Classified { id, ts_ms: LogEntry::now_ms(), tags: req.tags.clone() });

    let (provider, target_model) =
        match state.router.resolve(&req.model, &req.tags, &req.routes, &req.needed_capabilities()) {
            Some(p) => p,
            None => {
                state.in_flight.fetch_sub(1, Ordering::Relaxed);
                return ApiError::NoProvider(req.model.clone()).into_response();
            }
        };

    let provider_name = provider.name.clone();
    let sent_model = target_model.clone();
    let tags = req.tags.clone();
    let routes = req.routes.clone();
    tracing::Span::current().record("provider", &provider_name);
    tracing::Span::current().record("model", &sent_model);
    broadcast(
        &state,
        &RouterEvent::Routed {
            id,
            ts_ms: LogEntry::now_ms(),
            provider: provider_name.clone(),
            model: target_model.clone(),
        },
    );
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

    let stream_span =
        tracing::info_span!("dispatch_stream.body", request_id = id, provider = %provider_name, model = %sent_model);
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
            broadcast(
                &state_clone,
                &RouterEvent::Complete {
                    id,
                    entry: LogEntry {
                        ts_ms: LogEntry::now_ms(),
                        provider: provider_name,
                        requested_model,
                        sent_model,
                        duration_ms: duration.as_millis() as u64,
                        tags,
                        routes: req.routes,
                        plugins: Vec::new(),
                        system: req.system,
                        messages: req.messages,
                        response: None,
                        error: None,
                        trace_id: current_trace_id(),
                    },
                },
            );
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
    apply_route_headers(&mut response, &routes);
    response
}

async fn messages(State(state): State<AppState>, Json(body): Json<anthropic::AnthropicMessagesRequest>) -> Response {
    let streaming = body.stream;
    if streaming {
        dispatch_stream(state, body.into(), ClientFormat::Anthropic).await
    } else {
        match dispatch(&state, body.into()).await {
            Ok(outcome) => {
                let response_tags = outcome.response.tags.clone();
                let usage = outcome.response.usage.clone();
                let mut response = Json(anthropic::AnthropicMessagesResponse::from(outcome.response)).into_response();
                apply_router_headers(
                    &mut response,
                    &outcome.request_tags,
                    &response_tags,
                    &outcome.provider,
                    &outcome.model,
                    &usage,
                );
                apply_route_headers(&mut response, &outcome.routes);
                response
            }
            Err(e) => e.into_response(),
        }
    }
}

async fn responses_endpoint(State(state): State<AppState>, Json(body): Json<responses::ResponsesRequest>) -> Response {
    let streaming = body.stream;
    if streaming {
        dispatch_stream(state, body.into(), ClientFormat::Responses).await
    } else {
        match dispatch(&state, body.into()).await {
            Ok(outcome) => {
                let response_tags = outcome.response.tags.clone();
                let usage = outcome.response.usage.clone();
                let mut response = Json(responses::ResponsesResponse::from(outcome.response)).into_response();
                apply_router_headers(
                    &mut response,
                    &outcome.request_tags,
                    &response_tags,
                    &outcome.provider,
                    &outcome.model,
                    &usage,
                );
                apply_route_headers(&mut response, &outcome.routes);
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
        Err(cev::Error::Api { status, message }) => {
            Err(ApiError::Feedback(StatusCode::from_u16(status).unwrap_or(StatusCode::BAD_GATEWAY), message))
        }
        Err(err) => Err(ApiError::Feedback(StatusCode::BAD_GATEWAY, err.to_string())),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::classifiers::cev::tests::{mock_classifier, mock_classifier_and_runtime};
    use crate::classifiers::llm::LlmClassifier;
    use crate::classifiers::llm::tests::{classifier as llm_classifier, mock_provider, teacher as llm_teacher};
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

    /// Serves a provider that answers with the name of the model it was
    /// asked for, and returns its base URL.
    async fn echo_model_provider() -> String {
        serve(AxumRouter::new().route(
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
        .await
    }

    /// Waits for the `complete` event on the dashboard feed.
    async fn completed(logged: &mut broadcast::Receiver<Arc<str>>) -> Value {
        loop {
            let event: Value = serde_json::from_str(&logged.recv().await.unwrap()).unwrap();
            if event["type"] == "complete" {
                break event;
            }
        }
    }

    #[tokio::test]
    async fn cev_pick_routes_the_request_and_takes_feedback() {
        let provider = echo_model_provider().await;

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
            classifiers: Arc::new(ClassifierRegistry::with_cev(mock_classifier(), None)),
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
        let entry = completed(&mut logged).await;
        assert_eq!(entry["sent_model"], "deepseek-r1:latest");
        assert_eq!(entry["routes"][0]["decision_id"], decision_id.as_str());
        assert_eq!(entry["routes"][0]["choice"], "deepseek-r1:latest");

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

    /// Serves a router with the mock `cev` classifier and an `llm` one built
    /// by `llm` (from the URL of a provider that always answers "general"),
    /// routing on `llm`'s pick. Returns its URL, cev's runtime and the
    /// dashboard feed.
    async fn serve_cev_and_llm(
        llm: fn(&str) -> LlmClassifier,
    ) -> (String, Arc<cev_runtime::Cev>, broadcast::Receiver<Arc<str>>) {
        let provider = echo_model_provider().await;
        let (asked, _) = mock_provider("general").await;

        let config: Config = toml::from_str(&format!(
            r#"
            [[providers]]
            name = "ollama"
            format = "openai"
            base_url = "{provider}/v1"

            [[routers]]
            type = "llm"

            [[routers]]
            type = "fallback"
            "#
        ))
        .unwrap();

        let (cev, runtime) = mock_classifier_and_runtime();
        let (events, logged) = broadcast::channel(8);
        let state = AppState {
            router: Arc::new(ModelRouter::from_config(&config).unwrap()),
            client: reqwest::Client::new(),
            logger: None,
            plugins: Arc::new(PluginRegistry::from_config(&config)),
            classifiers: Arc::new(ClassifierRegistry::with_cev(cev, Some(llm(&asked)))),
            response_classifiers: Arc::new(ResponseClassifierRegistry::from_config(&config)),
            events,
            in_flight: Arc::new(AtomicU64::new(0)),
            next_id: Arc::new(AtomicU64::new(0)),
            api_key: None,
        };
        (serve(build_app(state, false)).await, runtime, logged)
    }

    #[tokio::test]
    async fn a_teacher_llm_routes_the_request_and_labels_cevs_decision() {
        let (router, runtime, mut logged) = serve_cev_and_llm(llm_teacher).await;
        let client = reqwest::Client::new();

        // cev would pick the coding model, but the `llm` rule routes on
        // llm's pick. cev's decision still comes back, to be corrected.
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
        assert_eq!(resp.headers()[X_ROUTER_MODEL], "llama3.1:8b");
        assert_eq!(resp.headers()[X_CEV_CHOICE], "deepseek-r1:latest");
        let decision_id = resp.headers()[X_CEV_DECISION_ID].to_str().unwrap().to_string();

        // Both picks are in the request log.
        let entry = completed(&mut logged).await;
        let picks: Vec<_> =
            entry["routes"].as_array().unwrap().iter().map(|r| (&r["classifier"], &r["choice"])).collect();
        assert_eq!(picks, [(&json!("cev"), &json!("deepseek-r1:latest")), (&json!("llm"), &json!("general"))]);

        // llm's pick labels cev's decision, in the background.
        let mut labels = Vec::new();
        for _ in 0..100 {
            labels = runtime.decision(&decision_id).unwrap().unwrap().1;
            if !labels.is_empty() {
                break;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        assert_eq!(labels.len(), 1, "no label from llm's pick");
        assert_eq!(serde_json::to_value(&labels[0].label).unwrap(), "general");
        assert_eq!(labels[0].metadata, Some(json!({"source": "llm", "provider": "router", "model": "router-model"})));

        // A correction replaces it.
        let resp = client
            .post(format!("{router}/v1/feedback"))
            .json(&json!({"decision_id": decision_id, "label": "deepseek-r1:latest"}))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let learned: Value = resp.json().await.unwrap();
        assert_eq!(learned["learned"], true);
        assert_eq!(learned["task_examples"], 2);
    }

    #[tokio::test]
    async fn an_llm_that_is_not_a_teacher_labels_nothing() {
        let (router, runtime, mut logged) = serve_cev_and_llm(llm_classifier).await;

        let resp = reqwest::Client::new()
            .post(format!("{router}/v1/chat/completions"))
            .json(&json!({
                "model": "gpt-4",
                "messages": [{"role": "user", "content": "Help me with debugging these algorithms"}],
            }))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        assert_eq!(resp.headers()[X_ROUTER_MODEL], "llama3.1:8b");
        let decision_id = resp.headers()[X_CEV_DECISION_ID].to_str().unwrap().to_string();
        completed(&mut logged).await;

        // Labels would have landed within this long, as in the test above.
        tokio::time::sleep(Duration::from_millis(200)).await;
        assert!(runtime.decision(&decision_id).unwrap().unwrap().1.is_empty());
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

        let entry = completed(&mut logged).await;
        assert_eq!(entry["routes"][0]["classifier"], "llm");
        assert_eq!(entry["routes"][0]["choice"], "general");
    }

    /// An upstream Anthropic streaming reply that says something, then calls
    /// a tool — shaped like the real API's, `ping` and a thinking block
    /// (which the router drops) included.
    const ANTHROPIC_TOOL_USE_SSE: &str = concat!(
        "event: message_start\n",
        r#"data: {"type":"message_start","message":{"id":"msg_up","type":"message","role":"assistant","content":[],"model":"claude-opus-4-8","stop_reason":null,"stop_sequence":null,"usage":{"input_tokens":25,"output_tokens":1}}}"#,
        "\n\n",
        "event: content_block_start\n",
        r#"data: {"type":"content_block_start","index":0,"content_block":{"type":"thinking","thinking":"","signature":""}}"#,
        "\n\n",
        "event: content_block_delta\n",
        r#"data: {"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"Need the weather tool."}}"#,
        "\n\n",
        "event: content_block_stop\n",
        r#"data: {"type":"content_block_stop","index":0}"#,
        "\n\n",
        "event: content_block_start\n",
        r#"data: {"type":"content_block_start","index":1,"content_block":{"type":"text","text":""}}"#,
        "\n\n",
        "event: ping\n",
        r#"data: {"type": "ping"}"#,
        "\n\n",
        "event: content_block_delta\n",
        r#"data: {"type":"content_block_delta","index":1,"delta":{"type":"text_delta","text":"Hel"}}"#,
        "\n\n",
        "event: content_block_delta\n",
        r#"data: {"type":"content_block_delta","index":1,"delta":{"type":"text_delta","text":"lo ☀"}}"#,
        "\n\n",
        "event: content_block_stop\n",
        r#"data: {"type":"content_block_stop","index":1}"#,
        "\n\n",
        "event: content_block_start\n",
        r#"data: {"type":"content_block_start","index":2,"content_block":{"type":"tool_use","id":"toolu_1","name":"get_weather","input":{}}}"#,
        "\n\n",
        "event: content_block_delta\n",
        r#"data: {"type":"content_block_delta","index":2,"delta":{"type":"input_json_delta","partial_json":"{\"city\": "}}"#,
        "\n\n",
        "event: content_block_delta\n",
        r#"data: {"type":"content_block_delta","index":2,"delta":{"type":"input_json_delta","partial_json":"\"Paris\"}"}}"#,
        "\n\n",
        "event: content_block_stop\n",
        r#"data: {"type":"content_block_stop","index":2}"#,
        "\n\n",
        "event: message_delta\n",
        r#"data: {"type":"message_delta","delta":{"stop_reason":"tool_use","stop_sequence":null},"usage":{"output_tokens":42}}"#,
        "\n\n",
        "event: message_stop\n",
        r#"data: {"type":"message_stop"}"#,
        "\n\n",
    );

    /// An upstream Anthropic streaming reply that starts normally, then fails
    /// with an `error` event — the API's way of reporting e.g. overload after
    /// it has already answered `200`.
    const ANTHROPIC_ERROR_SSE: &str = concat!(
        "event: message_start\n",
        r#"data: {"type":"message_start","message":{"id":"msg_up","type":"message","role":"assistant","content":[],"model":"claude-opus-4-8","usage":{"input_tokens":25,"output_tokens":1}}}"#,
        "\n\n",
        "event: content_block_start\n",
        r#"data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}"#,
        "\n\n",
        "event: content_block_delta\n",
        r#"data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hel"}}"#,
        "\n\n",
        "event: error\n",
        r#"data: {"type":"error","error":{"type":"overloaded_error","message":"Overloaded"}}"#,
        "\n\n",
    );

    /// Env var the test provider's `api_key_env` names; see
    /// [`spawn_router_to_anthropic`].
    const TEST_ANTHROPIC_KEY_ENV: &str = "OSLR_TEST_ANTHROPIC_STREAM_KEY";

    /// Headers and JSON body of every request a mock upstream received.
    type Captured = Arc<std::sync::Mutex<Vec<(HeaderMap, Value)>>>;

    /// Binds `app` to a random localhost port and serves it in the background.
    async fn serve_on_random_port(app: AxumRouter) -> std::net::SocketAddr {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        addr
    }

    /// A stand-in for the Anthropic API that answers every `POST
    /// /v1/messages` with `sse`, in small chunks so lines (and the multi-byte
    /// `☀`) can straddle reads.
    async fn spawn_mock_anthropic(sse: &'static str) -> (std::net::SocketAddr, Captured) {
        let captured = Captured::default();
        let captured_by_handler = captured.clone();
        let app = AxumRouter::new().route(
            "/v1/messages",
            post(move |headers: HeaderMap, Json(body): Json<Value>| {
                let captured = captured_by_handler.clone();
                async move {
                    captured.lock().unwrap().push((headers, body));
                    let chunks: Vec<Result<bytes::Bytes, Infallible>> =
                        sse.as_bytes().chunks(16).map(|c| Ok(bytes::Bytes::copy_from_slice(c))).collect();
                    Response::builder()
                        .header(header::CONTENT_TYPE, "text/event-stream")
                        .body(Body::from_stream(tokio_stream::iter(chunks)))
                        .unwrap()
                }
            }),
        );
        (serve_on_random_port(app).await, captured)
    }

    /// Starts the full router app with a single `format = "anthropic"`
    /// provider pointed at `upstream`, behind a fallback rule.
    async fn spawn_router_to_anthropic(upstream: std::net::SocketAddr) -> std::net::SocketAddr {
        // SAFETY: edition 2024 makes `set_var` unsafe because another thread
        // could read the environment mid-write. Only these tests touch this
        // variable, always writing the same value, and std serializes its
        // own `env::var`/`set_var` calls against each other.
        unsafe { std::env::set_var(TEST_ANTHROPIC_KEY_ENV, "test-key") };

        let config: crate::config::Config = toml::from_str(&format!(
            r#"
            [[providers]]
            name = "claude"
            format = "anthropic"
            base_url = "http://{upstream}/v1"
            api_key_env = "{TEST_ANTHROPIC_KEY_ENV}"

            [[routers]]
            type = "fallback"
            "#
        ))
        .unwrap();

        let (events, _) = broadcast::channel(16);
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
        serve_on_random_port(build_app(state, false)).await
    }

    /// Splits an Anthropic-style SSE body into `(event_name, data_json)`
    /// pairs, one per `event: ...\ndata: ...\n\n` block.
    fn parse_named_sse(body: &str) -> Vec<(String, Value)> {
        body.split("\n\n")
            .filter(|block| !block.is_empty())
            .map(|block| {
                let mut lines = block.lines();
                let event_name = lines.next().unwrap().trim_start_matches("event: ").to_string();
                let data = lines.next().unwrap().trim_start_matches("data: ");
                (event_name, serde_json::from_str(data).unwrap())
            })
            .collect()
    }

    /// Every `data:` payload of an OpenAI-style SSE body, as raw strings
    /// (so the closing `[DONE]` sentinel is kept).
    fn parse_data_sse(body: &str) -> Vec<String> {
        body.lines().filter_map(|l| l.strip_prefix("data: ")).map(str::to_string).collect()
    }

    #[tokio::test]
    async fn anthropic_provider_streams_end_to_end_to_every_client_format() {
        let (upstream, captured) = spawn_mock_anthropic(ANTHROPIC_TOOL_USE_SSE).await;
        let router = spawn_router_to_anthropic(upstream).await;
        let client = reqwest::Client::new();

        // An Anthropic client (e.g. Claude Code) on /v1/messages. The task
        // budget is only here to check the beta header reaches the upstream.
        let resp = client
            .post(format!("http://{router}/v1/messages"))
            .json(&json!({
                "model": "claude-opus-4-8",
                "max_tokens": 1024,
                "stream": true,
                "messages": [{"role": "user", "content": "Weather in Paris?"}],
                "tools": [{"name": "get_weather", "input_schema": {"type": "object"}}],
                "output_config": {"task_budget": {"type": "tokens", "total": 64000}},
            }))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        assert_eq!(resp.headers()[header::CONTENT_TYPE], "text/event-stream");
        let events = parse_named_sse(&resp.text().await.unwrap());

        let names: Vec<&str> = events.iter().map(|(n, _)| n.as_str()).collect();
        assert_eq!(
            names,
            vec![
                "message_start",
                "content_block_start",
                "content_block_delta",
                "content_block_delta",
                "content_block_stop",
                "content_block_start",
                "content_block_delta",
                "content_block_delta",
                "content_block_stop",
                "message_delta",
                "message_stop",
            ]
        );
        assert_eq!(events[1].1["content_block"]["type"], "text");
        assert_eq!(events[2].1["delta"]["text"], "Hel");
        assert_eq!(events[3].1["delta"]["text"], "lo ☀");
        assert_eq!(
            events[5].1["content_block"],
            json!({"type": "tool_use", "id": "toolu_1", "name": "get_weather", "input": {}})
        );
        let partial_json: String =
            events[6..8].iter().map(|(_, e)| e["delta"]["partial_json"].as_str().unwrap()).collect();
        assert_eq!(partial_json, r#"{"city": "Paris"}"#);
        assert_eq!(events[9].1["delta"]["stop_reason"], "tool_use");
        assert_eq!(events[9].1["usage"], json!({"input_tokens": 25, "output_tokens": 42}));

        // An OpenAI client (e.g. Copilot CLI) on /v1/chat/completions, routed
        // to the same Anthropic upstream.
        let resp = client
            .post(format!("http://{router}/v1/chat/completions"))
            .json(&json!({
                "model": "claude-opus-4-8",
                "stream": true,
                "messages": [{"role": "user", "content": "Weather in Paris?"}],
            }))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let data = parse_data_sse(&resp.text().await.unwrap());

        assert_eq!(data.last().map(String::as_str), Some("[DONE]"));
        let chunks: Vec<Value> = data[..data.len() - 1].iter().map(|d| serde_json::from_str(d).unwrap()).collect();
        let text: String = chunks.iter().filter_map(|c| c["choices"][0]["delta"]["content"].as_str()).collect();
        assert_eq!(text, "Hello ☀");
        let tool_deltas: Vec<&Value> =
            chunks.iter().filter_map(|c| c["choices"][0]["delta"]["tool_calls"].get(0)).collect();
        assert_eq!(tool_deltas[0]["id"], "toolu_1");
        assert_eq!(tool_deltas[0]["function"]["name"], "get_weather");
        let arguments: String = tool_deltas.iter().map(|t| t["function"]["arguments"].as_str().unwrap()).collect();
        assert_eq!(arguments, r#"{"city": "Paris"}"#);
        assert_eq!(chunks.last().unwrap()["choices"][0]["finish_reason"], "tool_calls");

        // A Responses API client (e.g. Codex) on /v1/responses.
        let resp = client
            .post(format!("http://{router}/v1/responses"))
            .json(&json!({
                "model": "claude-opus-4-8",
                "stream": true,
                "input": [{"type": "message", "role": "user", "content": [{"type": "input_text", "text": "Weather in Paris?"}]}],
            }))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let events = parse_named_sse(&resp.text().await.unwrap());

        let text: String = events
            .iter()
            .filter(|(n, _)| n == "response.output_text.delta")
            .map(|(_, e)| e["delta"].as_str().unwrap())
            .collect();
        assert_eq!(text, "Hello ☀");
        let call = events
            .iter()
            .find(|(n, e)| n == "response.output_item.added" && e["item"]["type"] == "function_call")
            .map(|(_, e)| &e["item"])
            .unwrap();
        assert_eq!(call["call_id"], "toolu_1");
        assert_eq!(call["name"], "get_weather");
        let arguments: String = events
            .iter()
            .filter(|(n, _)| n == "response.function_call_arguments.delta")
            .map(|(_, e)| e["delta"].as_str().unwrap())
            .collect();
        assert_eq!(arguments, r#"{"city": "Paris"}"#);
        let (last_name, completed) = events.last().unwrap();
        assert_eq!(last_name, "response.completed");
        assert_eq!(completed["response"]["usage"]["output_tokens"], 42);

        // All three requests reached the upstream as Anthropic streaming calls.
        let captured = captured.lock().unwrap();
        assert_eq!(captured.len(), 3);
        for (headers, body) in captured.iter() {
            assert_eq!(headers["anthropic-version"], "2023-06-01");
            assert_eq!(headers["x-api-key"], "test-key");
            assert_eq!(body["stream"], true);
            assert_eq!(body["model"], "claude-opus-4-8");
        }
        assert_eq!(captured[0].0["anthropic-beta"], "task-budgets-2026-03-13");
        assert_eq!(captured[0].1["tools"][0]["name"], "get_weather");
        assert!(captured[1].0.get("anthropic-beta").is_none());
        assert!(captured[2].0.get("anthropic-beta").is_none());
    }

    #[tokio::test]
    async fn anthropic_upstream_error_event_aborts_client_stream_instead_of_completing_it() {
        let (upstream, _) = spawn_mock_anthropic(ANTHROPIC_ERROR_SSE).await;
        let router = spawn_router_to_anthropic(upstream).await;

        let resp = reqwest::Client::new()
            .post(format!("http://{router}/v1/messages"))
            .json(&json!({
                "model": "claude-opus-4-8",
                "max_tokens": 1024,
                "stream": true,
                "messages": [{"role": "user", "content": "hi"}],
            }))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        let mut body = resp.bytes_stream();
        let mut received = String::new();
        let mut errored = false;
        while let Some(chunk) = body.next().await {
            match chunk {
                Ok(bytes) => received.push_str(&String::from_utf8_lossy(&bytes)),
                Err(_) => {
                    errored = true;
                    break;
                }
            }
        }

        assert!(errored, "client stream should fail, got: {received}");
        assert!(received.contains("Hel"), "text before the error should still arrive, got: {received}");
        assert!(!received.contains("message_stop"), "stream must not look complete, got: {received}");
    }

    /// A stand-in for the Anthropic API that answers every `POST
    /// /v1/messages` with the non-streaming reply `body`.
    async fn spawn_mock_anthropic_json(body: Value) -> (std::net::SocketAddr, Captured) {
        let captured = Captured::default();
        let captured_by_handler = captured.clone();
        let app = AxumRouter::new().route(
            "/v1/messages",
            post(move |headers: HeaderMap, Json(request): Json<Value>| {
                let captured = captured_by_handler.clone();
                let body = body.clone();
                async move {
                    captured.lock().unwrap().push((headers, request));
                    Json(body)
                }
            }),
        );
        (serve_on_random_port(app).await, captured)
    }

    #[tokio::test]
    async fn anthropic_provider_non_streaming_reply_with_thinking_block_is_not_a_502() {
        // What an adaptive-thinking model returns when it thinks and the
        // display is omitted.
        let (upstream, captured) = spawn_mock_anthropic_json(json!({
            "id": "msg_up", "type": "message", "role": "assistant", "model": "claude-opus-5-5",
            "content": [
                {"type": "thinking", "thinking": "", "signature": "EqQB"},
                {"type": "text", "text": "hi"},
            ],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 7, "output_tokens": 11},
        }))
        .await;
        let router = spawn_router_to_anthropic(upstream).await;
        let client = reqwest::Client::new();

        // An Anthropic client that kept the upstream's thinking block and
        // replays it on the next turn.
        let resp = client
            .post(format!("http://{router}/v1/messages"))
            .json(&json!({
                "model": "claude-opus-5-5",
                "max_tokens": 1024,
                "thinking": {"type": "adaptive"},
                "messages": [
                    {"role": "user", "content": "first"},
                    {"role": "assistant", "content": [
                        {"type": "thinking", "thinking": "", "signature": "EqQB"},
                        {"type": "text", "text": "earlier answer"},
                    ]},
                    {"role": "user", "content": "second"},
                ],
            }))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let body: Value = resp.json().await.unwrap();
        assert_eq!(body["content"], json!([{"type": "text", "text": "hi"}]));
        assert_eq!(body["stop_reason"], "end_turn");
        assert_eq!(body["usage"], json!({"input_tokens": 7, "output_tokens": 11}));

        // And through an OpenAI client.
        let resp = client
            .post(format!("http://{router}/v1/chat/completions"))
            .json(&json!({"model": "claude-opus-5-5", "messages": [{"role": "user", "content": "hi"}]}))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let body: Value = resp.json().await.unwrap();
        assert_eq!(body["choices"][0]["message"]["content"], "hi");

        // The replayed thinking block never went upstream; the rest of the
        // history did.
        let captured = captured.lock().unwrap();
        assert_eq!(captured[0].1["thinking"], json!({"type": "adaptive"}));
        assert_eq!(captured[0].1["messages"][1], json!({"role": "assistant", "content": "earlier answer"}));
    }
}
