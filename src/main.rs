mod canonical;
mod classifiers;
mod config;
mod formats;
mod logging;
mod plugins;
mod provider;
mod router;
mod server;
mod tui;
mod watch;

use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::AtomicU64;

use anyhow::Context;
use opentelemetry::trace::TracerProvider as _;
use opentelemetry_otlp::WithExportConfig;
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let mut args = std::env::args().skip(1).peekable();
    let config_path = match args.peek().map(String::as_str) {
        Some("watch") => {
            args.next();
            let url = args.next().unwrap_or_else(|| "http://localhost:8090".to_string());
            return watch::run(&url).await;
        }
        Some("tui") => {
            args.next();
            let url = args.next().unwrap_or_else(|| "http://localhost:8090".to_string());
            return tui::run(&url).await;
        }
        _ => args.next().unwrap_or_else(|| "config.toml".to_string()),
    };

    let config = config::Config::load(&PathBuf::from(&config_path))
        .with_context(|| format!("loading config from {config_path}"))?;

    let otel_state = init_telemetry(&config.telemetry)?;

    let addr = format!("{}:{}", config.server.host, config.server.port);
    let client = reqwest::Client::new();

    let mut model_router = router::ModelRouter::from_config(&config)?;
    model_router.discover_models(&client).await;
    let model_router = Arc::new(model_router);

    let logger = if config.logging.enabled {
        Some(Arc::new(
            logging::RequestLogger::new(&config.logging.path)
                .with_context(|| format!("opening log file {}", config.logging.path))?,
        ))
    } else {
        None
    };

    let api_key = config
        .server
        .api_key_env
        .as_ref()
        .map(|var| {
            std::env::var(var)
                .with_context(|| format!("[server] api_key_env = \"{var}\" but that variable is not set"))
        })
        .transpose()?;

    let plugin_registry = Arc::new(plugins::PluginRegistry::from_config(&config));
    let classifier_registry = Arc::new(classifiers::ClassifierRegistry::from_config(&config));
    let response_classifier_registry = Arc::new(classifiers::ResponseClassifierRegistry::from_config(&config));

    let (events, _) = tokio::sync::broadcast::channel(256);

    let app = server::build_app(
        server::AppState {
            router: model_router,
            client,
            logger,
            plugins: plugin_registry,
            classifiers: classifier_registry,
            response_classifiers: response_classifier_registry,
            events,
            in_flight: Arc::new(AtomicU64::new(0)),
            next_id: Arc::new(AtomicU64::new(0)),
            api_key,
        },
        config.server.dashboard,
    );

    let listener = tokio::net::TcpListener::bind(&addr)
        .await
        .with_context(|| format!("binding to {addr}"))?;
    tracing::info!("opensourcellmrouter listening on {addr}");
    axum::serve(listener, app).await?;

    if let Some(otel_state) = otel_state {
        otel_state
            .tracer_provider
            .shutdown()
            .context("shutting down OTel tracer provider")?;
        otel_state
            .meter_provider
            .shutdown()
            .context("shutting down OTel meter provider")?;
        otel_state
            .logger_provider
            .shutdown()
            .context("shutting down OTel logger provider")?;
    }

    Ok(())
}

/// Handles to the OTel providers, kept alive for the lifetime of `main` so
/// their batch exporters can flush on shutdown.
struct OtelState {
    tracer_provider: opentelemetry_sdk::trace::SdkTracerProvider,
    meter_provider: opentelemetry_sdk::metrics::SdkMeterProvider,
    logger_provider: opentelemetry_sdk::logs::SdkLoggerProvider,
}

/// Initializes the global `tracing` subscriber. When `telemetry.enabled` is
/// true, composes the plain stdout `fmt` layer with OTLP-exporting trace,
/// metric, and log layers; otherwise the subscriber behaves exactly as it
/// did before telemetry support existed.
fn init_telemetry(telemetry: &config::TelemetryConfig) -> anyhow::Result<Option<OtelState>> {
    if !telemetry.enabled {
        tracing_subscriber::registry()
            .with(tracing_subscriber::EnvFilter::from_default_env())
            .with(tracing_subscriber::fmt::layer())
            .init();
        return Ok(None);
    }

    let resource = opentelemetry_sdk::Resource::builder()
        .with_service_name(telemetry.service_name.clone())
        .build();

    let span_exporter = opentelemetry_otlp::SpanExporter::builder()
        .with_http()
        .with_endpoint(&telemetry.otlp_endpoint)
        .build()
        .context("building OTLP span exporter")?;
    let tracer_provider = opentelemetry_sdk::trace::SdkTracerProvider::builder()
        .with_batch_exporter(span_exporter)
        .with_resource(resource.clone())
        .with_sampler(opentelemetry_sdk::trace::Sampler::TraceIdRatioBased(
            telemetry.sample_ratio,
        ))
        .build();
    let tracer = tracer_provider.tracer("opensourcellmrouter");

    let metric_exporter = opentelemetry_otlp::MetricExporter::builder()
        .with_http()
        .with_endpoint(&telemetry.otlp_endpoint)
        .build()
        .context("building OTLP metric exporter")?;
    let meter_provider = opentelemetry_sdk::metrics::SdkMeterProvider::builder()
        .with_periodic_exporter(metric_exporter)
        .with_resource(resource.clone())
        .build();
    opentelemetry::global::set_meter_provider(meter_provider.clone());

    let log_exporter = opentelemetry_otlp::LogExporter::builder()
        .with_http()
        .with_endpoint(&telemetry.otlp_endpoint)
        .build()
        .context("building OTLP log exporter")?;
    let logger_provider = opentelemetry_sdk::logs::SdkLoggerProvider::builder()
        .with_batch_exporter(log_exporter)
        .with_resource(resource)
        .build();
    let log_bridge = opentelemetry_appender_tracing::layer::OpenTelemetryTracingBridge::new(&logger_provider);

    tracing_subscriber::registry()
        .with(tracing_subscriber::EnvFilter::from_default_env())
        .with(tracing_subscriber::fmt::layer())
        .with(tracing_opentelemetry::layer().with_tracer(tracer))
        .with(log_bridge)
        .init();

    Ok(Some(OtelState {
        tracer_provider,
        meter_provider,
        logger_provider,
    }))
}
