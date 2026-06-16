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

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();

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

    let plugin_registry = Arc::new(plugins::PluginRegistry::from_config(&config));
    let classifier_registry = Arc::new(classifiers::ClassifierRegistry::from_config(&config));

    let (events, _) = tokio::sync::broadcast::channel(256);

    let app = server::build_app(
        server::AppState {
            router: model_router,
            client,
            logger,
            plugins: plugin_registry,
            classifiers: classifier_registry,
            events,
            in_flight: Arc::new(AtomicU64::new(0)),
            next_id: Arc::new(AtomicU64::new(0)),
        },
        config.server.dashboard,
    );

    let listener = tokio::net::TcpListener::bind(&addr)
        .await
        .with_context(|| format!("binding to {addr}"))?;
    tracing::info!("opensourcellmrouter listening on {addr}");
    axum::serve(listener, app).await?;

    Ok(())
}
