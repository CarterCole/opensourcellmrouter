//! Request/response middleware, modeled after OpenRouter's `plugins` array.
//!
//! Clients opt into a plugin by name in the request body:
//!
//! ```json
//! {"model": "...", "messages": [...], "plugins": [{"id": "response-healing"}]}
//! ```
//!
//! The server can also enable plugins by default (and supply default
//! settings) via `[plugins.<id>]` in the config file. At request time,
//! [`PluginRegistry::resolve`] merges config defaults with any per-request
//! overrides into an ordered list of `(plugin, settings)` pairs, which
//! [`crate::server::dispatch`] runs as `pre_request`/`post_response` hooks
//! around the call to the chosen provider.

pub mod pareto_router;
pub mod pdf_input;
pub mod response_healing;
pub mod web_search;

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use async_trait::async_trait;
use serde_json::{Map, Value};

use crate::canonical::{ChatRequest, ChatResponse};
use crate::config::Config;

/// Resources and per-request settings handed to a plugin's hooks.
pub struct PluginContext {
    pub client: reqwest::Client,
    /// This plugin's settings: config defaults merged with (and overridden
    /// by) any matching entry in the request's `plugins` array.
    pub settings: Map<String, Value>,
}

impl PluginContext {
    pub fn get_str<'a>(&'a self, key: &str) -> Option<&'a str> {
        self.settings.get(key).and_then(Value::as_str)
    }
}

/// A point in the request pipeline where plugin hooks run:
///
/// ```text
/// Start -> classifiers -> PreRouting -> routers -> provider -> PostResponse -> End -> logging
/// ```
///
/// See [`crate::server::dispatch`] for the full pipeline.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Stage {
    /// Before classifiers run, on the request as the client sent it.
    Start,
    /// After classifiers, before the `routers` chain picks a provider.
    PreRouting,
    /// After a response is available — either from the provider, or because
    /// an earlier hook produced one directly.
    PostResponse,
    /// The last stage before the response is logged and returned.
    End,
}

/// Returned by a hook to control whether later hooks/stages run.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Flow {
    /// Fall through to the next hook, and eventually the next pipeline
    /// stage.
    Continue,
    /// Stop running further hooks.
    ///
    /// For [`Stage::Start`] and [`Stage::PreRouting`], this also skips
    /// classifiers/routing/the provider call entirely — the hook must have
    /// written a response into `resp` itself, or the request ends with no
    /// response and fails.
    ///
    /// For [`Stage::PostResponse`], this also skips [`Stage::End`].
    Stop,
}

#[async_trait]
pub trait Plugin: Send + Sync {
    /// Identifier used in `[plugins.<id>]` config and in a request's
    /// `plugins` array (`{"id": "<id>", ...}`).
    fn id(&self) -> &'static str;

    /// Runs first, before classifiers, on the request as the client sent
    /// it. May mutate `req`, or write a response into `*resp` and return
    /// [`Flow::Stop`] to answer the request without calling a provider at
    /// all (e.g. a moderation plugin that rejects the request outright).
    async fn on_start(
        &self,
        _ctx: &PluginContext,
        _req: &mut ChatRequest,
        _resp: &mut Option<ChatResponse>,
    ) -> anyhow::Result<Flow> {
        Ok(Flow::Continue)
    }

    /// Runs after classifiers, before routing. May mutate `req` — e.g. to
    /// inject context, or set `req.forced_provider` to bypass the `routers`
    /// chain — or write a response into `*resp` and return [`Flow::Stop`]
    /// to skip routing and the provider call.
    async fn pre_request(
        &self,
        _ctx: &PluginContext,
        _req: &mut ChatRequest,
        _resp: &mut Option<ChatResponse>,
    ) -> anyhow::Result<Flow> {
        Ok(Flow::Continue)
    }

    /// Runs once a response is available. May mutate `*resp`, e.g. to
    /// repair malformed JSON. An error here is logged and ignored rather
    /// than failing the request.
    async fn post_response(
        &self,
        _ctx: &PluginContext,
        _req: &ChatRequest,
        _resp: &mut Option<ChatResponse>,
    ) -> anyhow::Result<Flow> {
        Ok(Flow::Continue)
    }

    /// Runs last, just before the response is logged and returned to the
    /// client. An error here is logged and ignored rather than failing the
    /// request.
    async fn on_end(
        &self,
        _ctx: &PluginContext,
        _req: &ChatRequest,
        _resp: &mut Option<ChatResponse>,
    ) -> anyhow::Result<Flow> {
        Ok(Flow::Continue)
    }
}

struct PluginEntry {
    plugin: Arc<dyn Plugin>,
    enabled_by_default: bool,
    default_settings: Map<String, Value>,
}

pub struct PluginRegistry {
    /// All known plugins, in a fixed order (config-enabled ones run in this
    /// order; request-only ones are appended in the order the client listed
    /// them).
    entries: Vec<PluginEntry>,
    by_id: HashMap<&'static str, usize>,
}

impl PluginRegistry {
    pub fn from_config(config: &Config) -> Self {
        let plugins: Vec<Arc<dyn Plugin>> = vec![
            Arc::new(response_healing::ResponseHealingPlugin),
            Arc::new(pareto_router::ParetoRouterPlugin),
            Arc::new(web_search::WebSearchPlugin),
            Arc::new(pdf_input::PdfInputPlugin),
        ];

        let mut entries = Vec::with_capacity(plugins.len());
        let mut by_id = HashMap::with_capacity(plugins.len());
        for plugin in plugins {
            let id = plugin.id();
            let cfg = config.plugins.get(id);
            by_id.insert(id, entries.len());
            entries.push(PluginEntry {
                plugin,
                enabled_by_default: cfg.is_some_and(|c| c.enabled),
                default_settings: cfg.map(|c| c.settings.clone()).unwrap_or_default(),
            });
        }

        PluginRegistry { entries, by_id }
    }

    /// Resolve the ordered list of `(plugin, settings)` pairs to run for
    /// `req`: every plugin enabled by default in config, in registry order,
    /// followed by any plugins named in `req.plugins` that weren't already
    /// included. A plugin's settings are its config defaults overridden by
    /// the matching `req.plugins` entry (if any).
    pub fn resolve(&self, req: &ChatRequest) -> Vec<(Arc<dyn Plugin>, Map<String, Value>)> {
        let mut result = Vec::new();
        let mut included = HashSet::new();

        for entry in &self.entries {
            if entry.enabled_by_default {
                let mut settings = entry.default_settings.clone();
                if let Some(req_entry) = req.plugins.iter().find(|p| p.id == entry.plugin.id()) {
                    settings.extend(req_entry.settings.clone());
                }
                included.insert(entry.plugin.id());
                result.push((entry.plugin.clone(), settings));
            }
        }

        for req_entry in &req.plugins {
            if included.contains(req_entry.id.as_str()) {
                continue;
            }
            let Some(&idx) = self.by_id.get(req_entry.id.as_str()) else {
                tracing::warn!("ignoring unknown plugin id '{}'", req_entry.id);
                continue;
            };
            let entry = &self.entries[idx];
            let mut settings = entry.default_settings.clone();
            settings.extend(req_entry.settings.clone());
            included.insert(entry.plugin.id());
            result.push((entry.plugin.clone(), settings));
        }

        result
    }
}
