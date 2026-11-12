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

use crate::canonical::{ChatRequest, ChatResponse, PluginRequest};
use crate::config::Config;

/// Resources and per-request settings handed to a plugin's hooks.
pub struct PluginContext {
    #[allow(dead_code)]
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
    /// Hook ran but made no changes; fall through to the next hook.
    Continue,
    /// Hook actively mutated `req` or `resp`; fall through to the next hook.
    /// This is the signal used to record that a plugin did real work.
    Modified,
    /// Stop running further hooks.
    ///
    /// For [`Stage::Start`] and [`Stage::PreRouting`], this also skips
    /// classifiers/routing/the provider call entirely — the hook must have
    /// written a response into `resp` itself, or the request ends with no
    /// response and fails.
    ///
    /// For [`Stage::PostResponse`], this also skips [`Stage::End`].
    #[allow(dead_code)]
    Stop,
}

#[async_trait]
pub trait Plugin: Send + Sync {
    /// Identifier used in `[plugins.<id>]` config and in a request's
    /// `plugins` array (`{"id": "<id>", ...}`).
    fn id(&self) -> &'static str;

    /// Settings a request's `plugins` entry may set, overriding config —
    /// e.g. `pareto-router`'s `tier`. Any other key in the entry is ignored:
    /// settings are the server's to choose unless a plugin says otherwise,
    /// since some (like `pareto-router`'s `tiers`) would let a client route
    /// around the `routers` chain.
    fn request_settings(&self) -> &'static [&'static str] {
        &[]
    }

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
    /// the matching `req.plugins` entry (if any), in the settings the plugin
    /// lets a request set (see [`Plugin::request_settings`]).
    pub fn resolve(&self, req: &ChatRequest) -> Vec<(Arc<dyn Plugin>, Map<String, Value>)> {
        let mut result = Vec::new();
        let mut included = HashSet::new();

        for entry in &self.entries {
            if entry.enabled_by_default {
                let mut settings = entry.default_settings.clone();
                if let Some(req_entry) = req.plugins.iter().find(|p| p.id == entry.plugin.id()) {
                    apply_request_settings(&mut settings, entry.plugin.as_ref(), req_entry);
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
            apply_request_settings(&mut settings, entry.plugin.as_ref(), req_entry);
            included.insert(entry.plugin.id());
            result.push((entry.plugin.clone(), settings));
        }

        result
    }
}

/// Overrides `settings` with the ones in `req_entry` that `plugin` lets a
/// request set, ignoring (with a warning) any others.
fn apply_request_settings(settings: &mut Map<String, Value>, plugin: &dyn Plugin, req_entry: &PluginRequest) {
    for (key, value) in &req_entry.settings {
        if plugin.request_settings().contains(&key.as_str()) {
            settings.insert(key.clone(), value.clone());
        } else {
            tracing::warn!("ignoring setting '{key}' for plugin '{}': only config sets it", plugin.id());
        }
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    fn request(plugins: Value) -> ChatRequest {
        let mut req: ChatRequest = serde_json::from_value(json!({
            "model": "gpt-expensive",
            "system": null,
            "messages": [],
            "max_tokens": null,
            "temperature": null,
            "thinking": null,
            "effort": null,
            "task_budget": null,
            "output_schema": null,
            "stream": false,
        }))
        .unwrap();
        req.plugins = serde_json::from_value(plugins).unwrap();
        req
    }

    fn registry(config: &str) -> PluginRegistry {
        let config: Config = toml::from_str(&format!("providers = []\n{config}")).unwrap();
        PluginRegistry::from_config(&config)
    }

    fn settings_of(registry: &PluginRegistry, req: &ChatRequest, id: &str) -> Map<String, Value> {
        registry.resolve(req).into_iter().find(|(p, _)| p.id() == id).map(|(_, s)| s).unwrap()
    }

    #[test]
    fn a_request_picks_a_tier_but_not_what_is_in_it() {
        let registry = registry(
            r#"
            [plugins.pareto-router]
            enabled = true
            default_tier = "low"

            [plugins.pareto-router.tiers]
            low = ["cheap"]
            "#,
        );
        let req = request(json!([{"id": "pareto-router", "tier": "x", "tiers": {"x": ["premium"]}, "default_tier": "x"}]));

        let settings = settings_of(&registry, &req, "pareto-router");
        assert_eq!(settings["tier"], "x");
        assert_eq!(settings["default_tier"], "low");
        assert_eq!(settings["tiers"], json!({"low": ["cheap"]}));
    }

    #[test]
    fn a_plugin_only_a_request_enables_gets_no_settings_from_it_but_the_allowed_ones() {
        let registry = registry("");
        let req = request(json!([{"id": "pareto-router", "tier": "x", "tiers": {"x": ["premium"]}}]));

        let settings = settings_of(&registry, &req, "pareto-router");
        assert_eq!(settings.get("tiers"), None);
        assert_eq!(settings["tier"], "x");
    }
}
