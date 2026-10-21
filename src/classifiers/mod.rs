//! Tags a request before it's routed, so [`crate::router::ModelRouter`] can
//! make capability- or policy-based decisions (e.g. send "vision" requests
//! to a multimodal model, or "nsfw" requests to a moderation provider).
//!
//! Classifiers run first in the pipeline:
//!
//! ```text
//! prompt -> classifiers -> routing -> model -> logging
//! ```
//!
//! Every classifier enabled in config (via `[classifiers.<id>]`) runs, in
//! registry order, on every request. Each produces zero or more tags, which
//! are merged (de-duplicated) into [`ChatRequest::tags`](crate::canonical::ChatRequest::tags).
//! `routers` rules — in particular [`crate::config::RouterRule::Tag`] — can
//! then match on those tags. A classifier can also pick the model to route
//! to outright (see [`cev`] and [`llm`]), for a
//! [`crate::config::RouterRule::Cev`] or [`crate::config::RouterRule::Llm`]
//! rule to act on. A classifier that errors is logged and skipped; it never
//! fails the request.

pub mod cev;
pub mod keyword;
pub mod llm;
pub mod refusal;

use std::collections::HashSet;
use std::sync::Arc;

use anyhow::{Context, bail};
use async_trait::async_trait;
use serde::Deserialize;
use serde_json::{Map, Value};

use crate::canonical::{ChatRequest, ChatResponse, Role, RouteChoice};
use crate::config::{Config, ProviderConfig, RouterRule};

/// Per-classifier settings, from `[classifiers.<id>]` in config.
pub struct ClassifierContext {
    pub settings: Map<String, Value>,
}

/// What a classifier concluded about a request.
#[derive(Debug, Default)]
pub struct Classification {
    /// Tags the request should be labeled with, e.g. `["vision"]`.
    pub tags: Vec<String>,
    /// The model that should answer the request, if the classifier picks
    /// one.
    pub route: Option<RouteChoice>,
}

impl From<Vec<String>> for Classification {
    fn from(tags: Vec<String>) -> Self {
        Classification { tags, route: None }
    }
}

#[async_trait]
pub trait Classifier: Send + Sync {
    /// Identifier used in `[classifiers.<id>]` config.
    fn id(&self) -> &'static str;

    /// Inspects `req` (before routing) and returns the tags it should be
    /// labeled with, e.g. `["vision"]`, and/or the model it should be
    /// routed to. Returning neither means "no opinion".
    async fn classify(&self, ctx: &ClassifierContext, req: &ChatRequest) -> anyhow::Result<Classification>;
}

/// One model a routing classifier ([`cev`] or [`llm`]) can pick, from its
/// `[[classifiers.<id>.models]]` config.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModelOption {
    /// What the option is picked and labeled by; defaults to `model`.
    pub name: Option<String>,
    /// What the model is good at, shown to the classifier next to the name.
    pub description: Option<String>,
    pub provider: String,
    pub model: String,
}

impl ModelOption {
    pub fn name(&self) -> &str {
        self.name.as_deref().unwrap_or(&self.model)
    }
}

/// Checks that every option names one of `providers`, and that no two
/// share a name.
fn check_models(models: &[ModelOption], providers: &[ProviderConfig]) -> anyhow::Result<()> {
    let mut names = HashSet::new();
    for option in models {
        if !providers.iter().any(|p| p.name == option.provider) {
            bail!("model '{}' names unknown provider '{}'", option.name(), option.provider);
        }
        if !names.insert(option.name()) {
            bail!("two models are named '{}'; give one of them a different `name`", option.name());
        }
    }
    Ok(())
}

/// Text of the latest user message that has any — skipping turns that only
/// carry tool results, so an agent's tool loop is classified by what the
/// user last asked for.
fn latest_user_text(req: &ChatRequest) -> String {
    req.messages
        .iter()
        .rev()
        .filter(|m| m.role == Role::User)
        .map(|m| m.text_content())
        .find(|text| !text.is_empty())
        .unwrap_or_default()
}

/// The first `max` characters of `s`.
fn clip(s: &str, max: usize) -> &str {
    match s.char_indices().nth(max) {
        Some((end, _)) => &s[..end],
        None => s,
    }
}

struct ClassifierEntry {
    classifier: Arc<dyn Classifier>,
    enabled: bool,
    settings: Map<String, Value>,
}

pub struct ClassifierRegistry {
    /// All known classifiers, in a fixed order; enabled ones run in this
    /// order and their tags are merged in the same order.
    entries: Vec<ClassifierEntry>,
    /// The `cev` classifier, if enabled, for sending it feedback.
    cev: Option<Arc<cev::CevClassifier>>,
}

impl ClassifierRegistry {
    pub fn from_config(config: &Config) -> anyhow::Result<Self> {
        let mut classifiers: Vec<Arc<dyn Classifier>> = vec![Arc::new(keyword::KeywordClassifier)];

        // Unlike the classifiers above, `cev` holds a connection (or a
        // loaded model) built from its settings, so it only exists when
        // it's enabled.
        let cev = match config.classifiers.get("cev").filter(|c| c.enabled) {
            Some(cfg) => {
                let classifier = cev::CevClassifier::from_settings(&cfg.settings, &config.providers)
                    .context("configuring classifier 'cev'")?;
                Some(Arc::new(classifier))
            }
            None => None,
        };
        if let Some(cev) = &cev {
            classifiers.push(cev.clone());
        }

        // `llm` likewise holds the provider it asks.
        let llm = match config.classifiers.get("llm").filter(|c| c.enabled) {
            Some(cfg) => {
                let classifier = llm::LlmClassifier::from_settings(&cfg.settings, &config.providers)
                    .context("configuring classifier 'llm'")?;
                classifiers.push(Arc::new(classifier));
                true
            }
            None => false,
        };

        for (id, enabled, has_rule) in [
            ("cev", cev.is_some(), config.routers.iter().any(|r| matches!(r, RouterRule::Cev { .. }))),
            ("llm", llm, config.routers.iter().any(|r| matches!(r, RouterRule::Llm))),
        ] {
            if enabled && !has_rule {
                tracing::warn!("classifier '{id}' is enabled, but no `type = \"{id}\"` rule in `routers` uses its picks");
            } else if !enabled && has_rule {
                tracing::warn!("`routers` has a `type = \"{id}\"` rule, but classifier '{id}' isn't enabled, so it never matches");
            }
        }
        if cev.is_some() && llm {
            tracing::warn!("classifiers 'cev' and 'llm' are both enabled; only cev's pick is kept, so 'llm' is asked for nothing");
        }

        let entries = classifiers
            .into_iter()
            .map(|classifier| {
                let cfg = config.classifiers.get(classifier.id());
                ClassifierEntry {
                    enabled: cfg.is_some_and(|c| c.enabled),
                    settings: cfg.map(|c| c.settings.clone()).unwrap_or_default(),
                    classifier,
                }
            })
            .collect();

        Ok(ClassifierRegistry { entries, cev })
    }

    /// The `cev` classifier, if it's enabled.
    pub fn cev(&self) -> Option<&cev::CevClassifier> {
        self.cev.as_deref()
    }

    /// A registry running only `classifier`.
    #[cfg(test)]
    pub fn with_cev(classifier: cev::CevClassifier) -> Self {
        let classifier = Arc::new(classifier);
        ClassifierRegistry {
            entries: vec![ClassifierEntry {
                classifier: classifier.clone(),
                enabled: true,
                settings: Map::new(),
            }],
            cev: Some(classifier),
        }
    }

    /// Runs every enabled classifier over `req` and returns the de-duplicated
    /// union of tags they produce, in registry order, along with the first
    /// model one of them picks.
    pub async fn classify(&self, req: &ChatRequest) -> Classification {
        let mut merged = Classification::default();

        for entry in &self.entries {
            if !entry.enabled {
                continue;
            }

            let ctx = ClassifierContext {
                settings: entry.settings.clone(),
            };

            match entry.classifier.classify(&ctx, req).await {
                Ok(classification) => {
                    for tag in classification.tags {
                        if !merged.tags.contains(&tag) {
                            merged.tags.push(tag);
                        }
                    }
                    if merged.route.is_none() {
                        merged.route = classification.route;
                    }
                }
                Err(err) => {
                    tracing::warn!("classifier '{}' failed: {err:#}", entry.classifier.id());
                }
            }
        }

        merged
    }
}

/// Tags a response after the provider has replied, so refusals, policy
/// violations, etc. can be flagged without the client having asked for it.
///
/// Response classifiers run at [`crate::plugins::Stage::PostResponse`], after
/// plugins have had a chance to repair/transform the response. Every
/// classifier enabled in config (via `[response_classifiers.<id>]`) runs, in
/// registry order, and the merged tags are written to
/// [`ChatResponse::tags`](crate::canonical::ChatResponse::tags) — which is
/// never mapped into the OpenAI/Anthropic wire response types, so it reaches
/// clients only via the `X-Router-Tags` response header (see
/// [`crate::server`]). A classifier that errors is logged and skipped; it
/// never fails the request.
#[async_trait]
pub trait ResponseClassifier: Send + Sync {
    /// Identifier used in `[response_classifiers.<id>]` config.
    fn id(&self) -> &'static str;

    /// Inspects `resp` (and, if useful, the originating `req`) and returns
    /// the tags it should be labeled with, e.g. `["refusal"]`. Returning an
    /// empty list means "no opinion".
    async fn classify(
        &self,
        ctx: &ClassifierContext,
        req: &ChatRequest,
        resp: &ChatResponse,
    ) -> anyhow::Result<Vec<String>>;
}

struct ResponseClassifierEntry {
    classifier: Arc<dyn ResponseClassifier>,
    enabled: bool,
    settings: Map<String, Value>,
}

pub struct ResponseClassifierRegistry {
    /// All known response classifiers, in a fixed order; enabled ones run in
    /// this order and their tags are merged in the same order.
    entries: Vec<ResponseClassifierEntry>,
}

impl ResponseClassifierRegistry {
    pub fn from_config(config: &Config) -> Self {
        let classifiers: Vec<Arc<dyn ResponseClassifier>> = vec![Arc::new(refusal::RefusalClassifier)];

        let entries = classifiers
            .into_iter()
            .map(|classifier| {
                let cfg = config.response_classifiers.get(classifier.id());
                ResponseClassifierEntry {
                    enabled: cfg.is_some_and(|c| c.enabled),
                    settings: cfg.map(|c| c.settings.clone()).unwrap_or_default(),
                    classifier,
                }
            })
            .collect();

        ResponseClassifierRegistry { entries }
    }

    /// Runs every enabled response classifier over `resp` and returns the
    /// de-duplicated union of tags they produce, in registry order.
    pub async fn classify(&self, req: &ChatRequest, resp: &ChatResponse) -> Vec<String> {
        let mut tags: Vec<String> = Vec::new();

        for entry in &self.entries {
            if !entry.enabled {
                continue;
            }

            let ctx = ClassifierContext {
                settings: entry.settings.clone(),
            };

            match entry.classifier.classify(&ctx, req, resp).await {
                Ok(new_tags) => {
                    for tag in new_tags {
                        if !tags.contains(&tag) {
                            tags.push(tag);
                        }
                    }
                }
                Err(err) => {
                    tracing::warn!("response classifier '{}' failed: {err}", entry.classifier.id());
                }
            }
        }

        tags
    }
}
