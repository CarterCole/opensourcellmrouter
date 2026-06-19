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
//! then match on those tags. A classifier that errors is logged and skipped;
//! it never fails the request.

pub mod keyword;
pub mod refusal;

use std::sync::Arc;

use async_trait::async_trait;
use serde_json::{Map, Value};

use crate::canonical::{ChatRequest, ChatResponse};
use crate::config::Config;

/// Per-classifier settings, from `[classifiers.<id>]` in config.
pub struct ClassifierContext {
    pub settings: Map<String, Value>,
}

#[async_trait]
pub trait Classifier: Send + Sync {
    /// Identifier used in `[classifiers.<id>]` config.
    fn id(&self) -> &'static str;

    /// Inspects `req` (before routing) and returns the tags it should be
    /// labeled with, e.g. `["vision"]`. Returning an empty list means "no
    /// opinion".
    async fn classify(&self, ctx: &ClassifierContext, req: &ChatRequest) -> anyhow::Result<Vec<String>>;
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
}

impl ClassifierRegistry {
    pub fn from_config(config: &Config) -> Self {
        let classifiers: Vec<Arc<dyn Classifier>> = vec![Arc::new(keyword::KeywordClassifier)];

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

        ClassifierRegistry { entries }
    }

    /// Runs every enabled classifier over `req` and returns the de-duplicated
    /// union of tags they produce, in registry order.
    pub async fn classify(&self, req: &ChatRequest) -> Vec<String> {
        let mut tags: Vec<String> = Vec::new();

        for entry in &self.entries {
            if !entry.enabled {
                continue;
            }

            let ctx = ClassifierContext {
                settings: entry.settings.clone(),
            };

            match entry.classifier.classify(&ctx, req).await {
                Ok(new_tags) => {
                    for tag in new_tags {
                        if !tags.contains(&tag) {
                            tags.push(tag);
                        }
                    }
                }
                Err(err) => {
                    tracing::warn!("classifier '{}' failed: {err}", entry.classifier.id());
                }
            }
        }

        tags
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
