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

use std::sync::Arc;

use async_trait::async_trait;
use serde_json::{Map, Value};

use crate::canonical::ChatRequest;
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
