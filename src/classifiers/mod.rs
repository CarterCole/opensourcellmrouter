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
//! Every classifier enabled in config (via `[classifiers.<id>]`) runs, all
//! at once, on every request. Each produces zero or more tags, which are
//! merged (de-duplicated, in registry order) into [`ChatRequest::tags`](crate::canonical::ChatRequest::tags).
//! `routers` rules — in particular [`crate::config::RouterRule::Tag`] — can
//! then match on those tags. A classifier can also pick the model to route
//! to outright (see [`cev`] and [`llm`]), for a
//! [`crate::config::RouterRule::Cev`] or [`crate::config::RouterRule::Llm`]
//! rule to act on; every pick is kept, and the first of those rules in the
//! chain decides. Running both takes `mode = "teacher"` on `llm`, which
//! also makes its pick label `cev`'s decision, so cev learns from every
//! request whichever one routes it. A classifier that errors is logged and
//! skipped; it never fails the request.

pub mod cev;
pub mod keyword;
pub mod llm;
pub mod refusal;

use std::collections::HashSet;
use std::sync::Arc;

use anyhow::{Context, bail};
use async_trait::async_trait;
use futures_util::future::join_all;
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

/// What all the enabled classifiers concluded about a request.
#[derive(Debug, Default)]
pub struct Classified {
    /// Every tag any of them gave it, de-duplicated, in registry order.
    pub tags: Vec<String>,
    /// Every model one of them picked, in registry order.
    pub routes: Vec<RouteChoice>,
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
    /// The `llm` classifier, if enabled; if it's a teacher, its picks label
    /// cev's decisions.
    llm: Option<Arc<llm::LlmClassifier>>,
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
                Some(Arc::new(classifier))
            }
            None => None,
        };
        if let Some(llm) = &llm {
            classifiers.push(llm.clone());
        }

        // Running both is a choice: llm teaches cev, or one of them goes.
        match (&cev, &llm) {
            (Some(_), Some(llm)) if !llm.teaches() => bail!(
                "classifiers 'cev' and 'llm' are both enabled: set `mode = \"teacher\"` under \
                 [classifiers.llm] to run both, with llm's picks training cev, or enable only one"
            ),
            (None, Some(llm)) if llm.teaches() => bail!(
                "classifier 'llm' has `mode = \"teacher\"`, which trains the 'cev' classifier, \
                 but that isn't enabled"
            ),
            _ => {}
        }

        check_routing(config, cev.as_deref(), llm.as_deref());

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

        Ok(ClassifierRegistry { entries, cev, llm })
    }

    /// The `cev` classifier, if it's enabled.
    pub fn cev(&self) -> Option<&cev::CevClassifier> {
        self.cev.as_deref()
    }

    /// A registry running only `cev`, and `llm` if given.
    #[cfg(test)]
    pub fn with_cev(cev: cev::CevClassifier, llm: Option<llm::LlmClassifier>) -> Self {
        let cev = Arc::new(cev);
        let llm = llm.map(Arc::new);
        let mut classifiers: Vec<Arc<dyn Classifier>> = vec![cev.clone()];
        classifiers.extend(llm.clone().map(|llm| llm as Arc<dyn Classifier>));
        ClassifierRegistry {
            entries: classifiers
                .into_iter()
                .map(|classifier| ClassifierEntry { classifier, enabled: true, settings: Map::new() })
                .collect(),
            cev: Some(cev),
            llm,
        }
    }

    /// Runs every enabled classifier over `req`, all at once, and returns
    /// the de-duplicated union of tags they produce and every model they
    /// pick, both in registry order.
    pub async fn classify(&self, req: &ChatRequest) -> Classified {
        let classifications = join_all(self.entries.iter().filter(|entry| entry.enabled).map(|entry| async move {
            let ctx = ClassifierContext { settings: entry.settings.clone() };
            (entry.classifier.id(), entry.classifier.classify(&ctx, req).await)
        }))
        .await;

        let mut merged = Classified::default();
        for (id, classification) in classifications {
            match classification {
                Ok(classification) => {
                    for tag in classification.tags {
                        if !merged.tags.contains(&tag) {
                            merged.tags.push(tag);
                        }
                    }
                    merged.routes.extend(classification.route);
                }
                Err(err) => {
                    tracing::warn!("classifier '{id}' failed: {err:#}");
                }
            }
        }

        self.label_cev_with_llm(&merged.routes);
        merged
    }

    /// Labels cev's decision about a request with llm's pick for it, if
    /// llm is a teacher and both made a pick. The label is sent in the
    /// background: routing doesn't wait for cev to learn from it.
    fn label_cev_with_llm(&self, routes: &[RouteChoice]) {
        let (Some(cev), Some(llm)) = (&self.cev, self.llm.as_ref().filter(|llm| llm.teaches())) else {
            return;
        };
        let decision_id = routes.iter().find(|r| r.classifier == "cev").and_then(|r| r.decision_id.clone());
        let pick = routes.iter().find(|r| r.classifier == "llm");
        let (Some(decision_id), Some(pick)) = (decision_id, pick) else {
            return;
        };
        if !cev.learns_from_llm() {
            return;
        }

        let (cev, llm, pick) = (cev.clone(), llm.clone(), pick.clone());
        tokio::spawn(async move {
            if let Err(err) = cev.learn_from_llm(&decision_id, &pick, llm.asked()).await {
                tracing::warn!("couldn't label cev decision {decision_id} with llm's pick '{}': {err}", pick.choice);
            }
        });
    }
}

/// Warns about routing classifiers and `routers` rules that don't add up:
/// a rule for a classifier that isn't enabled, a classifier nothing uses,
/// or `llm` picks that can't label cev's decisions.
fn check_routing(config: &Config, cev: Option<&cev::CevClassifier>, llm: Option<&llm::LlmClassifier>) {
    let cev_rule = config.routers.iter().any(|r| matches!(r, RouterRule::Cev { .. }));
    let llm_rule = config.routers.iter().any(|r| matches!(r, RouterRule::Llm));
    // Whether llm's picks label cev's decisions.
    let teaching = llm.is_some_and(|llm| llm.teaches()) && cev.is_some_and(|cev| cev.learns_from_llm());

    for (id, enabled, has_rule) in [("cev", cev.is_some(), cev_rule), ("llm", llm.is_some(), llm_rule)] {
        if !enabled && has_rule {
            tracing::warn!(
                "`routers` has a `type = \"{id}\"` rule, but classifier '{id}' isn't enabled, so it never matches"
            );
        }
    }
    if cev.is_some() && !cev_rule {
        let learning = if teaching { " and learn from 'llm''s picks" } else { "" };
        tracing::info!(
            "classifier 'cev' routes nothing (no `type = \"cev\"` rule), but still decides every request so it can take feedback{learning}"
        );
    }
    if llm.is_some() && !llm_rule {
        if teaching {
            tracing::info!(
                "classifier 'llm' routes nothing (no `type = \"llm\"` rule); its picks only label cev's decisions"
            );
        } else {
            tracing::warn!("classifier 'llm' is enabled, but no `type = \"llm\"` rule in `routers` uses its picks");
        }
    }

    if let (Some(cev), Some(llm), true) = (cev, llm, teaching) {
        for option in llm.models() {
            if cev.option_for(&option.provider, &option.model).is_none() {
                tracing::warn!(
                    "classifier 'llm' can pick '{}', but none of cev's `models` is {}/{}, so that pick won't label cev's decisions",
                    option.name(),
                    option.provider,
                    option.model
                );
            }
        }
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

            let ctx = ClassifierContext { settings: entry.settings.clone() };

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

#[cfg(test)]
mod tests {
    use super::*;

    /// A config with the `cev` classifier, and the `llm` one with `llm_mode`
    /// (a line of its settings, or nothing), if each is enabled.
    fn config(cev: bool, llm: Option<&str>) -> Config {
        let cev = if cev {
            r#"
            [classifiers.cev]
            enabled = true
            url = "http://127.0.0.1:1"

            [[classifiers.cev.models]]
            provider = "ollama"
            model = "a"

            [[classifiers.cev.models]]
            provider = "ollama"
            model = "b"
            "#
        } else {
            ""
        };
        let llm = match llm {
            Some(mode) => format!(
                r#"
                [classifiers.llm]
                enabled = true
                provider = "ollama"
                model = "m"
                {mode}

                [[classifiers.llm.models]]
                provider = "ollama"
                model = "a"

                [[classifiers.llm.models]]
                provider = "ollama"
                model = "b"
                "#
            ),
            None => String::new(),
        };
        toml::from_str(&format!(
            r#"
            [[providers]]
            name = "ollama"
            format = "ollama"
            base_url = "http://localhost:11434"
            {cev}
            {llm}
            "#
        ))
        .unwrap()
    }

    fn build(config: &Config) -> anyhow::Result<()> {
        ClassifierRegistry::from_config(config).map(|_| ())
    }

    #[test]
    fn either_routing_classifier_runs_alone() {
        build(&config(true, None)).unwrap();
        build(&config(false, Some(""))).unwrap();
    }

    #[test]
    fn running_both_takes_teacher_mode() {
        let err = build(&config(true, Some(""))).unwrap_err();
        assert!(err.to_string().contains("set `mode = \"teacher\"`"), "{err}");

        let err = build(&config(true, Some("mode = \"route\""))).unwrap_err();
        assert!(err.to_string().contains("set `mode = \"teacher\"`"), "{err}");

        let registry = ClassifierRegistry::from_config(&config(true, Some("mode = \"teacher\""))).unwrap();
        assert!(registry.cev().is_some());
    }

    #[test]
    fn teacher_mode_needs_cev_to_teach() {
        let err = build(&config(false, Some("mode = \"teacher\""))).unwrap_err();
        assert!(err.to_string().contains("isn't enabled"), "{err}");
    }
}
