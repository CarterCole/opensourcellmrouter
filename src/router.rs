//! Resolves a request's `model` to a [`Provider`] by walking the
//! `routers` chain from config in order.
//!
//! Each [`RouterRule`] either decides a provider for the request, or
//! "passes through" so the next rule gets a chance. The first decision wins.

use std::collections::{HashMap, HashSet};

use anyhow::bail;
use rand::seq::SliceRandom;

use crate::config::{Config, ProviderConfig, RandomCandidate, RouterRule};
use crate::provider::Provider;

#[derive(Clone, Copy)]
enum Direction {
    Lower,
    Higher,
}
use Direction::{Higher, Lower};

pub struct ModelRouter {
    providers: HashMap<String, Provider>,
    provider_configs: HashMap<String, ProviderConfig>,
    rules: Vec<RouterRule>,
    /// Models each provider reports having available, populated by
    /// [`Self::discover_models`]. Empty (and every [`RouterRule::Discover`]
    /// passes through) until that's been called.
    available_models: HashMap<String, HashSet<String>>,
}

impl ModelRouter {
    pub fn from_config(config: &Config) -> anyhow::Result<Self> {
        let mut providers = HashMap::new();
        let mut provider_configs = HashMap::new();
        for provider_config in &config.providers {
            // Check API key env var if declared.
            if let Some(var) = &provider_config.api_key_env {
                let missing = matches!(std::env::var(var), Ok(v) if v.is_empty())
                    || std::env::var(var).is_err();
                if missing {
                    if provider_config.strict {
                        tracing::warn!(
                            "skipping provider '{}': ${var} is not set (strict = true)",
                            provider_config.name
                        );
                        continue;
                    } else {
                        tracing::warn!(
                            "provider '{}': ${var} is not set — requests will fail until it is",
                            provider_config.name
                        );
                    }
                }
            }
            providers.insert(provider_config.name.clone(), Provider::from_config(provider_config));
            provider_configs.insert(provider_config.name.clone(), provider_config.clone());
        }

        if providers.is_empty() {
            bail!("no providers available (all require API keys that are not set)");
        }

        Ok(ModelRouter {
            providers,
            provider_configs,
            rules: config.routers.clone(),
            available_models: HashMap::new(),
        })
    }

    /// Queries every provider for the models it currently has available
    /// (see [`Provider::list_models`]) and caches the result for
    /// [`RouterRule::Discover`] rules. Best-effort: a provider that's
    /// unreachable is logged as a warning and left with no known models, so
    /// `discover` rules for it simply pass through.
    pub async fn discover_models(&mut self, client: &reqwest::Client) {
        for (name, provider) in &self.providers {
            match provider.list_models(client).await {
                Ok(models) if !models.is_empty() => {
                    tracing::info!("provider '{name}' reports {} model(s): {models:?}", models.len());
                    self.available_models.insert(name.clone(), models.into_iter().collect());
                }
                Ok(_) => {}
                Err(err) => {
                    tracing::warn!("failed to list models for provider '{name}': {err:#}");
                }
            }
        }
    }

    /// Look up a provider by name directly, bypassing the `routers` chain.
    /// Used when a plugin (e.g. `pareto-router`) forces a specific provider.
    pub fn provider(&self, name: &str) -> Option<&Provider> {
        self.providers.get(name)
    }

    /// Resolve the requested model to a provider and the model name to send
    /// it (after any `rewrite_model`). `tags` are the classifier tags
    /// assigned to this request (see [`crate::classifiers`]), consulted by
    /// [`RouterRule::Tag`] rules.
    pub fn resolve(&self, model: &str, tags: &[String]) -> Option<(&Provider, String)> {
        for rule in &self.rules {
            if let Some((name, target_model)) = self.apply_rule(rule, model, tags) {
                if let Some(provider) = self.providers.get(&name) {
                    return Some((provider, target_model));
                }
            }
        }
        None
    }

    fn apply_rule(&self, rule: &RouterRule, model: &str, tags: &[String]) -> Option<(String, String)> {
        match rule {
            RouterRule::Prefix {
                model_prefix,
                provider,
                rewrite_model,
            } => {
                if model.starts_with(model_prefix.as_str()) {
                    let target = rewrite_model.clone().unwrap_or_else(|| model.to_string());
                    Some((provider.clone(), target))
                } else {
                    None
                }
            }
            RouterRule::Tag {
                tag,
                provider,
                rewrite_model,
            } => {
                if tags.iter().any(|t| t == tag) {
                    let target = rewrite_model.clone().unwrap_or_else(|| model.to_string());
                    Some((provider.clone(), target))
                } else {
                    None
                }
            }
            RouterRule::Price {
                providers,
                max_cost_per_1m_tokens,
            } => self.best_by(
                providers,
                *max_cost_per_1m_tokens,
                model,
                |pc| Some(pc.cost_per_1m_tokens),
                Lower,
            ),
            RouterRule::Latency {
                providers,
                max_latency_ms,
            } => self.best_by(providers, *max_latency_ms, model, |pc| pc.latency_ms, Lower),
            RouterRule::Throughput {
                providers,
                min_tokens_per_sec,
            } => self.best_by(
                providers,
                *min_tokens_per_sec,
                model,
                |pc| pc.throughput_tokens_per_sec,
                Higher,
            ),
            RouterRule::Random {
                providers,
                rewrite_model,
                candidates,
            } => {
                if !candidates.is_empty() {
                    candidates
                        .choose(&mut rand::thread_rng())
                        .map(|RandomCandidate { provider, model }| (provider.clone(), model.clone()))
                } else {
                    let names: Vec<String> = self.candidate_names(providers).collect();
                    names.choose(&mut rand::thread_rng()).map(|name| {
                        let target = rewrite_model.clone().unwrap_or_else(|| model.to_string());
                        (name.clone(), target)
                    })
                }
            }
            RouterRule::Discover { provider } => {
                if self
                    .available_models
                    .get(provider)
                    .is_some_and(|models| models.contains(model))
                {
                    Some((provider.clone(), model.to_string()))
                } else {
                    None
                }
            }
            RouterRule::Fallback { providers, quality_bias, rewrite_model } => {
                let mut scored: Vec<(String, f64)> = self
                    .candidate_names(providers)
                    .filter_map(|name| {
                        let pc = self.provider_configs.get(&name)?;
                        let score =
                            quality_bias * pc.quality - (1.0 - quality_bias) * pc.cost_per_1m_tokens;
                        Some((name, score))
                    })
                    .collect();
                scored.sort_by(|a, b| b.1.total_cmp(&a.1));
                scored.into_iter().next().map(|(name, _)| {
                    let target = rewrite_model.clone().unwrap_or_else(|| model.to_string());
                    (name, target)
                })
            }
        }
    }

    fn candidate_names<'a>(&'a self, providers: &'a [String]) -> impl Iterator<Item = String> + 'a {
        if providers.is_empty() {
            Box::new(self.provider_configs.keys().cloned()) as Box<dyn Iterator<Item = String>>
        } else {
            Box::new(providers.iter().cloned())
        }
    }

    /// Picks the candidate that optimizes `metric` (lowest, for
    /// [`Direction::Lower`], or highest, for [`Direction::Higher`]).
    /// Candidates for which `metric` returns `None`, or which fail
    /// `threshold` (≤ for `Lower`, ≥ for `Higher`), are excluded. Returns
    /// `None` (pass through) if no candidate qualifies.
    fn best_by(
        &self,
        providers: &[String],
        threshold: Option<f64>,
        model: &str,
        metric: impl Fn(&ProviderConfig) -> Option<f64>,
        direction: Direction,
    ) -> Option<(String, String)> {
        let mut best: Option<(String, f64)> = None;

        for name in self.candidate_names(providers) {
            let Some(pc) = self.provider_configs.get(&name) else { continue };
            let Some(value) = metric(pc) else { continue };

            if let Some(threshold) = threshold {
                let qualifies = match direction {
                    Direction::Lower => value <= threshold,
                    Direction::Higher => value >= threshold,
                };
                if !qualifies {
                    continue;
                }
            }

            let better = match &best {
                None => true,
                Some((_, best_value)) => match direction {
                    Direction::Lower => value < *best_value,
                    Direction::Higher => value > *best_value,
                },
            };
            if better {
                best = Some((name, value));
            }
        }

        best.map(|(name, _)| (name, model.to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Config;

    #[test]
    fn tag_rule_routes_image_and_video_requests_to_vision_provider() {
        let config: Config = toml::from_str(
            r#"
            [[providers]]
            name = "local"
            format = "openai"
            base_url = "http://localhost:8080/v1"
            cost_per_1m_tokens = 0.0
            quality = 40

            [[providers]]
            name = "vision-provider"
            format = "openai"
            base_url = "http://localhost:9090/v1"
            cost_per_1m_tokens = 5.0
            quality = 80

            [[routers]]
            type = "tag"
            tag = "vision"
            provider = "vision-provider"
            rewrite_model = "vision-model"

            [[routers]]
            type = "tag"
            tag = "video"
            provider = "vision-provider"
            rewrite_model = "video-model"

            [[routers]]
            type = "fallback"
            "#,
        )
        .unwrap();

        let router = ModelRouter::from_config(&config).unwrap();

        // A request tagged "vision" by a classifier is routed to the
        // vision-capable provider and rewritten, regardless of `model`.
        let (provider, model) = router.resolve("gpt-4", &["vision".to_string()]).unwrap();
        assert_eq!(provider.name, "vision-provider");
        assert_eq!(model, "vision-model");

        // Likewise for "video".
        let (provider, model) = router.resolve("gpt-4", &["video".to_string()]).unwrap();
        assert_eq!(provider.name, "vision-provider");
        assert_eq!(model, "video-model");

        // No matching tag: the `tag` rules pass through to `fallback`,
        // which picks the highest-scoring provider (here, the
        // higher-quality "vision-provider") and leaves `model` untouched.
        let (provider, model) = router.resolve("gpt-4", &[]).unwrap();
        assert_eq!(provider.name, "vision-provider");
        assert_eq!(model, "gpt-4");
    }

    #[test]
    fn discover_rule_routes_to_provider_with_matching_model() {
        let config: Config = toml::from_str(
            r#"
            [[providers]]
            name = "ollama"
            format = "ollama"
            base_url = "http://localhost:11434"

            [[providers]]
            name = "openai"
            format = "openai"
            base_url = "https://api.openai.com/v1"
            cost_per_1m_tokens = 5.0
            quality = 80

            [[routers]]
            type = "discover"
            provider = "ollama"

            [[routers]]
            type = "fallback"
            "#,
        )
        .unwrap();

        let mut router = ModelRouter::from_config(&config).unwrap();

        // Before discovery has run, "discover" passes through to fallback.
        let (provider, _) = router.resolve("llama3:8b", &[]).unwrap();
        assert_eq!(provider.name, "openai");

        // Simulate discovery having found "llama3:8b" on the ollama provider.
        router
            .available_models
            .insert("ollama".to_string(), ["llama3:8b".to_string()].into_iter().collect());

        let (provider, model) = router.resolve("llama3:8b", &[]).unwrap();
        assert_eq!(provider.name, "ollama");
        assert_eq!(model, "llama3:8b");

        // A model ollama doesn't report having falls through to fallback.
        let (provider, _) = router.resolve("gpt-4", &[]).unwrap();
        assert_eq!(provider.name, "openai");
    }
}
