//! `pareto-router`: picks a provider from a named "coding quality tier"
//! (e.g. low/medium/high), letting a request opt into a cheaper or
//! higher-quality model without hardcoding a provider name.
//!
//! Config:
//!
//! ```toml
//! [plugins.pareto-router]
//! enabled = true
//! default_tier = "medium"
//!
//! [plugins.pareto-router.tiers]
//! low = ["local-llama"]
//! medium = ["openai"]
//! high = ["anthropic"]
//! ```
//!
//! A request can override the tier per-call:
//!
//! ```json
//! {"plugins": [{"id": "pareto-router", "tier": "high"}]}
//! ```

use async_trait::async_trait;
use serde_json::Value;

use super::{Flow, Plugin, PluginContext};
use crate::canonical::{ChatRequest, ChatResponse};

pub struct ParetoRouterPlugin;

#[async_trait]
impl Plugin for ParetoRouterPlugin {
    fn id(&self) -> &'static str {
        "pareto-router"
    }

    async fn pre_request(
        &self,
        ctx: &PluginContext,
        req: &mut ChatRequest,
        _resp: &mut Option<ChatResponse>,
    ) -> anyhow::Result<Flow> {
        let tier = ctx
            .get_str("tier")
            .or_else(|| ctx.get_str("default_tier"))
            .unwrap_or("medium");

        let tiers = ctx
            .settings
            .get("tiers")
            .and_then(Value::as_object)
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "pareto-router: no `tiers` configured (e.g. [plugins.pareto-router.tiers])"
                )
            })?;

        let providers = tiers.get(tier).and_then(Value::as_array).ok_or_else(|| {
            let known: Vec<&str> = tiers.keys().map(String::as_str).collect();
            anyhow::anyhow!("pareto-router: unknown tier '{tier}' (configured tiers: {known:?})")
        })?;

        let provider = providers
            .first()
            .and_then(Value::as_str)
            .ok_or_else(|| anyhow::anyhow!("pareto-router: tier '{tier}' has no providers configured"))?;

        req.forced_provider = Some(provider.to_string());
        Ok(Flow::Continue)
    }
}
