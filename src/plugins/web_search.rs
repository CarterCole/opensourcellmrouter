//! `web`: augments a request with real-time web search results.
//!
//! Not yet implemented — enabling this plugin (via `[plugins.web]
//! enabled = true` or a request's `plugins` array) fails the request with a
//! clear error rather than silently doing nothing.
//!
//! To implement: configure a search backend (`base_url`, `api_key_env`,
//! `max_results`, similar to [`crate::config::ProviderConfig`]), call it
//! from `pre_request` with the last user message as the query, and prepend
//! the results to `req.system` as context.

use async_trait::async_trait;

use super::{Flow, Plugin, PluginContext};
use crate::canonical::{ChatRequest, ChatResponse};

pub struct WebSearchPlugin;

#[async_trait]
impl Plugin for WebSearchPlugin {
    fn id(&self) -> &'static str {
        "web"
    }

    async fn pre_request(
        &self,
        _ctx: &PluginContext,
        _req: &mut ChatRequest,
        _resp: &mut Option<ChatResponse>,
    ) -> anyhow::Result<Flow> {
        anyhow::bail!(
            "plugin 'web' (Web Search) is enabled but not yet implemented: configure a search \
             backend and implement WebSearchPlugin::pre_request"
        )
    }
}
