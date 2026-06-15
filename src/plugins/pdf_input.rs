//! `pdf`: parse and extract content from uploaded PDF files.
//!
//! Not yet implemented — enabling this plugin (via `[plugins.pdf]
//! enabled = true` or a request's `plugins` array) fails the request with a
//! clear error rather than silently doing nothing.
//!
//! To implement: extend [`crate::canonical::Message`] to carry base64
//! document attachments (mirroring Anthropic's `document` content blocks
//! and OpenAI's `file` content parts), add a PDF text-extraction crate, and
//! have `pre_request` replace each PDF attachment with its extracted text.

use async_trait::async_trait;

use super::{Flow, Plugin, PluginContext};
use crate::canonical::{ChatRequest, ChatResponse};

pub struct PdfInputPlugin;

#[async_trait]
impl Plugin for PdfInputPlugin {
    fn id(&self) -> &'static str {
        "pdf"
    }

    async fn pre_request(
        &self,
        _ctx: &PluginContext,
        _req: &mut ChatRequest,
        _resp: &mut Option<ChatResponse>,
    ) -> anyhow::Result<Flow> {
        anyhow::bail!(
            "plugin 'pdf' (PDF Inputs) is enabled but not yet implemented: extend \
             canonical::Message to support attachments and add PDF text extraction"
        )
    }
}
