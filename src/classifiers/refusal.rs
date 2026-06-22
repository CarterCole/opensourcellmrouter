//! `refusal`: tags a response that looks like a model refusal.
//!
//! Config:
//!
//! ```toml
//! [response_classifiers.refusal]
//! enabled = true
//! phrases = ["i cannot help with that", "i can't assist with"]
//! ```
//!
//! `phrases` defaults to [`DEFAULT_PHRASES`] if omitted. The response
//! content is searched case-insensitively for any configured phrase; a hit
//! adds the `"refusal"` tag.
//!
//! This is a simple substring match, not a real refusal classifier — it's
//! meant as a configurable baseline (and an example for writing more
//! sophisticated [`super::ResponseClassifier`] implementations, e.g. ones
//! that call out to a moderation model).

use async_trait::async_trait;
use serde_json::Value;

use super::{ClassifierContext, ResponseClassifier};
use crate::canonical::{ChatRequest, ChatResponse};

const DEFAULT_PHRASES: &[&str] = &[
    "i cannot help with that",
    "i can't help with that",
    "i cannot assist with",
    "i can't assist with",
    "i'm not able to help with",
    "i am not able to help with",
    "as an ai language model",
    "i won't be able to help with that",
];

pub struct RefusalClassifier;

#[async_trait]
impl ResponseClassifier for RefusalClassifier {
    fn id(&self) -> &'static str {
        "refusal"
    }

    async fn classify(
        &self,
        ctx: &ClassifierContext,
        _req: &ChatRequest,
        resp: &ChatResponse,
    ) -> anyhow::Result<Vec<String>> {
        let configured: Vec<String> = ctx
            .settings
            .get("phrases")
            .and_then(Value::as_array)
            .map(|arr| arr.iter().filter_map(Value::as_str).map(str::to_lowercase).collect())
            .unwrap_or_default();

        let phrases: Vec<&str> = if configured.is_empty() {
            DEFAULT_PHRASES.to_vec()
        } else {
            configured.iter().map(String::as_str).collect()
        };

        let haystack = resp.content.to_lowercase();
        let hit = phrases.iter().any(|phrase| haystack.contains(phrase));

        Ok(if hit { vec!["refusal".to_string()] } else { Vec::new() })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::canonical::{StopReason, Usage};
    use serde_json::json;

    fn response(content: &str) -> ChatResponse {
        ChatResponse {
            id: "test".to_string(),
            model: "test-model".to_string(),
            content: content.to_string(),
            stop_reason: StopReason::EndTurn,
            tool_calls: Vec::new(),
            usage: Usage::default(),
            tags: Vec::new(),
        }
    }

    fn request() -> ChatRequest {
        ChatRequest {
            model: "test-model".to_string(),
            system: None,
            messages: Vec::new(),
            max_tokens: None,
            temperature: None,
            thinking: None,
            effort: None,
            task_budget: None,
            output_schema: None,
            tools: Vec::new(),
            stream: false,
            plugins: Vec::new(),
            forced_provider: None,
            tags: Vec::new(),
        }
    }

    #[tokio::test]
    async fn tags_default_refusal_phrase() {
        let classifier = RefusalClassifier;
        let ctx = ClassifierContext {
            settings: Default::default(),
        };

        let resp = response("I'm sorry, but I cannot help with that request.");
        let tags = classifier.classify(&ctx, &request(), &resp).await.unwrap();
        assert_eq!(tags, vec!["refusal".to_string()]);
    }

    #[tokio::test]
    async fn no_match_returns_empty() {
        let classifier = RefusalClassifier;
        let ctx = ClassifierContext {
            settings: Default::default(),
        };

        let resp = response("Sure, here's a haiku about autumn.");
        let tags = classifier.classify(&ctx, &request(), &resp).await.unwrap();
        assert!(tags.is_empty());
    }

    #[tokio::test]
    async fn uses_configured_phrases() {
        let classifier = RefusalClassifier;
        let ctx = ClassifierContext {
            settings: json!({"phrases": ["not going to do that"]}).as_object().unwrap().clone(),
        };

        let resp = response("Not Going To Do That, sorry.");
        let tags = classifier.classify(&ctx, &request(), &resp).await.unwrap();
        assert_eq!(tags, vec!["refusal".to_string()]);

        let resp = response("i cannot help with that");
        let tags = classifier.classify(&ctx, &request(), &resp).await.unwrap();
        assert!(tags.is_empty(), "default phrases shouldn't apply once phrases are configured");
    }
}
