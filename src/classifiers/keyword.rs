//! `keyword`: tags a request based on configurable keyword/phrase lists.
//!
//! Config:
//!
//! ```toml
//! [classifiers.keyword]
//! enabled = true
//!
//! [classifiers.keyword.tags]
//! vision = ["image", "photo", "picture", "screenshot"]
//! nsfw = ["explicit term", "another term"]
//! ```
//!
//! The system prompt and every message's content are concatenated and
//! searched case-insensitively for each tag's keywords; any tag with at
//! least one match is added to the request's tags. Downstream `routers`
//! rules (see [`crate::config::RouterRule::Tag`]) can then route on those
//! tags, e.g. sending "vision"-tagged requests to a multimodal model or
//! "nsfw"-tagged requests to a moderation provider.
//!
//! This is a simple substring match, not a real moderation or modality
//! classifier — it's meant as a configurable baseline (and an example for
//! writing more sophisticated [`super::Classifier`] implementations).

use async_trait::async_trait;
use serde_json::Value;

use super::{Classifier, ClassifierContext};
use crate::canonical::ChatRequest;

pub struct KeywordClassifier;

#[async_trait]
impl Classifier for KeywordClassifier {
    fn id(&self) -> &'static str {
        "keyword"
    }

    async fn classify(&self, ctx: &ClassifierContext, req: &ChatRequest) -> anyhow::Result<Vec<String>> {
        let Some(tags) = ctx.settings.get("tags").and_then(Value::as_object) else {
            return Ok(Vec::new());
        };

        let mut haystack = req.system.clone().unwrap_or_default();
        for message in &req.messages {
            haystack.push(' ');
            haystack.push_str(&message.content);
        }
        let haystack = haystack.to_lowercase();

        let mut matched = Vec::new();
        for (tag, keywords) in tags {
            let Some(keywords) = keywords.as_array() else {
                continue;
            };
            let hit = keywords
                .iter()
                .filter_map(Value::as_str)
                .any(|kw| haystack.contains(&kw.to_lowercase()));
            if hit {
                matched.push(tag.clone());
            }
        }

        Ok(matched)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::canonical::{Message, Role};
    use serde_json::json;

    fn request(system: Option<&str>, content: &str) -> ChatRequest {
        ChatRequest {
            model: "test-model".to_string(),
            system: system.map(str::to_string),
            messages: vec![Message {
                role: Role::User,
                content: content.to_string(),
            }],
            max_tokens: None,
            temperature: None,
            stream: false,
            plugins: Vec::new(),
            forced_provider: None,
            tags: Vec::new(),
        }
    }

    #[tokio::test]
    async fn tags_on_keyword_match() {
        let classifier = KeywordClassifier;
        let ctx = ClassifierContext {
            settings: json!({"tags": {"vision": ["image", "photo"], "nsfw": ["banned"]}})
                .as_object()
                .unwrap()
                .clone(),
        };

        let req = request(None, "Can you describe this Image for me?");
        let tags = classifier.classify(&ctx, &req).await.unwrap();
        assert_eq!(tags, vec!["vision".to_string()]);
    }

    #[tokio::test]
    async fn tags_video_requests() {
        let classifier = KeywordClassifier;
        let ctx = ClassifierContext {
            settings: json!({"tags": {
                "vision": ["image", "photo", "picture", "screenshot"],
                "video": ["video", "clip", "footage"],
            }})
            .as_object()
            .unwrap()
            .clone(),
        };

        let req = request(None, "Can you summarize this video clip for me?");
        let tags = classifier.classify(&ctx, &req).await.unwrap();
        assert_eq!(tags, vec!["video".to_string()]);
    }

    #[tokio::test]
    async fn no_match_returns_empty() {
        let classifier = KeywordClassifier;
        let ctx = ClassifierContext {
            settings: json!({"tags": {"vision": ["image", "photo"]}})
                .as_object()
                .unwrap()
                .clone(),
        };

        let req = request(None, "What's the weather like today?");
        let tags = classifier.classify(&ctx, &req).await.unwrap();
        assert!(tags.is_empty());
    }

    #[tokio::test]
    async fn no_config_returns_empty() {
        let classifier = KeywordClassifier;
        let ctx = ClassifierContext {
            settings: Default::default(),
        };

        let req = request(None, "anything");
        let tags = classifier.classify(&ctx, &req).await.unwrap();
        assert!(tags.is_empty());
    }
}
