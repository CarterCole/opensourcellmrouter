//! `response-healing`: best-effort repair of malformed JSON in a model's
//! reply (markdown code fences, leading/trailing prose, trailing commas,
//! truncated/unbalanced brackets).
//!
//! Only touches the response if its content isn't already valid JSON and a
//! repair attempt produces valid JSON; plain prose responses are left
//! untouched.

use async_trait::async_trait;
use serde_json::Value;

use super::{Flow, Plugin, PluginContext};
use crate::canonical::{ChatRequest, ChatResponse};

pub struct ResponseHealingPlugin;

#[async_trait]
impl Plugin for ResponseHealingPlugin {
    fn id(&self) -> &'static str {
        "response-healing"
    }

    async fn post_response(
        &self,
        _ctx: &PluginContext,
        _req: &ChatRequest,
        resp: &mut Option<ChatResponse>,
    ) -> anyhow::Result<Flow> {
        let Some(resp) = resp else {
            return Ok(Flow::Continue);
        };

        if serde_json::from_str::<Value>(&resp.content).is_ok() {
            return Ok(Flow::Continue);
        }

        if let Some(healed) = heal_json(&resp.content) {
            resp.content = healed;
            return Ok(Flow::Modified);
        }

        Ok(Flow::Continue)
    }
}

/// Attempts to turn `input` into valid JSON. Returns `None` if no repair
/// produces something `serde_json` accepts.
fn heal_json(input: &str) -> Option<String> {
    let text = strip_code_fence(input.trim());

    // Trim anything before the first `{`/`[` and after the matching last
    // `}`/`]`, dropping any prose the model wrote around the JSON.
    let start = text.find(['{', '[']);
    let end = text.rfind(['}', ']']);
    let text = match (start, end) {
        (Some(start), Some(end)) if end >= start => &text[start..=end],
        _ => text,
    };

    let candidate = balance_brackets(&remove_trailing_commas(text));

    serde_json::from_str::<Value>(&candidate)
        .ok()
        .map(|_| candidate)
}

/// Strips a surrounding ```` ```json ... ``` ```` (or plain ` ``` `) fence,
/// if present.
fn strip_code_fence(text: &str) -> &str {
    let Some(rest) = text.strip_prefix("```") else {
        return text;
    };
    let rest = rest.strip_prefix("json").unwrap_or(rest);
    let rest = rest.trim_start_matches(['\r', '\n']);
    match rest.rfind("```") {
        Some(end) => rest[..end].trim(),
        None => rest.trim(),
    }
}

/// Removes commas that are immediately followed (modulo whitespace) by a
/// closing `}` or `]`.
fn remove_trailing_commas(input: &str) -> String {
    let mut out = String::with_capacity(input.len());
    let mut chars = input.chars().peekable();

    while let Some(c) = chars.next() {
        if c == ',' {
            let mut lookahead = chars.clone();
            let mut drop_comma = false;
            while let Some(&next) = lookahead.peek() {
                if next.is_whitespace() {
                    lookahead.next();
                    continue;
                }
                drop_comma = next == '}' || next == ']';
                break;
            }
            if drop_comma {
                continue;
            }
        }
        out.push(c);
    }

    out
}

/// Appends closing brackets/braces/quotes for anything left open, to handle
/// responses truncated by a token limit.
fn balance_brackets(input: &str) -> String {
    let mut stack = Vec::new();
    let mut in_string = false;
    let mut escaped = false;

    for c in input.chars() {
        if in_string {
            match (escaped, c) {
                (false, '\\') => escaped = true,
                (false, '"') => in_string = false,
                _ => escaped = false,
            }
            continue;
        }

        match c {
            '"' => in_string = true,
            '{' => stack.push('}'),
            '[' => stack.push(']'),
            '}' | ']' if stack.last() == Some(&c) => {
                stack.pop();
            }
            _ => {}
        }
    }

    let mut out = input.to_string();
    if in_string {
        out.push('"');
    }
    while let Some(c) = stack.pop() {
        out.push(c);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn passes_through_plain_text() {
        assert_eq!(heal_json("Hello, I'm here."), None);
    }

    #[test]
    fn strips_markdown_fence() {
        let input = "```json\n{\"a\": 1}\n```";
        assert_eq!(heal_json(input), Some("{\"a\": 1}".to_string()));
    }

    #[test]
    fn strips_surrounding_prose() {
        let input = "Sure, here's the JSON:\n{\"a\": 1}\nLet me know if you need anything else.";
        assert_eq!(heal_json(input), Some("{\"a\": 1}".to_string()));
    }

    #[test]
    fn removes_trailing_commas() {
        let input = "{\"a\": 1, \"b\": [1, 2,],}";
        let healed = heal_json(input).expect("should heal");
        assert!(serde_json::from_str::<Value>(&healed).is_ok());
    }

    #[test]
    fn balances_truncated_output() {
        let input = "{\"a\": 1, \"b\": [1, 2";
        let healed = heal_json(input).expect("should heal");
        assert!(serde_json::from_str::<Value>(&healed).is_ok());
    }

    #[test]
    fn already_valid_json_is_unchanged_upstream() {
        // post_response short-circuits before calling heal_json at all, but
        // heal_json itself should also be a no-op on valid input.
        let input = "{\"a\": 1}";
        assert_eq!(heal_json(input), Some(input.to_string()));
    }
}
